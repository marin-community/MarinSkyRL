"""Build globally padded training tensors from normalized trajectory rows."""

from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import DictConfig
from marinskyrl.runtime_options import AdvantageEstimator

from skyrl_train.batch_metrics import consumed_stop_metrics
from skyrl_train.batch_sampling import RowOwnership, filter_trajectory_batch
from skyrl_train.dataset.preprocess import collate_response_token_channel, convert_prompts_responses_to_batch_tensors
from skyrl_train.dataset.routed_expert_batch import RoutedExpertRows
from skyrl_train.group_admission import GroupAdvantageInvariant
from skyrl_train.rollouts.buffer import RolloutGroup, RowFacts
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.trajectory_runners.trajectory_processing import normalize_trajectory_batches, token_reward_rows
from skyrl_train.trajectory_runners.types import BatchFields, RolloutObservations, TrajectoryBatch
from skyrl_train.utils.advantage_estimators import compute_advantages_and_returns


@dataclass(frozen=True)
class BatchLoadResult:
    """Report one actor's load duration and its uniquely owned group observations."""

    load_seconds: float
    observations: tuple[tuple[int, RolloutObservations], ...]


@dataclass(frozen=True)
class BatchPlan:
    """Global row order, widths and scalar facts shared by every policy rank."""

    batch_id: int
    global_step: int
    uids: tuple[str, ...]
    dp_size: int
    facts: RowFacts
    fields: BatchFields
    rollout_staleness: tuple[int, ...]
    num_experts: int | None

    @property
    def max_prompt_len(self) -> int:
        return int(self.facts.prompt_len.max())

    @property
    def max_response_len(self) -> int:
        return int(self.facts.response_len.max())

    def rank_rows(self, dp: int) -> range:
        if not 0 <= dp < self.dp_size:
            raise ValueError(f"DP rank {dp} is outside [0, {self.dp_size})")
        size, remainder = divmod(len(self.uids), self.dp_size)
        if remainder:
            raise ValueError("worker batch rows must be divisible by DP size")
        return range(dp * size, (dp + 1) * size)


def fields_from_facts(facts: Sequence[RowFacts]) -> BatchFields:
    """Resolve whole-batch field rules from ordered admission facts."""
    return BatchFields.from_groups(
        [
            BatchFields(
                frozenset(group.fields),
                frozenset(group.first_group_list_fields),
                group.route_geometry,
                group.token_rewards,
            )
            for group in facts
        ]
    )


def plan_batch(
    *,
    batch_id: int,
    global_step: int,
    uids: Sequence[str],
    facts: RowFacts,
    fields: BatchFields,
    dp_size: int,
    rollout_staleness: Sequence[int],
    num_experts: int | None,
) -> BatchPlan:
    """Fix an actual batch's layout before whole-batch or per-rank assembly."""
    count = len(uids)
    if not count or dp_size < 1:
        raise ValueError("batch planning requires rows and a positive DP size")
    if (
        any(
            values.shape != (count,)
            for values in (
                facts.prompt_len,
                facts.response_len,
                facts.score,
                facts.loss_tokens,
                facts.is_last_step,
                facts.exclude_from_baseline,
            )
        )
        or len(rollout_staleness) != count
    ):
        raise ValueError("batch row facts and staleness must align with uids")
    return BatchPlan(batch_id, global_step, tuple(uids), dp_size, facts, fields, tuple(rollout_staleness), num_experts)


def outcome_advantages(facts: RowFacts, uids: Sequence[str], algorithm: DictConfig) -> np.ndarray:
    """Run the configured outcome estimator on the committed float32 row scores."""
    if not facts.scalar_rewards:
        raise ValueError("worker batch requires scalar rewards")
    if algorithm.advantage_estimator not in (
        AdvantageEstimator.RLOO,
        AdvantageEstimator.RLOO_N,
        AdvantageEstimator.GRPO,
    ):
        raise ValueError("worker batch requires an outcome advantage estimator: rloo, rloo_n, grpo")
    scores = torch.from_numpy(facts.score)[:, None]
    advantages, _ = compute_advantages_and_returns(
        token_level_rewards=scores,
        response_mask=torch.ones_like(scores, dtype=torch.int64),
        index=uids,
        adv_estimator=algorithm.advantage_estimator,
        config=algorithm,
        values=None,
        gamma=algorithm.gamma,
        lambd=algorithm.lambd,
        grpo_norm_by_std=algorithm.grpo_norm_by_std,
        exclude_from_baseline=facts.exclude_from_baseline,
        group_advantage_invariant=GroupAdvantageInvariant.from_config(algorithm.resolved_group_advantage),
    )
    return advantages[:, 0].numpy().copy()


def assemble_slice(
    plan: BatchPlan,
    rows: range,
    trajectory_batch: TrajectoryBatch,
    *,
    pad_token_id: int,
    algorithm: DictConfig,
) -> TrainingInputBatch:
    """Build normalized rows at the plan's global widths and retain whole-batch metadata."""
    batch = trajectory_batch
    if rows.step != 1 or rows.start < 0 or rows.stop > len(plan.uids) or len(rows) != len(batch["response_ids"]):
        raise ValueError("normalized rows must match their contiguous batch-plan range")
    token_channel = bool(algorithm.get("enable_token_reward_channel", False))
    sequences, attention, response, rewards, losses, logprobs, shaping, tags = (
        convert_prompts_responses_to_batch_tensors(
            SimpleNamespace(pad_token_id=pad_token_id),
            batch["prompt_token_ids"],
            batch["response_ids"],
            batch["rewards"],
            batch["loss_masks"],
            batch.get("rollout_logprobs"),
            batch.get("token_level_shaping") if token_channel else None,
            batch.get("response_span_tags") if token_channel else None,
            max_prompt_len=plan.max_prompt_len,
            max_response_len=plan.max_response_len,
        )
    )
    route_rows = None
    if "rollout_routed_experts" in plan.fields.present:
        route_rows = RoutedExpertRows(tuple(batch["rollout_routed_experts"]), plan.max_response_len, plan.num_experts)
    training_input = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": attention,
            "response_mask": response,
            "rewards": rewards,
            "loss_mask": losses,
            "rollout_logprobs": logprobs,
            "rollout_staleness": torch.tensor(plan.rollout_staleness[rows.start : rows.stop], dtype=torch.int32),
            "is_last_step": (
                torch.tensor(batch["is_last_step"], dtype=torch.bool) if batch.get("is_last_step") is not None else None
            ),
        },
        routed_expert_rows=route_rows,
    )
    if shaping is not None:
        training_input["token_level_shaping"] = shaping
    if tags is not None:
        training_input["response_span_tags"] = tags
    loop_advantages = collate_response_token_channel(
        batch.get("loop_advantages"),
        response,
        dtype=torch.float,
        expected_lengths=[len(row) for row in batch["response_ids"]],
    )
    if loop_advantages is not None:
        training_input["loop_advantages"] = loop_advantages
    facts = plan.facts
    training_input.metadata = {
        "uids": list(plan.uids),
        "response_length": plan.max_response_len,
        "avg_response_length": int(facts.response_len.sum()) / len(plan.uids),
        "consumed_stop_metrics": consumed_stop_metrics(facts.stop_reasons, len(plan.uids)),
        "pad_size": 0,
        "global_step": plan.global_step,
    }
    if "exclude_from_baseline" in plan.fields.present:
        training_input.metadata["exclude_from_baseline"] = facts.exclude_from_baseline
    return training_input


def assemble_worker_slice(
    plan: BatchPlan,
    dp: int,
    groups: Sequence[RolloutGroup],
    advantages: np.ndarray,
    *,
    pad_token_id: int,
    algorithm: DictConfig,
) -> TrainingInputBatch:
    """Select a DP rank's rows, normalize globally and install its outcome advantages."""
    rows = plan.rank_rows(dp)
    by_uid = {group.uid: group.trajectory_batch for group in groups}
    selected = []
    offset = 0
    while offset < len(plan.uids):
        uid = plan.uids[offset]
        end = offset + 1
        while end < len(plan.uids) and plan.uids[end] == uid:
            end += 1
        start, stop = max(offset, rows.start), min(end, rows.stop)
        if start < stop:
            selected.append(
                filter_trajectory_batch(
                    by_uid[uid], list(range(start - offset, stop - offset)), row_ownership=RowOwnership.BORROWED
                )
            )
        offset = end
    batch = normalize_trajectory_batches(selected, fields=plan.fields)
    batch["rewards"] = token_reward_rows(batch["rewards"], batch["response_ids"])
    if any(value != 0 for row in batch.get("loop_advantages") or () for value in row):
        raise ValueError("worker batch does not support nonzero loop credit")
    data = assemble_slice(plan, rows, batch, pad_token_id=pad_token_id, algorithm=algorithm)
    scalar_advantages = torch.from_numpy(advantages[rows.start : rows.stop].copy())[:, None]
    data["advantages"] = scalar_advantages * data["response_mask"]
    data["returns"] = data["advantages"]
    return data


def forward_input(data: TrainingInputBatch) -> TrainingInputBatch:
    """Select policy-forward tensors and metadata, retaining compact route rows."""
    keys = ["sequences", "attention_mask"]
    if data.routed_experts is not None:
        keys.append("rollout_routed_experts")
    return data.select(keys=keys, metadata_keys=["response_length", "global_step"])
