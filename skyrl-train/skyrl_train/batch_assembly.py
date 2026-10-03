"""Build globally padded training tensors for a whole batch or one DP slice."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from marinskyrl.runtime_options import AdvantageEstimator

from skyrl_train.dataset.preprocess import collate_response_token_channel, convert_prompts_responses_to_batch_tensors
from skyrl_train.dataset.routed_expert_batch import RoutedExpertRows
from skyrl_train.rollouts.buffer import RolloutGroup
from skyrl_train.rollouts.context import RolloutBatchMetadata
from skyrl_train.group_admission import GroupAdvantageInvariant
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.trajectory_runners.trajectory_processing import scalar_reward_token_credit
from skyrl_train.trajectory_runners.types import TrajectoryBatch
from skyrl_train.utils.advantage_estimators import compute_advantages_and_returns


@dataclass(frozen=True)
class BatchPlan:
    """Global row order, widths and scalar metadata shared by every policy rank."""

    batch_id: int
    policy_step: int
    uids: tuple[str, ...]
    dp_size: int
    max_prompt_len: int
    max_response_len: int
    advantages: np.ndarray | None
    fields: frozenset[str]
    route_geometry: tuple[int, int, np.dtype] | None
    num_experts: int | None
    group_counts: tuple[int, ...]
    rollout_staleness: tuple[int, ...]
    metadata: dict[str, Any]

    def rank_rows(self, dp: int) -> range:
        if not 0 <= dp < self.dp_size:
            raise ValueError(f"DP rank {dp} is outside [0, {self.dp_size})")
        size, remainder = divmod(len(self.uids), self.dp_size)
        if remainder:
            raise ValueError("worker batch rows must be divisible by DP size")
        return range(dp * size, (dp + 1) * size)


def plan_batch(meta: RolloutBatchMetadata, *, dp_size: int, algorithm: DictConfig) -> BatchPlan:
    """Plan DP rows for outcome configs without a critic, KL, FTPO, distillation, trajectory selection,
    stepwise training, batch advantage normalization or loop credit, with resolved meta.moe_router_replay
    and meta.num_experts.
    """
    # trainer_utils imports policy workers.
    from skyrl_train.utils.trainer_utils import consumed_stop_metrics

    if not meta.groups:
        raise ValueError("worker batch requires admitted groups")
    facts = []
    for group in meta.groups:
        if group.row_facts is None:
            raise ValueError(f"worker batch missing RowFacts for {group.uid}; its checkpoint can use driver mode")
        if not group.row_facts.scalar_rewards:
            raise ValueError(f"worker batch requires scalar rewards: {group.uid}")
        if any(
            values.shape != (group.sample_count,)
            for values in (
                group.row_facts.prompt_len,
                group.row_facts.response_len,
                group.row_facts.score,
                group.row_facts.loss_tokens,
                group.row_facts.is_last_step,
                group.row_facts.exclude_from_baseline,
            )
        ):
            raise ValueError(f"worker batch RowFacts must align with admitted rows: {group.uid}")
        facts.append(group.row_facts)
    uids = tuple(group.uid for group in meta.groups for _ in range(group.sample_count))
    if dp_size < 1 or len(uids) % dp_size:
        raise ValueError("worker batch rows must be divisible by DP size")
    if algorithm.advantage_estimator not in (
        AdvantageEstimator.RLOO,
        AdvantageEstimator.RLOO_N,
        AdvantageEstimator.GRPO,
    ):
        raise ValueError("worker batch requires an outcome advantage estimator: rloo, rloo_n, grpo")
    fields = frozenset(key for group in facts for key in group.fields)
    if "is_last_step" not in facts[0].fields:
        fields = fields - {"is_last_step"}
    elif any("is_last_step" not in group.fields for group in facts):
        raise ValueError("worker batch requires is_last_step on every group when the first group carries it")
    if "loop_advantages" not in facts[0].fields:
        fields = fields - {"loop_advantages"}
    elif any("loop_advantages" not in group.fields for group in facts):
        raise ValueError("worker batch requires loop_advantages on every group when the first group carries it")
    route_geometry = next((group.route_geometry for group in facts if group.route_geometry is not None), None)
    if meta.moe_router_replay:
        if meta.num_experts is None:
            raise ValueError("worker batch router replay requires resolved num_experts")
        if "rollout_routed_experts" not in fields or route_geometry is None:
            raise ValueError("worker batch router replay requires rollout routes")
    else:
        fields = fields - {"rollout_routed_experts"}
        route_geometry = None
    excluded = np.concatenate([group.exclude_from_baseline for group in facts])
    scores = torch.from_numpy(np.concatenate([group.score for group in facts]))[:, None]
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
        exclude_from_baseline=excluded,
        group_advantage_invariant=GroupAdvantageInvariant.from_config(algorithm.resolved_group_advantage),
    )
    response_len = np.concatenate([group.response_len for group in facts])
    stop_reasons = None
    if "stop_reasons" in facts[0].fields:
        if any(group.stop_reasons is None for group in facts):
            raise ValueError("worker batch requires stop_reasons on every group when the first group carries them")
        stop_reasons = [reason for group in facts for reason in group.stop_reasons]
    metadata = {
        "uids": list(uids),
        "response_length": int(response_len.max()),
        "avg_response_length": int(response_len.sum()) / len(uids),
        "consumed_stop_metrics": consumed_stop_metrics(stop_reasons, len(uids)),
        "pad_size": 0,
        "response_lengths": response_len,
        "consumed_work": (
            len(uids),
            int(response_len.sum()),
            sum(int(group.loss_tokens.sum()) for group in facts),
        ),
    }
    if "exclude_from_baseline" in fields:
        metadata["exclude_from_baseline"] = excluded
    return BatchPlan(
        batch_id=meta.batch_id,
        policy_step=meta.batch_id,
        uids=uids,
        dp_size=dp_size,
        max_prompt_len=max(int(group.prompt_len.max()) for group in facts),
        max_response_len=metadata["response_length"],
        advantages=advantages[:, 0].numpy().copy(),
        fields=fields,
        route_geometry=route_geometry,
        num_experts=meta.num_experts if meta.moe_router_replay else None,
        group_counts=tuple(group.sample_count for group in meta.groups),
        rollout_staleness=tuple(
            meta.batch_id - group.policy_step for group in meta.groups for _ in range(group.sample_count)
        ),
        metadata=metadata,
    )


def assemble_slice(
    plan: BatchPlan,
    dp: int | None,
    groups: list[RolloutGroup] | TrajectoryBatch,
    *,
    pad_token_id: int,
    algorithm: DictConfig,
) -> TrainingInputBatch:
    """Assemble selected rows with the plan's whole-batch field rules and widths.

    The driver passes its postprocessed trajectory batch. Workers pass the groups
    intersecting their arithmetic row range, including groups split across ranks.
    Metadata matches TensorBatch's shared whole-batch metadata when it chunks rows.
    """
    rows = range(len(plan.uids)) if dp is None else plan.rank_rows(dp)
    if isinstance(groups, dict):
        if dp is not None:
            raise ValueError("a concatenated trajectory batch is a whole-batch input")
        batch = groups
    else:
        by_uid = {group.uid: group.trajectory_batch for group in groups}
        selected = []
        offset = 0
        for count in plan.group_counts:
            start, stop = max(offset, rows.start), min(offset + count, rows.stop)
            if start < stop:
                group = by_uid[plan.uids[offset]]
                selected.extend((group, index - offset) for index in range(start, stop))
            offset += count
        batch = {
            key: [group[key][index] for group, index in selected]
            for key in ("prompt_token_ids", "response_ids", "loss_masks", "rewards")
        }
        batch["rewards"] = [
            reward if isinstance(reward, list) else scalar_reward_token_credit(reward, response)
            for reward, response in zip(batch["rewards"], batch["response_ids"], strict=True)
        ]
        if any(
            value != 0
            for group, index in selected
            if group.get("loop_advantages") is not None
            for value in group["loop_advantages"][index]
        ):
            raise ValueError("worker batch does not support nonzero loop credit")
        for key, dtype in (
            ("rollout_logprobs", np.float32),
            ("token_level_shaping", np.float32),
            ("response_span_tags", np.int64),
            ("loop_advantages", np.float32),
        ):
            if key in plan.fields:
                batch[key] = [
                    group[key][index]
                    if group.get(key) is not None
                    else np.zeros(len(group["response_ids"][index]), dtype=dtype)
                    for group, index in selected
                ]
        if "is_last_step" in plan.fields:
            batch["is_last_step"] = [group["is_last_step"][index] for group, index in selected]
        if "rollout_routed_experts" in plan.fields:
            if plan.route_geometry is None:
                raise ValueError("worker batch routes require global route geometry")
            layers, top_k, dtype = plan.route_geometry
            batch["rollout_routed_experts"] = [
                group["rollout_routed_experts"][index]
                if group.get("rollout_routed_experts") is not None
                else np.zeros((len(group["response_ids"][index]), layers, top_k), dtype=dtype)
                for group, index in selected
            ]

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
    if "rollout_routed_experts" in plan.fields:
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
    if plan.advantages is not None:
        advantages = torch.from_numpy(plan.advantages[rows.start : rows.stop].copy())[:, None] * response
        training_input["advantages"] = advantages
        training_input["returns"] = advantages
    training_input.metadata = {
        key: value for key, value in plan.metadata.items() if key not in ("response_lengths", "consumed_work")
    }
    return training_input
