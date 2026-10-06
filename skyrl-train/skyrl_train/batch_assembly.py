"""Build globally padded training tensors from normalized trajectory rows."""

from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from omegaconf import DictConfig

from skyrl_train.batch_metrics import consumed_stop_metrics
from skyrl_train.dataset.preprocess import collate_response_token_channel, convert_prompts_responses_to_batch_tensors
from skyrl_train.dataset.routed_expert_batch import RoutedExpertRows
from skyrl_train.rollouts.buffer import RowFacts
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.trajectory_runners.types import BatchFields, TrajectoryBatch


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


def forward_input(data: TrainingInputBatch) -> TrainingInputBatch:
    """Select policy-forward tensors and metadata, retaining compact route rows."""
    keys = ["sequences", "attention_mask"]
    if data.routed_experts is not None:
        keys.append("rollout_routed_experts")
    return data.select(keys=keys, metadata_keys=["response_length", "global_step"])
