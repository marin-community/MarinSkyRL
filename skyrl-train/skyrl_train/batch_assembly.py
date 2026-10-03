"""Build globally padded training tensors for a whole batch or one DP slice."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from skyrl_train.dataset.preprocess import collate_response_token_channel, convert_prompts_responses_to_batch_tensors
from skyrl_train.dataset.routed_expert_batch import RoutedExpertRows
from skyrl_train.rollouts.buffer import RolloutGroup
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.trajectory_runners.trajectory_processing import scalar_reward_token_credit
from skyrl_train.trajectory_runners.types import TrajectoryBatch


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
        for key, dtype in (
            ("rollout_logprobs", np.float32),
            ("token_level_shaping", np.float32),
            ("response_span_tags", np.int64),
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
    training_input.metadata = dict(plan.metadata)
    return training_input
