"""Shared accumulation for sampling policies that select complete rollout groups."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, MutableMapping, cast

from skyrl_train.trajectory_runners.base import TrajectoryBatch
from skyrl_train.trajectory_runners.trajectory_processing import concatenate_trajectory_batches
from skyrl_train.trajectory_runners.trajectory_reward_shaping import refresh_trajectory_reward_shaping_metrics


@dataclass(frozen=True)
class SelectedGroupBatch:
    trajectory_batch: TrajectoryBatch
    uids: list[str]
    keep_sampling: bool


def filter_trajectory_batch(output: TrajectoryBatch, kept_indices: list[int]) -> TrajectoryBatch:
    """Select aligned trajectory rows while preserving batch-level metadata."""
    row_count = len(output["response_ids"])
    filtered = {}
    for key, value in output.items():
        if isinstance(value, list) and len(value) == row_count:
            filtered[key] = [deepcopy(value[index]) for index in kept_indices]
        else:
            filtered[key] = value
    refresh_trajectory_reward_shaping_metrics(filtered)
    return cast(TrajectoryBatch, filtered)


def _rekey_uid_collisions(uids: list[str], collected_uids: list[str], sample_batch_count: int) -> list[str]:
    """Keep later draws of a dataset row distinct from previously collected groups."""
    occupied_uids = set(collected_uids)
    remapped_uids: dict[str, str] = {}
    for uid in dict.fromkeys(uids):
        candidate = uid
        collision_index = 0
        while candidate in occupied_uids:
            collision_index += 1
            candidate = f"{uid}:sample_batch_{sample_batch_count}:{collision_index}"
        remapped_uids[uid] = candidate
        occupied_uids.add(candidate)
    return [remapped_uids[uid] for uid in uids]


def accumulate_selected_groups(
    trajectory_batch: TrajectoryBatch,
    uids: list[str],
    selected_uids: list[str],
    *,
    target_group_count: int,
    sample_batch_count: int,
    tis_lcs_alert_threshold: float,
    require_rollout_logprobs: bool,
    state: MutableMapping[str, Any],
) -> SelectedGroupBatch:
    """Bank selected groups and return an exact-size batch when enough are available."""
    selected_uid_set = set(selected_uids)
    selected_indices = [index for index, uid in enumerate(uids) if uid in selected_uid_set]
    batch_uids = [uids[index] for index in selected_indices]
    collected_uids = state.setdefault("collected_uids", [])
    if collected_uids:
        batch_uids = _rekey_uid_collisions(batch_uids, collected_uids, sample_batch_count)

    if selected_indices:
        selected_batch = filter_trajectory_batch(trajectory_batch, selected_indices)
        collected_batch = state.get("collected_trajectory_batch")
        state["collected_trajectory_batch"] = (
            selected_batch
            if collected_batch is None
            else concatenate_trajectory_batches(
                [collected_batch, selected_batch],
                require_rollout_logprobs=require_rollout_logprobs,
                tis_lcs_alert_threshold=tis_lcs_alert_threshold,
            )
        )
        collected_uids.extend(batch_uids)

    unique_uids = list(dict.fromkeys(collected_uids))
    state["num_prompts_in_batch"] = len(unique_uids)
    if len(unique_uids) < target_group_count:
        return SelectedGroupBatch(trajectory_batch, uids, True)

    admitted_uids = set(unique_uids[:target_group_count])
    admitted_indices = [index for index, uid in enumerate(collected_uids) if uid in admitted_uids]
    collected_batch = state.get("collected_trajectory_batch")
    assert collected_batch is not None
    return SelectedGroupBatch(
        filter_trajectory_batch(collected_batch, admitted_indices),
        [collected_uids[index] for index in admitted_indices],
        False,
    )
