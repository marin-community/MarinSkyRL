from copy import deepcopy
from typing import List, Dict, Any, Union, Callable, Optional, TypedDict, cast
from dataclasses import dataclass
from omegaconf import OmegaConf, DictConfig
from enum import Enum
import ray
from skyrl_train.workers.worker import PPORayActorGroup
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
import os
from loguru import logger
import json
import torch
import numpy as np
from collections import defaultdict
from skyrl_train.dynamic_sampling import (
    DynamicSamplingCriteria,
    DynamicSamplingType,
    group_is_informative_for_dynamic_sampling,
)
from skyrl_train.group_admission import (
    AdmissionRejection,
    GroupAdmissionPolicy,
    GroupAdvantageInvariant,
    TrainingGroupInvariantError,
)
from skyrl_train.trajectory_runners.trajectory_processing import (
    get_metrics_from_trajectory_batch,
    concatenate_trajectory_batches,
)
from skyrl_train.trajectory_runners.base import TrajectoryBatch
from skyrl_train.trajectory_runners.trajectory_reward_shaping import (
    REWARD_SHAPING_ROW_KEYS,
    refresh_trajectory_reward_shaping_metrics,
)
from transformers import AutoTokenizer
from pathlib import Path
from skyrl_train.io import io
from skyrl_train.checkpoint_listing import extract_step_from_path, list_checkpoint_dirs
from marinskyrl.checkpoint_paths import GLOBAL_STEP_PREFIX
from skyrl_train.curriculum import CurriculumConfig, CurriculumSampler
from skyrl_train.dataset import PromptDataset
from torchdata.stateful_dataloader import StatefulDataLoader

BasicType = Union[int, float, str, bool, type(None)]


class ResumeMode(Enum):
    NONE = "none"
    LATEST = "latest"
    FROM_PATH = "from_path"

    @classmethod
    def _missing_(cls, value):
        if value is None:
            return cls.NONE
        return super()._missing_(value)


def get_node_ids(
    policy_model: PPORayActorGroup, critic_model: Optional[PPORayActorGroup], ref_model: Optional[PPORayActorGroup]
) -> List[str]:
    """Get the node ids of the policy, critic, and ref models.

    Args:
        policy_model: Policy model actor group
        critic_model: Critic model actor group (Optional)
        ref_model: Ref model actor group (Optional)
    """
    policy_node_ids: List[str] = ray.get(policy_model.async_run_ray_method("pass_through", "get_ray_node_id"))
    if critic_model is not None:
        critic_node_ids: List[str] = ray.get(critic_model.async_run_ray_method("pass_through", "get_ray_node_id"))
    else:
        critic_node_ids = []
    if ref_model is not None:
        ref_node_ids: List[str] = ray.get(ref_model.async_run_ray_method("pass_through", "get_ray_node_id"))
    else:
        ref_node_ids = []

    unique_node_ids = list(set(policy_node_ids + critic_node_ids + ref_node_ids))
    return unique_node_ids


def run_on_each_node(node_ids: List[str], fn: Callable, *args, **kwargs):
    """Simple helper to run a function on each node.

    Args:
        node_ids: List of node ids to run the function on
        fn: Function to run
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
    """
    node_ids = list(set(node_ids))
    task = ray.remote(num_cpus=0.25)(fn)
    refs = []

    for node_id in node_ids:
        node_task = task.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            )
        )
        refs.append(node_task.remote(*args, **kwargs))

    return ray.get(refs)


def cleanup_old_checkpoints(
    checkpoint_base_path: str, max_checkpoints: int, protected_steps: set[int] | None = None
) -> None:
    """
    Keep the most recent `max_checkpoints` and protected checkpoints; remove the rest.

    Args:
        checkpoint_base_path: Base path where checkpoints are stored
        max_checkpoints: Maximum number of recent checkpoints to keep
        protected_steps: Additional checkpoint steps retained until external work completes
    """
    if max_checkpoints < 0:
        return

    checkpoint_dirs = list_checkpoint_dirs(checkpoint_base_path)

    if len(checkpoint_dirs) <= max_checkpoints:
        return

    # Sort by step number (extract number from global_step_N)
    def extract_step(dirname):
        try:
            return int(dirname.split("global_step_")[1])
        except (IndexError, ValueError):
            return 0

    checkpoint_dirs.sort(key=extract_step)

    protected_steps = protected_steps or set()
    recent = set(checkpoint_dirs[-max_checkpoints:]) if max_checkpoints > 0 else set()
    dirs_to_remove = [
        directory
        for directory in checkpoint_dirs
        if directory not in recent and extract_step(directory) not in protected_steps
    ]

    for dir_name in dirs_to_remove:
        full_path = os.path.join(checkpoint_base_path, dir_name)
        try:
            io.remove(full_path)
            step_num = extract_step(dir_name)
            logger.info(f"Cleaned up old checkpoint: global_step_{step_num} at {full_path}")
        except Exception as e:
            logger.warning(f"Failed to remove old checkpoint {full_path}: {e}")


def validate_consistency_for_latest_checkpoint(
    root_ckpt_folder: str, ckpt_iteration: int, checkpoint_path: str, latest_checkpoint_file: str, save_interval: int
):
    """Validate that the checkpoint folder is consistent with the latest checkpoint file.

    Asserts that the folder with the highest global step is the latest checkpoint tracked by `latest_checkpoint_file`.
    Otherwise, the folder state is inconsistent and the user should delete other checkpoints.
    """
    if io.exists(root_ckpt_folder):
        checkpoint_dirs = list_checkpoint_dirs(root_ckpt_folder)
        if checkpoint_dirs:
            global_step_values = [extract_step_from_path(d) for d in checkpoint_dirs]
            max_global_step_in_folder = max(global_step_values)
            # NOTE (sumanthrh): We allow a checkpoint folder to be `save_interval` steps ahead of the latest checkpoint in `latest_checkpoint_file`. This is because the last checkpoint can be an incomplete checkpoint.
            if max_global_step_in_folder - ckpt_iteration > save_interval:
                max_global_step_in_folder_path = os.path.join(
                    root_ckpt_folder, f"{GLOBAL_STEP_PREFIX}{max_global_step_in_folder}"
                )
                raise ValueError(
                    f"Inconsistent checkpoint folder. Latest checkpoint file {latest_checkpoint_file} points to {ckpt_iteration}, but the folder has checkpoints with higher global step - Found global steps {max_global_step_in_folder_path}. This is likely because checkpoint {max_global_step_in_folder_path} was created in a previous run while the latest run is at {checkpoint_path}. Please delete/move checkpoints from older runs and try again."
                )


def sanitize_data_source(data_source: str) -> str:
    """Sanitize data source name for use in file paths."""
    if data_source is None:
        return "unknown"
    return data_source.replace("/", "_")


def calculate_per_dataset_metrics(
    trajectory_batch: TrajectoryBatch,
    concat_uids: List[str],
    concat_data_sources: List[str],
    n_samples_per_prompt: int,
) -> Dict[str, float]:
    """Calculate metrics per data source."""
    eval_metrics = {}

    # Group indices by data source
    data_source_indices = {}
    for i, data_source in enumerate(concat_data_sources):
        if data_source is None:
            data_source = "unknown"
        if data_source not in data_source_indices:
            data_source_indices[data_source] = []
        data_source_indices[data_source].append(i)

    # Calculate metrics for each data source
    for data_source, indices in data_source_indices.items():
        # Extract subset for this data source
        subset_trajectory_batch = {
            key: [value[i] for i in indices] for key, value in trajectory_batch.items() if isinstance(value, list)
        }
        subset_uids = [concat_uids[i] for i in indices]

        # Calculate metrics for this subset
        avg_score, pass_at_n = get_metrics_from_trajectory_batch(subset_trajectory_batch, subset_uids)

        # Add to eval metrics with proper naming
        sanitized_data_source = sanitize_data_source(data_source)
        eval_metrics[f"eval/{sanitized_data_source}/avg_score"] = avg_score
        eval_metrics[f"eval/{sanitized_data_source}/pass_at_{n_samples_per_prompt}"] = pass_at_n

    return eval_metrics


def dump_per_dataset_eval_results(
    dump_dir_path: Path,
    tokenizer: AutoTokenizer,
    trajectory_batch: TrajectoryBatch,
    concat_data_sources: List[str],
    concat_all_envs: List[str],
    concat_env_extras: List[Dict[str, Any]],
    eval_metrics: Dict[str, float],
):
    """Dump evaluation results per dataset and overall aggregated results."""

    # Prepare common data
    input_prompts = [tokenizer.decode(prompt) for prompt in trajectory_batch["prompt_token_ids"]]
    output_responses = [tokenizer.decode(response) for response in trajectory_batch["response_ids"]]

    # Group indices by data source
    data_source_indices = {}
    for i, data_source in enumerate(concat_data_sources):
        if data_source is None:
            data_source = "unknown"
        if data_source not in data_source_indices:
            data_source_indices[data_source] = []
        data_source_indices[data_source].append(i)

    # Dump per-dataset files
    for data_source, indices in data_source_indices.items():
        sanitized_data_source = sanitize_data_source(data_source)
        filename = dump_dir_path / f"{sanitized_data_source}.jsonl"

        with open(filename, "w") as f:
            for i in indices:
                entry = {
                    "input_prompt": input_prompts[i],
                    "output_response": output_responses[i],
                    "score": trajectory_batch["rewards"][i],
                    "stop_reason": trajectory_batch.get("stop_reasons", [None] * len(input_prompts))[i],
                    "env_class": concat_all_envs[i],
                    "env_extras": concat_env_extras[i],
                    "data_source": data_source,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.info(f"Dumped eval data for {data_source} to {filename}")

    # Dump aggregated results file
    aggregated_filename = dump_dir_path / "aggregated_results.jsonl"
    with open(aggregated_filename, "w") as f:
        f.write(json.dumps(eval_metrics, ensure_ascii=False) + "\n")

    logger.info(f"Dumped aggregated eval metrics to {aggregated_filename}")


class DynamicSamplingState(TypedDict, total=False):
    """Schema for dynamic sampling state dictionary.

    Fields:
        sample_batch_count: Counter for the number of sample batches processed
        collected_trajectory_batch: Accumulated trajectory batch (filter strategy only)
        collected_uids: Accumulated UIDs (filter strategy only)
        num_prompts_in_batch: Number of prompts collected so far (filter strategy only)
    """

    sample_batch_count: int
    collected_trajectory_batch: Optional[TrajectoryBatch]
    collected_uids: Optional[List[str]]
    num_prompts_in_batch: Optional[int]


class GroupAdmissionSamplingState(TypedDict, total=False):
    """Accepted synchronous groups collected while replacements are generated."""

    sample_batch_count: int
    collected_trajectory_batch: Optional[TrajectoryBatch]
    collected_uids: List[str]
    num_prompts_in_batch: int
    rejection_counts: Dict[str, int]
    inspected_count: int


@dataclass(frozen=True)
class DynamicSamplingResult:
    trajectory_batch: TrajectoryBatch
    uids: List[str]
    keep_sampling: bool
    state: Optional[DynamicSamplingState]


@dataclass(frozen=True)
class GroupAdmissionSamplingResult:
    trajectory_batch: TrajectoryBatch
    uids: List[str]
    keep_sampling: bool
    state: Optional[GroupAdmissionSamplingState]
    rejection_counts: Dict[AdmissionRejection, int]
    inspected_count: int


@dataclass(frozen=True)
class _AdmissionAccumulation:
    trajectory_batch: TrajectoryBatch
    uids: List[str]
    keep_sampling: bool
    state: Optional[GroupAdmissionSamplingState]


def handle_group_admission_sampling(
    trajectory_batch: TrajectoryBatch,
    uids: List[str],
    *,
    invariant: GroupAdvantageInvariant,
    rollout_logprobs_required: bool,
    target_batch_size: int,
    tis_lcs_alert_threshold: float,
    collected_state: GroupAdmissionSamplingState,
) -> GroupAdmissionSamplingResult:
    """Drop retryable sync groups and collect a complete replacement batch."""
    policy = GroupAdmissionPolicy(
        invariant,
        max_staleness_steps=0,
        rollout_logprobs_required=rollout_logprobs_required,
    )
    admissions = policy.evaluate_batch(trajectory_batch, uids, global_step=0)
    retryable = {AdmissionRejection.FULLY_MASKED, AdmissionRejection.BELOW_MINIMUM_GROUP_SIZE}
    rejection_counts = {rejection: 0 for rejection in AdmissionRejection}
    kept_uids = []
    for admission in admissions:
        if admission.decision.accepted:
            kept_uids.append(admission.uid)
            continue
        assert admission.decision.primary_rejection is not None
        rejection_counts[admission.decision.primary_rejection] += 1
        if any(rejection not in retryable for rejection in admission.decision.rejections):
            raise TrainingGroupInvariantError(
                uid=admission.uid,
                decision=admission.decision,
                physical_count=admission.physical_count,
                expected_physical_count=invariant.physical_group_size,
                row_indices=admission.row_indices,
            )

    accumulated_rejections = collected_state.setdefault("rejection_counts", {})
    for rejection, count in rejection_counts.items():
        accumulated_rejections[rejection.value] = accumulated_rejections.get(rejection.value, 0) + count
    collected_state["inspected_count"] = collected_state.get("inspected_count", 0) + len(admissions)

    if (
        len(kept_uids) == target_batch_size
        and collected_state.get("collected_trajectory_batch") is None
        and not any(accumulated_rejections.values())
    ):
        return GroupAdmissionSamplingResult(
            trajectory_batch=trajectory_batch,
            uids=uids,
            keep_sampling=False,
            state=None,
            rejection_counts={rejection: 0 for rejection in AdmissionRejection},
            inspected_count=collected_state["inspected_count"],
        )

    accumulated = _accumulate_admitted_groups(
        trajectory_batch,
        uids,
        kept_uids,
        target_batch_size=target_batch_size,
        tis_lcs_alert_threshold=tis_lcs_alert_threshold,
        require_rollout_logprobs=rollout_logprobs_required,
        collected_state=collected_state,
    )
    return GroupAdmissionSamplingResult(
        trajectory_batch=accumulated.trajectory_batch,
        uids=accumulated.uids,
        keep_sampling=accumulated.keep_sampling,
        state=accumulated.state,
        rejection_counts={
            rejection: accumulated_rejections.get(rejection.value, 0) for rejection in AdmissionRejection
        },
        inspected_count=collected_state["inspected_count"],
    )


def handle_dynamic_sampling(
    trajectory_batch: TrajectoryBatch,
    uids: List[str],
    sampling_config: Dict[str, Any],
    collected_state: Optional[DynamicSamplingState] = None,
) -> DynamicSamplingResult:
    """
    Handle dynamic sampling with different strategies (filter, replace).

    filter (used in DAPO) - filter out groups with std == 0 and group size > 1 and resample until we have enough prompts
    replace (used in POLARIS, WebSailor) - replace bad (std == 0) samples with good (std > 0) samples

    Args:
        trajectory_batch: Current trajectory batch
        uids: Current batch UIDs
        sampling_config: Configuration dict with sampling parameters
        collected_state: State for accumulating data across batches (for filter strategy)

    Returns:
        The processed batch, UIDs, continuation decision, and updated state.
    """
    sampling_type_value = sampling_config.get("type", None)

    if sampling_type_value is None:
        return DynamicSamplingResult(trajectory_batch, uids, False, None)

    try:
        sampling_type = DynamicSamplingType(sampling_type_value)
    except ValueError:
        raise ValueError(f"Invalid dynamic sampling type: {sampling_type_value}") from None

    if sampling_type is DynamicSamplingType.REPLACE:
        return handle_replace_sampling(trajectory_batch, uids, sampling_config)
    if sampling_type is DynamicSamplingType.FILTER:
        return handle_filter_sampling(trajectory_batch, uids, sampling_config, collected_state)
    raise AssertionError(f"unhandled dynamic sampling type: {sampling_type}")


def handle_replace_sampling(
    trajectory_batch: TrajectoryBatch, uids: List[str], sampling_config: Dict[str, Any]
) -> DynamicSamplingResult:
    """
    Handle replace sampling strategy based on POLARIS implementation

    Reference: https://github.com/ChenxinAn-fdu/POLARIS/blob/8c82adb16b8e45c1a34f6d0e23e35deb66dd1ae7/verl/verl/trainer/ppo/ray_trainer.py#L995-L1022.

    Args:
        trajectory_batch: Current trajectory batch
        uids: Current batch UIDs
        sampling_config: Configuration dict with sampling parameters
    Returns:
        The processed batch, UIDs, and continuation decision.
    """
    n_samples_per_prompt = sampling_config["n_samples_per_prompt"]
    min_replace_ratio = sampling_config["min_replace_ratio"]

    # Extract rewards and convert to sequence-level if needed
    rewards_list = trajectory_batch["rewards"]
    if rewards_list and isinstance(rewards_list[0], list):
        # Token-level rewards: sum to get sequence rewards
        rewards = np.array([sum(r) for r in rewards_list])
    else:
        rewards = np.array(rewards_list)

    # get mapping of uids to list of indices and metrics
    uid2indices = defaultdict(list)
    uid2metric_vals = defaultdict(list)
    for idx, uid in enumerate(uids):
        uid2indices[uid].append(idx)
        uid2metric_vals[uid].append(rewards[idx])

    # Group by UID and calculate metrics
    uid2metric_std = {}
    for uid, metric_vals in uid2metric_vals.items():
        uid2metric_std[uid] = np.std(metric_vals)

    # Determine good UIDs: those with std > 0 (or group size == 1)
    good_uids = set([uid for uid, std in uid2metric_std.items() if std > 0 or n_samples_per_prompt == 1])
    bad_uids = set([uid for uid, std in uid2metric_std.items() if std == 0 and n_samples_per_prompt > 1])

    logger.info(f"Replace sampling: {len(good_uids)} good UIDs out of {len(uid2metric_vals)} total prompts")

    # Check if we have enough good UIDs (more than min_replace_ratio of the batch)
    if len(good_uids) > len(uid2metric_vals) * min_replace_ratio:
        logger.info("============= Dynamic sampling replace ===========")
        logger.info(f"Number of good prompts: {len(good_uids)}")
        logger.info(f"Number of bad prompts: {len(bad_uids)}")

        # Get good uids to replace the bad uids (length of bad uids)
        replacement_uids = get_bad_sample_replacements(good_uids, bad_uids)  # uids to replace the bad uids
        # get replacement indices
        replacement_indices = []
        for uid in replacement_uids:
            replacement_indices.extend(uid2indices[uid])
        # get bad indices
        bad_indices = []
        for uid in bad_uids:
            bad_indices.extend(uid2indices[uid])

        # Replace bad samples with good ones (modify in place because replacement_idx and bad_idx should not overlap)
        for bad_idx, replacement_idx in zip(bad_indices, replacement_indices):
            trajectory_batch["prompt_token_ids"][bad_idx] = trajectory_batch["prompt_token_ids"][replacement_idx].copy()
            trajectory_batch["response_ids"][bad_idx] = trajectory_batch["response_ids"][replacement_idx].copy()
            replacement_reward = trajectory_batch["rewards"][replacement_idx]
            trajectory_batch["rewards"][bad_idx] = (
                replacement_reward.copy() if isinstance(replacement_reward, list) else replacement_reward
            )
            if trajectory_batch.get("unshaped_rewards") is not None:
                trajectory_batch["unshaped_rewards"][bad_idx] = trajectory_batch["unshaped_rewards"][replacement_idx]
            trajectory_batch["loss_masks"][bad_idx] = trajectory_batch["loss_masks"][replacement_idx].copy()
            if trajectory_batch["stop_reasons"]:
                trajectory_batch["stop_reasons"][bad_idx] = trajectory_batch["stop_reasons"][replacement_idx]

            if trajectory_batch["rollout_logprobs"]:
                trajectory_batch["rollout_logprobs"][bad_idx] = trajectory_batch["rollout_logprobs"][replacement_idx]
            for key in REWARD_SHAPING_ROW_KEYS:
                if trajectory_batch.get(key) is not None:
                    trajectory_batch[key][bad_idx] = deepcopy(trajectory_batch[key][replacement_idx])

        # Update UIDs accordingly
        replaced_uids = uids.copy()
        for bad_idx, replacement_idx in zip(bad_indices, replacement_indices):
            replaced_uids[bad_idx] = uids[replacement_idx]

        logger.info(f"After replacement - Replaced {len(bad_indices) // n_samples_per_prompt} bad prompts")
        logger.info("==================================================")
        refresh_trajectory_reward_shaping_metrics(trajectory_batch)

        return DynamicSamplingResult(trajectory_batch, replaced_uids, False, None)
    else:
        logger.warning("===================== Warning (Dynamic sampling replace) ====================")
        logger.warning("In this mini-batch, most training samples receive low variance rewards.")
        logger.warning("If you continue to see this warning, please check your data difficulty distribution.")
        logger.warning("==================================================")

        return DynamicSamplingResult(trajectory_batch, uids, True, None)


def _rekey_collected_uid_collisions(uids: List[str], collected_uids: List[str], sample_batch_count: int) -> List[str]:
    """Keep later draws of a dataset row distinct from groups collected in earlier sampling rounds."""
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


def _accumulate_admitted_groups(
    trajectory_batch: TrajectoryBatch,
    uids: List[str],
    kept_uids: List[str],
    *,
    target_batch_size: int,
    tis_lcs_alert_threshold: float,
    require_rollout_logprobs: bool,
    collected_state: GroupAdmissionSamplingState,
) -> _AdmissionAccumulation:
    kept_uid_set = set(kept_uids)
    kept_indices = [index for index, uid in enumerate(uids) if uid in kept_uid_set]
    filtered_uids = [uids[index] for index in kept_indices]
    collected_uids = collected_state.get("collected_uids", [])
    if collected_uids:
        filtered_uids = _rekey_collected_uid_collisions(
            filtered_uids,
            collected_uids,
            collected_state["sample_batch_count"],
        )

    if kept_indices:
        filtered_output = filter_trajectory_batch(trajectory_batch, kept_indices)
        collected_batch = collected_state.get("collected_trajectory_batch")
        if collected_batch is None:
            collected_state["collected_trajectory_batch"] = filtered_output
        else:
            collected_state["collected_trajectory_batch"] = concatenate_trajectory_batches(
                [collected_batch, filtered_output],
                require_rollout_logprobs=require_rollout_logprobs,
                tis_lcs_alert_threshold=tis_lcs_alert_threshold,
            )
        collected_uids.extend(filtered_uids)
        collected_state["collected_uids"] = collected_uids

    collected_group_count = len(dict.fromkeys(collected_uids))
    collected_state["num_prompts_in_batch"] = collected_group_count
    if collected_group_count < target_batch_size:
        return _AdmissionAccumulation(trajectory_batch, uids, True, collected_state)

    selected_uids = set(list(dict.fromkeys(collected_uids))[:target_batch_size])
    selected_indices = [index for index, uid in enumerate(collected_uids) if uid in selected_uids]
    final_batch = collected_state.get("collected_trajectory_batch")
    assert final_batch is not None
    return _AdmissionAccumulation(
        filter_trajectory_batch(final_batch, selected_indices),
        [collected_uids[index] for index in selected_indices],
        False,
        None,
    )


def handle_filter_sampling(
    trajectory_batch: TrajectoryBatch,
    uids: List[str],
    sampling_config: Dict[str, Any],
    collected_state: Optional[DynamicSamplingState],
) -> DynamicSamplingResult:
    """
    Handle filter-based sampling strategy (like DAPO).

    Args:
        trajectory_batch: Current trajectory batch
        uids: Current batch UIDs
        sampling_config: Configuration dict with sampling parameters
        collected_state: State for accumulating data across batches

    Returns:
        The processed batch, UIDs, continuation decision, and updated state.
    """
    target_batch_size = sampling_config["train_batch_size"]

    uid2indices = defaultdict(list)
    for row_index, uid in enumerate(uids):
        uid2indices[uid].append(row_index)
    criteria = sampling_config["criteria"]
    if not isinstance(criteria, DynamicSamplingCriteria):
        raise ValueError("dynamic sampling filter requires resolved DynamicSamplingCriteria")
    kept_uids = [
        uid
        for uid, row_indices in uid2indices.items()
        if group_is_informative_for_dynamic_sampling(
            trajectory_batch,
            row_indices,
            criteria=criteria,
        )
    ]
    kept_uids_set = set(kept_uids)

    # Filter trajectories based on kept UIDs
    kept_traj_idxs = []
    for idx, traj_uid in enumerate(uids):
        if traj_uid in kept_uids_set:
            kept_traj_idxs.append(idx)

    filtered_output = filter_trajectory_batch(trajectory_batch, kept_traj_idxs)
    filtered_uids = [uids[idx] for idx in kept_traj_idxs]

    # Dataset UIDs repeat across epochs. Re-key only later draws that would merge with a collected training group.
    collected_uids = collected_state.get("collected_uids")
    if collected_uids is not None:
        filtered_uids = _rekey_collected_uid_collisions(
            filtered_uids, collected_uids, collected_state["sample_batch_count"]
        )

    if "collected_trajectory_batch" not in collected_state:
        collected_state.update(
            {
                "collected_trajectory_batch": filtered_output,
                "collected_uids": filtered_uids.copy(),
                "num_prompts_in_batch": len(kept_uids),
            }
        )
    else:
        collected_state["collected_trajectory_batch"] = concatenate_trajectory_batches(
            [collected_state["collected_trajectory_batch"], filtered_output],
            tis_lcs_alert_threshold=float(sampling_config["tis_lcs_alert_threshold"]),
        )
        collected_state["collected_uids"].extend(filtered_uids)
        collected_state["num_prompts_in_batch"] += len(kept_uids)

    # Check if we have enough prompts
    if collected_state["num_prompts_in_batch"] < target_batch_size:
        logger.info("============= Dynamic sampling filter =============")
        logger.info(f"Dynamic sampling: {collected_state['num_prompts_in_batch']} < {target_batch_size} prompts")
        logger.info(f"Resample batch {collected_state['sample_batch_count']}, continue sampling...")
        logger.info("==================================================")
        return DynamicSamplingResult(trajectory_batch, uids, True, collected_state)
    else:
        logger.info("============= Dynamic sampling filter =============")
        logger.info(
            f"Dynamic sampling: collected {collected_state['num_prompts_in_batch']} >= {target_batch_size} prompts"
        )
        logger.info("==================================================")
        # Truncate to exact batch size if needed
        n_samples_per_prompt = sampling_config.get("n_samples_per_prompt", 1)
        max_trajectories = target_batch_size * n_samples_per_prompt
        final_output = collected_state["collected_trajectory_batch"]
        final_uids = collected_state["collected_uids"]

        if len(final_uids) > max_trajectories:
            final_output = filter_trajectory_batch(final_output, list(range(max_trajectories)))
            final_uids = final_uids[:max_trajectories]

        return DynamicSamplingResult(final_output, final_uids, False, None)


def get_bad_sample_replacements(good_uids: List[str], bad_uids: List[str]) -> List[str]:
    num_replacements = len(bad_uids)
    num_candidates = len(good_uids)

    if num_candidates >= num_replacements:
        perm = np.random.permutation(num_candidates)
        chosen_replacement_uids = np.array(list(good_uids))[perm[:num_replacements]]
    else:
        indices = np.random.randint(low=0, high=num_candidates, size=(num_replacements,))
        chosen_replacement_uids = np.array(list(good_uids))[indices]

    return chosen_replacement_uids


def filter_trajectory_batch(output: TrajectoryBatch, kept_indices: List[int]) -> TrajectoryBatch:
    """Filter TrajectoryBatch based on kept indices."""
    row_count = len(output["response_ids"])
    filtered = {}
    for key, value in output.items():
        if isinstance(value, list) and len(value) == row_count:
            filtered[key] = [deepcopy(value[index]) for index in kept_indices]
        else:
            filtered[key] = value
    refresh_trajectory_reward_shaping_metrics(filtered)

    return cast(TrajectoryBatch, filtered)


def build_dataloader(
    cfg: DictConfig, dataset: PromptDataset, is_train=True, is_fully_async=False
) -> StatefulDataLoader:
    """
    Build the dataloader for the training or evaluation dataset.

    Args:
        cfg: Config object
        dataset: Dataset object
        is_train: Whether to build the dataloader for training or evaluation
        is_fully_async: If is_train, whether to build the dataloader for fully async training, which
            mainly makes the batch size 1.
    """
    # prepare dataloader
    batch_size = cfg.trainer.train_batch_size if is_train else cfg.trainer.eval_batch_size

    # Seed the dataloader for reproducibility.
    seeded_generator = torch.Generator()
    seeded_generator.manual_seed(cfg.trainer.seed)

    sampler = None
    if is_train and cfg.data.sampling.kind is not None:
        if is_fully_async:
            raise ValueError("data.sampling.kind is not supported with fully async training")
        if cfg.trainer.step_wise_training:
            raise ValueError("data.sampling.kind requires per-prompt uids; step_wise_training is not supported")
        sampling_seed = cfg.data.sampling.seed if cfg.data.sampling.seed is not None else cfg.trainer.seed
        sampler = CurriculumSampler(
            dataset,
            CurriculumConfig.from_dict_config(cfg.data.sampling, group_size=cfg.generator.n_samples_per_prompt),
            sampling_seed,
            batch_size=batch_size,
        )

    dataloader = StatefulDataLoader(
        dataset,
        batch_size=batch_size if not is_fully_async else 1,
        shuffle=is_train and sampler is None,
        sampler=sampler,
        collate_fn=dataset.collate_fn,
        # Curriculum sampling stays single-process: worker prefetch would draw several
        # batches of indices ahead of the per-step weight updates.
        # TODO(Charlie): debug why inference http endpoint is slow when num_workers is 8
        num_workers=0 if (sampler is not None or cfg.generator.enable_http_endpoint) else 8,
        drop_last=True if is_train else False,
        generator=seeded_generator,
    )
    if is_train:
        if not is_fully_async:
            logger.info(f"Total steps: {len(dataloader) * cfg.trainer.epochs}")
        else:
            logger.info(f"Total steps: {len(dataloader) // cfg.trainer.train_batch_size * cfg.trainer.epochs}")
    else:
        logger.info(f"Validation set size: {len(dataloader)}")

    return dataloader


def get_rope_scaling_config(trainer_cfg: DictConfig) -> dict[str, Any]:
    if "rope_scaling" not in trainer_cfg:
        return {}
    if trainer_cfg.rope_scaling is None:
        return None
    return OmegaConf.to_container(trainer_cfg.rope_scaling)


def get_rope_theta_config(trainer_cfg: DictConfig) -> int | None:
    if "rope_theta" not in trainer_cfg:
        return None
    return trainer_cfg.rope_theta
