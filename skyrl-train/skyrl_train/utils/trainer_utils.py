from typing import List, Dict, Any, Union, Callable, Optional
from omegaconf import OmegaConf, DictConfig
from enum import Enum
import ray
from skyrl_train.workers.worker import PPORayActorGroup
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
import os
from loguru import logger
import json
import torch
from torch.utils.data import Dataset, Subset
from skyrl_train.trajectory_runners.trajectory_processing import (
    get_metrics_from_trajectory_batch,
)
from skyrl_train.trajectory_runners.base import TrajectoryBatch
from transformers import AutoTokenizer
from skyrl_train.io import io
from marinskyrl.resource_locator import join_resource_path
from skyrl_train.checkpoint_listing import extract_step_from_path, list_checkpoint_dirs
from marinskyrl.checkpoint_paths import GLOBAL_STEP_PREFIX
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
    dump_dir_path: str,
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
        filename = join_resource_path(dump_dir_path, f"{sanitized_data_source}.jsonl")

        with io.open_file(filename, "w") as f:
            for i in indices:
                entry = {
                    "input_prompt": input_prompts[i],
                    "output_response": output_responses[i],
                    "score": trajectory_batch["rewards"][i],
                    "stop_reason": trajectory_batch.get("stop_reasons", [None] * len(input_prompts))[i],
                    "exception_type": (trajectory_batch.get("exception_types") or [None] * len(input_prompts))[i],
                    "error_treatment": (trajectory_batch.get("error_treatments") or [None] * len(input_prompts))[i],
                    "env_class": concat_all_envs[i],
                    "env_extras": concat_env_extras[i],
                    "data_source": data_source,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.info(f"Dumped eval data for {data_source} to {filename}")

    # Dump aggregated results file
    aggregated_filename = join_resource_path(dump_dir_path, "aggregated_results.jsonl")
    with io.open_file(aggregated_filename, "w") as f:
        f.write(json.dumps(eval_metrics, ensure_ascii=False) + "\n")

    logger.info(f"Dumped aggregated eval metrics to {aggregated_filename}")


def _evaluation_dataset(dataset: PromptDataset, num_prompts: int | None, seed: int) -> Dataset:
    if num_prompts is None:
        return dataset
    if num_prompts <= 0:
        raise ValueError("trainer.eval_num_prompts must be positive")
    if num_prompts >= len(dataset):
        return dataset

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:num_prompts].tolist()
    return Subset(dataset, indices)


def build_eval_dataloader(cfg: DictConfig, dataset: PromptDataset) -> StatefulDataLoader:
    """Build the evaluation dataloader over ``trainer.eval_num_prompts`` seeded prompts, or the whole dataset."""
    dataloader = StatefulDataLoader(
        _evaluation_dataset(dataset, cfg.trainer.eval_num_prompts, cfg.trainer.seed),
        batch_size=cfg.trainer.eval_batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        # Items are in-memory row lookups; worker processes would cost more to start than they save.
        num_workers=0,
        drop_last=False,
    )
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
