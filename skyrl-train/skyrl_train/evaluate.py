import torch
from skyrl_train.utils.progress import tqdm
from typing import Dict, List, Any
from pathlib import Path
from loguru import logger
from collections import defaultdict

from skyrl_train.utils import Timer

from skyrl_train.trajectory_runners.utils import (
    concatenate_trajectory_batches,
    get_metrics_from_trajectory_batch,
    prepare_trajectory_request,
)
from skyrl_train.trajectory_runners.base import (
    TrajectoryBatch,
    TrajectoryRunner,
)
from skyrl_train.utils.trainer_utils import (
    calculate_per_dataset_metrics,
    dump_per_dataset_eval_results,
    validate_trajectory_batch,
)
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.utils.logging_utils import log_example
from skyrl_train.trajectory_runners.trajectory_retention import (
    TrajectorySink,
    make_trajectory_sink,
)

from omegaconf import DictConfig
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer


@torch.no_grad()
async def evaluate(
    eval_dataloader: StatefulDataLoader,
    trajectory_runner: TrajectoryRunner,
    cfg: DictConfig,
    global_step: int | None,
    tokenizer: AutoTokenizer,
    val_set_name: str | None = None,
    trajectory_sink: TrajectorySink | None = None,
) -> Dict[str, float]:
    """Runs generation and evaluation of trajectories.

    Args:
        eval_dataloader (StatefulDataLoader): dataloader of the eval dataset
        trajectory_runner (TrajectoryRunner): runner to use
        cfg (DictConfig): config
        global_step (int | None): current global step, or
            `None` to indicate a non-training context (e.g., eval-only)
        tokenizer (AutoTokenizer): tokenizer to use
        val_set_name (str | None): optional name of the validation set being evaluated,
            used for unique orchestrator naming
        trajectory_sink (TrajectorySink | None): trainer-owned retention sink; a standalone evaluation builds one from cfg

    Returns:
        Dict[str, float]: evaluation metrics
    """
    # Start a fresh eval session if the runner supports it
    # This creates a dedicated QueueOrchestrator for this eval run
    run_name = getattr(cfg.trainer, "run_name", None) or "eval"
    eval_step = global_step if global_step is not None else 0
    has_eval_session = hasattr(trajectory_runner, "start_eval_session")
    active_sink = trajectory_sink or make_trajectory_sink(cfg.generator, tokenizer)
    trajectory_runner.set_trajectory_sink(active_sink)

    if has_eval_session:
        await trajectory_runner.start_eval_session(
            run_name=run_name,
            eval_step=eval_step,
            val_set_name=val_set_name,
        )

    try:
        # 1. Get all trajectory batches
        trajectory_batches: List[TrajectoryBatch] = []
        concat_all_envs: List[str] = []
        concat_env_extras: List[Dict[str, Any]] = []
        concat_uids: List[str] = []
        sampling_params = cfg.generator.eval_sampling_params
        pbar = tqdm(total=len(eval_dataloader), initial=0, desc="Evaluation Progress")
        for _, prompts in enumerate(eval_dataloader):
            pbar.update(1)
            trajectory_request, uids = prepare_trajectory_request(
                prompts,
                cfg.generator.eval_n_samples_per_prompt,
                get_sampling_params_for_backend(cfg.generator.backend, sampling_params),
                cfg.environment.env_class,
                "eval",
                global_step,
            )
            trajectory_batch: TrajectoryBatch = await trajectory_runner.run(trajectory_request)
            validate_trajectory_batch(len(trajectory_request["prompts"]), trajectory_batch)
            trajectory_batches.append(trajectory_batch)
            concat_all_envs.extend(trajectory_request["env_classes"])
            concat_env_extras.extend(trajectory_request["env_extras"])
            concat_uids.extend(uids)
        concat_trajectory_batches: TrajectoryBatch = concatenate_trajectory_batches(trajectory_batches)
    finally:
        # Always stop the eval session, even if evaluation fails
        if has_eval_session:
            await trajectory_runner.stop_eval_session()

    # Extract data_sources from env_extras
    concat_data_sources = [env_extra.get("data_source") for env_extra in concat_env_extras]
    vis = tokenizer.decode(trajectory_batch["response_ids"][0])
    log_example(
        logger,
        prompt=trajectory_request["prompts"][0],
        response=vis,
        reward=trajectory_batch["rewards"][0],
    )

    # 2. Group data by data source and calculate per-dataset metrics
    eval_metrics = calculate_per_dataset_metrics(
        concat_trajectory_batches, concat_uids, concat_data_sources, cfg.generator.eval_n_samples_per_prompt
    )

    # 3. Calculate overall metrics across all datasets
    overall_avg_score, overall_pass_at_n = get_metrics_from_trajectory_batch(concat_trajectory_batches, concat_uids)
    eval_metrics.update(
        {
            "eval/all/avg_score": overall_avg_score,
            f"eval/all/pass_at_{cfg.generator.eval_n_samples_per_prompt}": overall_pass_at_n,
        }
    )

    # 4. Prepare dumping data
    # TODO[Ben] update this to be cloud-compatible
    if cfg.trainer.dump_eval_results:
        with Timer("dump_eval_results"):
            data_save_dir = (
                Path(cfg.trainer.export_path)
                / "dumped_evals"
                / ("eval_only" if global_step is None else f"global_step_{global_step}_evals")
            )
            data_save_dir.mkdir(parents=True, exist_ok=True)
            dump_per_dataset_eval_results(
                data_save_dir,
                tokenizer,
                concat_trajectory_batches,
                concat_data_sources,
                concat_all_envs,
                concat_env_extras,
                eval_metrics,
            )

    return eval_metrics


@torch.no_grad()
async def evaluate_step_wise(
    eval_dataloader: StatefulDataLoader,
    trajectory_runner: TrajectoryRunner,
    cfg: DictConfig,
    global_step: int | None,
    tokenizer: AutoTokenizer,
    trajectory_sink: TrajectorySink,
    val_set_name: str | None = None,
) -> Dict[str, float]:
    """Runs generation and evaluation of trajectories for step-wise training.

    Currently assumes that the rewards are assigned to the last step of each trajectory.

    Args:
        eval_dataloader (StatefulDataLoader): dataloader of the eval dataset
        trajectory_runner (TrajectoryRunner): runner to use
        cfg (DictConfig): config
        global_step (int | None): current global step, or
            `None` to indicate a non-training context (e.g., eval-only)
        tokenizer (AutoTokenizer): tokenizer to use
        val_set_name (str | None): optional name of the validation set being evaluated,
            used for unique orchestrator naming
        trajectory_sink (TrajectorySink): trainer-owned retention sink

    Returns:
        Dict[str, float]: evaluation metrics
    """
    # Start a fresh eval session if the runner supports it
    # This creates a dedicated QueueOrchestrator for this eval run
    run_name = getattr(cfg.trainer, "run_name", None) or "eval"
    eval_step = global_step if global_step is not None else 0
    has_eval_session = hasattr(trajectory_runner, "start_eval_session")
    trajectory_runner.set_trajectory_sink(trajectory_sink)

    if has_eval_session:
        await trajectory_runner.start_eval_session(
            run_name=run_name,
            eval_step=eval_step,
            val_set_name=val_set_name,
        )

    try:
        # 1. Get all trajectory batches
        trajectory_batches: List[TrajectoryBatch] = []
        concat_all_envs: List[str] = []
        concat_env_extras: List[Dict[str, Any]] = []
        concat_uids: List[str] = []
        sampling_params = cfg.generator.eval_sampling_params
        pbar = tqdm(total=len(eval_dataloader), initial=0, desc="Evaluation Progress")
        for _, prompts in enumerate(eval_dataloader):
            pbar.update(1)
            trajectory_request, uids = prepare_trajectory_request(
                prompts,
                cfg.generator.eval_n_samples_per_prompt,
                get_sampling_params_for_backend(cfg.generator.backend, sampling_params),
                cfg.environment.env_class,
                "eval",
                global_step,
            )
            trajectory_batch: TrajectoryBatch = await trajectory_runner.run(trajectory_request)
            traj_id_to_input = {
                traj_id.instance_id: {"env_class": env_class, "env_extras": env_extra}
                for traj_id, env_class, env_extra in zip(
                    trajectory_request["trajectory_ids"],
                    trajectory_request["env_classes"],
                    trajectory_request["env_extras"],
                )
            }
            for traj_id in trajectory_batch["trajectory_ids"]:
                assert traj_id.instance_id in traj_id_to_input, (
                    f"Trajectory ID {traj_id.instance_id} not found in input"
                )
                concat_all_envs.append(traj_id_to_input[traj_id.instance_id]["env_class"])
                concat_env_extras.append(traj_id_to_input[traj_id.instance_id]["env_extras"])
                concat_uids.append(traj_id.instance_id)
            # validate_trajectory_batch(trajectory_request, trajectory_batch)
            trajectory_batches.append(trajectory_batch)
        concat_trajectory_batches: TrajectoryBatch = concatenate_trajectory_batches(trajectory_batches)
    finally:
        # Always stop the eval session, even if evaluation fails
        if has_eval_session:
            await trajectory_runner.stop_eval_session()

    # Extract data_sources from env_extras
    concat_data_sources = [env_extra.get("data_source") for env_extra in concat_env_extras]
    vis = tokenizer.decode(trajectory_batch["response_ids"][0])
    logger.info(f"Eval output example: {vis}")

    # Only use the final step metrics
    trajectory_batch_last_step = defaultdict(list)
    is_last_step_mask = concat_trajectory_batches["is_last_step"]
    for key in concat_trajectory_batches:
        if isinstance(concat_trajectory_batches[key], list):
            assert len(concat_trajectory_batches[key]) == len(is_last_step_mask)
            trajectory_batch_last_step[key] = [
                val for val, is_last_step in zip(concat_trajectory_batches[key], is_last_step_mask) if is_last_step
            ]
    uids_last_step = [uid for uid, is_last_step in zip(concat_uids, is_last_step_mask) if is_last_step]
    data_sources_last_step = [
        data_source for data_source, is_last_step in zip(concat_data_sources, is_last_step_mask) if is_last_step
    ]

    # 2. Group data by data source and calculate per-dataset metrics
    eval_metrics = calculate_per_dataset_metrics(
        trajectory_batch_last_step, uids_last_step, data_sources_last_step, cfg.generator.eval_n_samples_per_prompt
    )
    # 3. Calculate overall metrics across all datasets
    overall_avg_score, overall_pass_at_n = get_metrics_from_trajectory_batch(trajectory_batch_last_step, uids_last_step)
    eval_metrics.update(
        {
            "eval/all/avg_score": overall_avg_score,
            f"eval/all/pass_at_{cfg.generator.eval_n_samples_per_prompt}": overall_pass_at_n,
        }
    )

    # 4. Prepare dumping data
    # TODO[Ben] update this to be cloud-compatible
    if cfg.trainer.dump_eval_results:
        with Timer("dump_eval_results"):
            data_save_dir = (
                Path(cfg.trainer.export_path)
                / "dumped_evals"
                / ("eval_only" if global_step is None else f"global_step_{global_step}_evals")
            )
            data_save_dir.mkdir(parents=True, exist_ok=True)
            dump_per_dataset_eval_results(
                data_save_dir,
                tokenizer,
                concat_trajectory_batches,
                concat_data_sources,
                concat_all_envs,
                concat_env_extras,
                eval_metrics,
            )

    return eval_metrics
