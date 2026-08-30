import torch
from skyrl_train.utils.progress import tqdm
from typing import Any, Dict, List, Protocol
from pathlib import Path
from loguru import logger
from collections import defaultdict
from dataclasses import dataclass

from skyrl_train.utils import Timer

from skyrl_train.trajectory_runners.trajectory_processing import (
    concatenate_trajectory_batches,
    get_metrics_from_trajectory_batch,
    prepare_trajectory_request,
)
from skyrl_train.trajectory_runners.base import (
    ConversationType,
    TrajectoryBatch,
    TrajectoryRequestBatch,
    TrajectoryRunner,
)
from skyrl_train.utils.trainer_utils import (
    calculate_per_dataset_metrics,
    dump_per_dataset_eval_results,
)
from skyrl_train.trajectory_runners.trajectory_processing import validate_trajectory_batch
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.utils.logging_utils import log_example
from skyrl_train.trajectory_runners.trajectory_retention import (
    TrajectorySink,
    make_trajectory_sink,
)

from omegaconf import DictConfig
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer


@dataclass
class _EvaluationRollouts:
    batch: TrajectoryBatch
    env_classes: List[str]
    env_extras: List[Dict[str, Any]]
    uids: List[str]
    example_prompt: ConversationType
    example_batch: TrajectoryBatch


class _EvaluationAccumulator(Protocol):
    env_classes: List[str]
    env_extras: List[Dict[str, Any]]
    uids: List[str]

    def record(self, request: TrajectoryRequestBatch, batch: TrajectoryBatch, uids: List[str]) -> None: ...


@dataclass
class _WholeTrajectoryAccumulator:
    env_classes: List[str]
    env_extras: List[Dict[str, Any]]
    uids: List[str]

    def record(self, request: TrajectoryRequestBatch, batch: TrajectoryBatch, uids: List[str]) -> None:
        validate_trajectory_batch(len(request["prompts"]), batch)
        self.env_classes.extend(request["env_classes"])
        self.env_extras.extend(request["env_extras"])
        self.uids.extend(uids)


@dataclass
class _StepWiseAccumulator:
    env_classes: List[str]
    env_extras: List[Dict[str, Any]]
    uids: List[str]

    def record(self, request: TrajectoryRequestBatch, batch: TrajectoryBatch, uids: List[str]) -> None:
        del uids
        trajectory_ids = request.get("trajectory_ids")
        output_ids = batch.get("trajectory_ids")
        if trajectory_ids is None or output_ids is None:
            raise ValueError("step-wise evaluation requires trajectory IDs")
        inputs_by_id = {
            trajectory_id.instance_id: (env_class, env_extra)
            for trajectory_id, env_class, env_extra in zip(
                trajectory_ids, request["env_classes"], request["env_extras"]
            )
        }
        for trajectory_id in output_ids:
            if trajectory_id.instance_id not in inputs_by_id:
                raise ValueError(f"Trajectory ID {trajectory_id.instance_id} not found in input")
            env_class, env_extra = inputs_by_id[trajectory_id.instance_id]
            self.env_classes.append(env_class)
            self.env_extras.append(env_extra)
            self.uids.append(trajectory_id.instance_id)


async def _collect_evaluation_rollouts(
    eval_dataloader: StatefulDataLoader,
    trajectory_runner: TrajectoryRunner,
    cfg: DictConfig,
    global_step: int | None,
    sink: TrajectorySink,
    val_set_name: str | None,
    accumulator: _EvaluationAccumulator,
) -> _EvaluationRollouts:
    """Own the common evaluation session and trajectory collection lifecycle."""
    trajectory_runner.set_trajectory_sink(sink)
    await trajectory_runner.start_eval_session(
        run_name=getattr(cfg.trainer, "run_name", None) or "eval",
        eval_step=global_step if global_step is not None else 0,
        val_set_name=val_set_name,
    )

    trajectory_batches: List[TrajectoryBatch] = []
    last_request = None
    last_batch = None
    try:
        pbar = tqdm(total=len(eval_dataloader), initial=0, desc="Evaluation Progress")
        for prompts in eval_dataloader:
            pbar.update(1)
            request, uids = prepare_trajectory_request(
                prompts,
                cfg.generator.eval_n_samples_per_prompt,
                get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.eval_sampling_params),
                cfg.environment.env_class,
                "eval",
                global_step,
            )
            batch = await trajectory_runner.run(request)
            trajectory_batches.append(batch)
            last_request, last_batch = request, batch

            accumulator.record(request, batch, uids)
    finally:
        await trajectory_runner.stop_eval_session()

    if last_request is None or last_batch is None:
        raise ValueError("evaluation dataloader produced no batches")
    return _EvaluationRollouts(
        batch=concatenate_trajectory_batches(
            trajectory_batches,
            tis_lcs_alert_threshold=float(cfg.trainer.algorithm.tis_lcs_alert_threshold),
        ),
        env_classes=accumulator.env_classes,
        env_extras=accumulator.env_extras,
        uids=accumulator.uids,
        example_prompt=last_request["prompts"][0],
        example_batch=last_batch,
    )


def _calculate_eval_metrics(
    batch: TrajectoryBatch,
    uids: List[str],
    data_sources: List[str | None],
    samples_per_prompt: int,
) -> Dict[str, float]:
    metrics = calculate_per_dataset_metrics(batch, uids, data_sources, samples_per_prompt)
    overall_avg_score, overall_pass_at_n = get_metrics_from_trajectory_batch(batch, uids)
    metrics.update(
        {
            "eval/all/avg_score": overall_avg_score,
            f"eval/all/pass_at_{samples_per_prompt}": overall_pass_at_n,
        }
    )
    return metrics


def _dump_eval_results(
    cfg: DictConfig,
    global_step: int | None,
    tokenizer: AutoTokenizer,
    rollouts: _EvaluationRollouts,
    data_sources: List[str | None],
    metrics: Dict[str, float],
) -> None:
    if not cfg.trainer.dump_eval_results:
        return
    with Timer("dump_eval_results"):
        # TODO(Ben): route eval dumps through skyrl_train.io when evaluation exports support cloud paths.
        data_save_dir = (
            Path(cfg.trainer.export_path)
            / "dumped_evals"
            / ("eval_only" if global_step is None else f"global_step_{global_step}_evals")
        )
        data_save_dir.mkdir(parents=True, exist_ok=True)
        dump_per_dataset_eval_results(
            data_save_dir,
            tokenizer,
            rollouts.batch,
            data_sources,
            rollouts.env_classes,
            rollouts.env_extras,
            metrics,
        )


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
    owns_sink = trajectory_sink is None
    active_sink = trajectory_sink or make_trajectory_sink(cfg.generator, tokenizer)
    try:
        accumulator = _WholeTrajectoryAccumulator([], [], [])
        rollouts = await _collect_evaluation_rollouts(
            eval_dataloader, trajectory_runner, cfg, global_step, active_sink, val_set_name, accumulator
        )
        concatenated_batch = rollouts.batch
        concat_data_sources = [env_extra.get("data_source") for env_extra in rollouts.env_extras]
        vis = tokenizer.decode(rollouts.example_batch["response_ids"][0])
        log_example(
            logger,
            prompt=rollouts.example_prompt,
            response=vis,
            reward=rollouts.example_batch["rewards"][0],
        )

        eval_metrics = _calculate_eval_metrics(
            concatenated_batch,
            rollouts.uids,
            concat_data_sources,
            cfg.generator.eval_n_samples_per_prompt,
        )
        _dump_eval_results(cfg, global_step, tokenizer, rollouts, concat_data_sources, eval_metrics)

        return eval_metrics
    finally:
        if owns_sink:
            active_sink.close()


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
    accumulator = _StepWiseAccumulator([], [], [])
    rollouts = await _collect_evaluation_rollouts(
        eval_dataloader, trajectory_runner, cfg, global_step, trajectory_sink, val_set_name, accumulator
    )
    concatenated_batch = rollouts.batch
    concat_data_sources = [env_extra.get("data_source") for env_extra in rollouts.env_extras]
    vis = tokenizer.decode(rollouts.example_batch["response_ids"][0])
    logger.info(f"Eval output example: {vis}")

    # Only use the final step metrics
    trajectory_batch_last_step = defaultdict(list)
    is_last_step_mask = concatenated_batch["is_last_step"]
    for key in concatenated_batch:
        if isinstance(concatenated_batch[key], list):
            assert len(concatenated_batch[key]) == len(is_last_step_mask)
            trajectory_batch_last_step[key] = [
                val for val, is_last_step in zip(concatenated_batch[key], is_last_step_mask) if is_last_step
            ]
    uids_last_step = [uid for uid, is_last_step in zip(rollouts.uids, is_last_step_mask) if is_last_step]
    data_sources_last_step = [
        data_source for data_source, is_last_step in zip(concat_data_sources, is_last_step_mask) if is_last_step
    ]

    eval_metrics = _calculate_eval_metrics(
        trajectory_batch_last_step,
        uids_last_step,
        data_sources_last_step,
        cfg.generator.eval_n_samples_per_prompt,
    )
    _dump_eval_results(cfg, global_step, tokenizer, rollouts, concat_data_sources, eval_metrics)

    return eval_metrics
