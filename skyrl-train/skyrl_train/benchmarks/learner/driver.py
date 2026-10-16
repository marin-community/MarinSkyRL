"""Learner-only benchmark driver: replay a stored batch ensemble through the Megatron policy worker.

The driver builds a production policy config, starts the policy worker group, and runs ``ppo_train`` on stored
RL batches with router replay, cycling through the ensemble's length profiles. Warmup updates are excluded from
the summaries. Each update records wall time, the workers' ``ppo_train`` phase breakdown, allocator peaks, tokens,
throughput and MFU; the run emits one JSON record (and optionally a W&B run).

Only ``run_benchmark`` touches Ray and the GPU worker. Config construction, scheduling and aggregation are pure.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Protocol

import hydra
import numpy as np
import ray
import wandb
from loguru import logger
from omegaconf import DictConfig, OmegaConf, open_dict
from ray.util.placement_group import placement_group
from transformers import AutoConfig

from skyrl_train.benchmarks.learner.batch_artifact import (
    EnsembleBatchManifest,
    LengthProfile,
    list_ensemble,
    read_ensemble_batch,
    read_manifest,
)
from skyrl_train.benchmarks.learner.flops import (
    PEAK_BF16_DENSE_FLOPS,
    Accelerator,
    GrugFlopsModel,
    RecomputePolicy,
    accelerator_from_device_name,
    mfu,
)
from skyrl_train.benchmarks.learner.truncate_checkpoint import (
    TRUNCATION_PROVENANCE_FILENAME,
    read_truncation_provenance,
    slice_routes,
)
from skyrl_train.dataset.routed_expert_batch import RoutedExpertRows
from skyrl_train.distributed.dispatch import concatenate_outputs_after_mesh_dispatch
from skyrl_train.entrypoints.main_base import config_dir
from skyrl_train.io import io
from skyrl_train.models.grug_moe import GrugMoeConfig
from skyrl_train.timing_observability import PPO_TRAIN_PHASE_SECONDS_KEY
from skyrl_train.training_batch import TrainingInputBatch, pad_training_input
from skyrl_train.utils.utils import get_ray_pg_ready_with_timeout, initialize_ray, validate_cfg
from skyrl_train.workers.worker import PPORayActorGroup

SCHEMA_VERSION = 1
PLACEMENT_GROUP_TIMEOUT_SECONDS = 600
PROFILER_RANKS = (0,)
PERCENTILES = (10, 50, 90)
ALL_PROFILES = "all"


class OptimizerSettings(Protocol):
    name: ClassVar[str]

    def apply(self, cfg: DictConfig) -> None: ...


@dataclass(frozen=True)
class MuonHSettings:
    """Hero's MuonH optimizer; ``offload_momentum`` keeps Muon momentum in host memory."""

    name: ClassVar[str] = "MuonH"
    lr: float
    adam_lr: float
    offload_momentum: bool
    adam_betas: tuple[float, float] = (0.9, 0.95)

    def apply(self, cfg: DictConfig) -> None:
        optimizer = cfg.trainer.policy.optimizer_config
        optimizer.optimizer = self.name
        optimizer.lr = self.lr
        optimizer.weight_decay = 0.0
        optimizer.max_grad_norm = 0.0
        optimizer.adam_betas = list(self.adam_betas)
        optimizer.optimizer_kwargs = {"adam_lr": self.adam_lr, "offload_momentum": self.offload_momentum}


@dataclass(frozen=True)
class AdamWSettings:
    """Megatron AdamW; ``offload_fraction`` > 0 places that share of optimizer state on the CPU."""

    name: ClassVar[str] = "AdamW"
    lr: float
    offload_fraction: float
    weight_decay: float = 0.0
    adam_betas: tuple[float, float] = (0.9, 0.999)

    def apply(self, cfg: DictConfig) -> None:
        if not 0.0 <= self.offload_fraction <= 1.0:
            raise ValueError(f"offload_fraction must be in [0, 1], got {self.offload_fraction}")
        optimizer = cfg.trainer.policy.optimizer_config
        optimizer.optimizer = self.name
        optimizer.lr = self.lr
        optimizer.weight_decay = self.weight_decay
        optimizer.adam_betas = list(self.adam_betas)
        offload = self.offload_fraction > 0
        kwargs = cfg.trainer.policy.megatron_config.optimizer_config_kwargs
        # Megatron's hybrid CPU optimizer requires precision-aware state without remainders.
        kwargs.use_precision_aware_optimizer = offload
        kwargs.optimizer_cpu_offload = offload
        kwargs.optimizer_offload_fraction = self.offload_fraction
        if offload:
            with open_dict(kwargs):
                kwargs.store_param_remainders = False


@dataclass(frozen=True)
class ParallelLayout:
    tensor_parallel: int
    pipeline_parallel: int
    expert_parallel: int
    context_parallel: int
    expert_tensor_parallel: int
    num_nodes: int
    gpus_per_node: int

    @property
    def world_size(self) -> int:
        return self.num_nodes * self.gpus_per_node

    @property
    def data_parallel(self) -> int:
        model_parallel = self.tensor_parallel * self.pipeline_parallel * self.context_parallel
        if self.world_size % model_parallel:
            raise ValueError(f"world size {self.world_size} is not divisible by TP*PP*CP={model_parallel}")
        return self.world_size // model_parallel


@dataclass(frozen=True)
class BenchmarkSpec:
    """Everything that determines a benchmark run; serialized verbatim into the result record.

    Attributes:
        megatron_overrides: Extra Hydra dotlist overrides (``key=value``) applied last, e.g. MoE dispatcher knobs.
        profile_update: Capture a torch profiler operator table for the first measured update on rank 0; that
            update is excluded from summaries.
        score_old_logprobs: Run the policy forward before each update and train against its log-probs, timing
            it separately, as the production trainer does.
        routes_checkpoint: Checkpoint id whose stored routes the learner replays, or ``None`` to route natively
            with router replay off.
        memory_breakdown: After the last update, record each rank's parameter, gradient, FP32 master and
            optimizer-state bytes by optimizer route (MuonH only).
        num_sequences: Benchmark only the ensemble batches with this many sequences, or every batch if ``None``.
    """

    run_name: str
    harness_revision: str
    model_path: str
    layout: ParallelLayout
    optimizer: MuonHSettings | AdamWSettings
    recompute: RecomputePolicy
    micro_batch_size_per_gpu: int
    sample_packing: bool
    flash_attn: bool
    ensemble_root: str
    ensemble_version: str
    profiles: tuple[LengthProfile, ...]
    routes_checkpoint: str | None
    warmup_updates: int
    measured_updates: int
    score_old_logprobs: bool
    profile_update: bool
    profiler_dir: str | None
    output_path: str
    wandb_project: str | None
    megatron_overrides: tuple[str, ...] = ()
    memory_breakdown: bool = False
    num_sequences: int | None = None


@dataclass(frozen=True)
class EnsembleEntry:
    path: str
    profile: LengthProfile
    batch_index: int
    num_sequences: int


@dataclass(frozen=True)
class ScheduledUpdate:
    index: int
    entry: EnsembleEntry
    warmup: bool
    profiled: bool


def schedule_updates(
    entries: Sequence[EnsembleEntry],
    profiles: Sequence[LengthProfile],
    *,
    warmup_updates: int,
    measured_updates: int,
    profile_update: bool,
) -> tuple[ScheduledUpdate, ...]:
    """Interleave profiles round-robin, cycling each profile's batches in index order.

    The first ``warmup_updates`` are warmup. With ``profile_update`` the first measured update is profiled.
    """
    if warmup_updates < 0 or measured_updates < 1:
        raise ValueError(f"need warmup >= 0 and measured >= 1 updates, got {warmup_updates} and {measured_updates}")
    queues = {
        profile: sorted((e for e in entries if e.profile == profile), key=lambda e: e.batch_index)
        for profile in profiles
    }
    missing = [profile for profile, queue in queues.items() if not queue]
    if missing:
        raise ValueError(f"ensemble has no batches for profiles {missing}")
    total = warmup_updates + measured_updates
    updates = []
    for index in range(total):
        profile = profiles[index % len(profiles)]
        queue = queues[profile]
        entry = queue[(index // len(profiles)) % len(queue)]
        updates.append(
            ScheduledUpdate(
                index=index,
                entry=entry,
                warmup=index < warmup_updates,
                profiled=profile_update and index == warmup_updates,
            )
        )
    return tuple(updates)


def rows_per_update(entries: Sequence[EnsembleEntry], layout: ParallelLayout, micro_batch_size_per_gpu: int) -> int:
    """Return the padded row count every update trains on.

    The policy worker fixes its per-rank mini-batch size at init, so all scheduled batches must pad to the same
    multiple of ``data_parallel * micro_batch_size_per_gpu``. One update is one optimizer step.
    """
    multiple = layout.data_parallel * micro_batch_size_per_gpu
    padded = {math.ceil(entry.num_sequences / multiple) * multiple for entry in entries}
    if len(padded) != 1:
        sizes = sorted({entry.num_sequences for entry in entries})
        raise ValueError(
            f"scheduled batches pad to different sizes {sorted(padded)} (rows {sizes}, multiple {multiple})"
        )
    rows = padded.pop()
    # pad_training_input pads by cloning leading rows, so it cannot more than double a batch.
    too_small = sorted({entry.num_sequences for entry in entries if 2 * entry.num_sequences < rows})
    if too_small:
        raise ValueError(f"batches with {too_small} rows cannot be padded to {rows} rows by cloning")
    return rows


def compose_base_config() -> DictConfig:
    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        return hydra.compose(config_name="ppo_base_config")


def build_policy_cfg(spec: BenchmarkSpec, rows: int, base: DictConfig) -> DictConfig:
    """Return a validated policy-only config for ``spec`` training on ``rows`` sequences per update."""
    cfg = copy.deepcopy(base)
    trainer = cfg.trainer
    trainer.strategy = "megatron"
    trainer.logger = "console"
    trainer.policy.model.path = spec.model_path
    trainer.critic.model.path = None
    trainer.bf16 = True
    trainer.flash_attn = spec.flash_attn
    trainer.use_sample_packing = spec.sample_packing
    trainer.gradient_checkpointing = spec.recompute is not RecomputePolicy.NONE
    trainer.policy_train_spans = True
    # One optimizer step per update over the whole stored batch.
    trainer.train_batch_size = rows
    trainer.policy_mini_batch_size = rows
    # Validation requires equal sizes so the scoring and training forwards select the same kernels.
    trainer.micro_train_batch_size_per_gpu = spec.micro_batch_size_per_gpu
    trainer.micro_forward_batch_size_per_gpu = spec.micro_batch_size_per_gpu
    trainer.update_epochs_per_batch = 1
    trainer.algorithm.use_kl_loss = False
    trainer.algorithm.use_kl_in_reward = False
    trainer.algorithm.use_entropy_loss = False
    trainer.placement.colocate_all = False
    trainer.placement.policy_num_nodes = spec.layout.num_nodes
    trainer.placement.policy_num_gpus_per_node = spec.layout.gpus_per_node
    cfg.generator.n_samples_per_prompt = 1

    megatron = trainer.policy.megatron_config
    megatron.tensor_model_parallel_size = spec.layout.tensor_parallel
    megatron.pipeline_model_parallel_size = spec.layout.pipeline_parallel
    megatron.context_parallel_size = spec.layout.context_parallel
    megatron.expert_model_parallel_size = spec.layout.expert_parallel
    megatron.expert_tensor_parallel_size = spec.layout.expert_tensor_parallel
    megatron.moe_router_replay = spec.routes_checkpoint is not None
    recompute = megatron.transformer_config_kwargs
    if spec.recompute is RecomputePolicy.FULL_LAYERS:
        recompute.recompute_granularity = "full"
        recompute.recompute_method = "uniform"
        recompute.recompute_num_layers = 1
    elif spec.recompute is RecomputePolicy.CORE_ATTENTION:
        recompute.recompute_granularity = "selective"
        recompute.recompute_modules = ["core_attn"]
        recompute.recompute_method = None
        recompute.recompute_num_layers = None

    profiler = megatron.torch_profiler_config
    profiler.enable = spec.profile_update
    if spec.profile_update:
        if spec.profiler_dir is None:
            raise ValueError("profile_update requires profiler_dir")
        profiler.ranks = list(PROFILER_RANKS)
        profiler.save_path = spec.profiler_dir
        profiler.update_index = spec.warmup_updates

    spec.optimizer.apply(cfg)
    if spec.megatron_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(spec.megatron_overrides)))
    validate_cfg(cfg)
    return cfg


@dataclass(frozen=True)
class Percentiles:
    p10: float
    median: float
    p90: float

    @classmethod
    def of(cls, values: Sequence[float]) -> Percentiles:
        p10, median, p90 = np.percentile(np.asarray(values, dtype=np.float64), PERCENTILES)
        return cls(p10=float(p10), median=float(median), p90=float(p90))


@dataclass(frozen=True)
class UpdateRecord:
    """One ``ppo_train`` call. Rank-dependent values are maxima over ranks (the critical path).

    Attributes:
        tokens: Attended tokens of the stored rows; padding rows added for data-parallel divisibility cost compute
            but are excluded here and from the FLOP counts.
        wall_seconds: Driver-observed ``ppo_train`` time, including batch dispatch to the workers. Throughput, MFU
            and HFU use it.
        worker_seconds: The slowest rank's ``ppo_train`` time measured inside the worker.
        phase_seconds: Each worker phase's slowest-rank seconds, keyed as the worker's PhaseBreakdown names them.
    """

    index: int
    batch_path: str
    profile: LengthProfile
    batch_index: int
    warmup: bool
    profiled: bool
    num_sequences: int
    padded_rows: int
    tokens: int
    response_tokens: int
    max_sequence_tokens: int
    wall_seconds: float
    worker_seconds: float
    phase_seconds: dict[str, float]
    scoring_seconds: float | None
    model_flops: float
    hardware_flops: float
    scoring_flops: float | None
    tokens_per_second_per_gpu: float
    mfu: float
    hfu: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    train_status: dict[str, float]


@dataclass(frozen=True)
class WorkerObservation:
    """What the policy workers report for one update: per-rank metadata and allocator peaks."""

    train_status: dict[str, float]
    phase_seconds_by_rank: tuple[dict[str, float], ...]
    peak_allocated_by_rank: tuple[int, ...]
    peak_reserved_by_rank: tuple[int, ...]


def update_record(
    update: ScheduledUpdate,
    batch: TrainingInputBatch,
    *,
    padded_rows: int,
    wall_seconds: float,
    scoring_seconds: float | None,
    observation: WorkerObservation,
    flops_model: GrugFlopsModel,
    recompute: RecomputePolicy,
    num_gpus: int,
    accelerator: Accelerator,
) -> UpdateRecord:
    """Combine one update's unpadded batch, timings and worker reports into a record."""
    seq_lens = [int(length) for length in batch["attention_mask"].sum(dim=-1).tolist()]
    flops = flops_model.batch_flops(seq_lens, recompute)
    phases = sorted({phase for rank in observation.phase_seconds_by_rank for phase in rank})
    phase_seconds = {phase: max(rank.get(phase, 0.0) for rank in observation.phase_seconds_by_rank) for phase in phases}
    scoring_flops = None if scoring_seconds is None else flops.forward
    return UpdateRecord(
        index=update.index,
        batch_path=update.entry.path,
        profile=update.entry.profile,
        batch_index=update.entry.batch_index,
        warmup=update.warmup,
        profiled=update.profiled,
        num_sequences=len(seq_lens),
        padded_rows=padded_rows,
        tokens=flops.tokens,
        response_tokens=int(batch["response_mask"].sum().item()),
        max_sequence_tokens=max(seq_lens),
        wall_seconds=wall_seconds,
        worker_seconds=phase_seconds["ppo_train"],
        phase_seconds=phase_seconds,
        scoring_seconds=scoring_seconds,
        model_flops=flops.training_model,
        hardware_flops=flops.training_hardware,
        scoring_flops=scoring_flops,
        tokens_per_second_per_gpu=flops.tokens / wall_seconds / num_gpus,
        mfu=mfu(flops.training_model, wall_seconds, num_gpus, accelerator),
        hfu=mfu(flops.training_hardware, wall_seconds, num_gpus, accelerator),
        peak_allocated_bytes=max(observation.peak_allocated_by_rank),
        peak_reserved_bytes=max(observation.peak_reserved_by_rank),
        train_status=dict(observation.train_status),
    )


@dataclass(frozen=True)
class ProfileSummary:
    """Measured, unprofiled updates of one length profile (or ``all``)."""

    profile: str
    updates: int
    tokens: int
    wall_seconds: Percentiles
    tokens_per_second_per_gpu: Percentiles
    mfu: Percentiles
    hfu: Percentiles
    phase_seconds: dict[str, Percentiles]
    peak_reserved_bytes: int


def _summary(profile: str, records: Sequence[UpdateRecord]) -> ProfileSummary:
    phases = sorted({phase for record in records for phase in record.phase_seconds})
    return ProfileSummary(
        profile=profile,
        updates=len(records),
        tokens=sum(record.tokens for record in records),
        wall_seconds=Percentiles.of([record.wall_seconds for record in records]),
        tokens_per_second_per_gpu=Percentiles.of([record.tokens_per_second_per_gpu for record in records]),
        mfu=Percentiles.of([record.mfu for record in records]),
        hfu=Percentiles.of([record.hfu for record in records]),
        phase_seconds={
            phase: Percentiles.of([record.phase_seconds.get(phase, 0.0) for record in records]) for phase in phases
        },
        peak_reserved_bytes=max(record.peak_reserved_bytes for record in records),
    )


def summarize(records: Sequence[UpdateRecord]) -> tuple[ProfileSummary, ...]:
    """Per-profile and overall summaries over measured updates, excluding warmup and the profiled update."""
    measured = [record for record in records if not record.warmup and not record.profiled]
    if not measured:
        raise ValueError("no measured, unprofiled updates to summarize")
    profiles = list(dict.fromkeys(record.profile for record in measured))
    summaries = [_summary(str(profile), [r for r in measured if r.profile == profile]) for profile in profiles]
    return (*summaries, _summary(ALL_PROFILES, measured))


@dataclass(frozen=True)
class ModelDescription:
    path: str
    model_type: str
    num_layers: int
    hidden_size: int
    num_experts: int
    experts_per_token: int
    latent_dim: int | None
    # Model training FLOPs per token for a single sequence of 2048 tokens: a fixed reference point.
    training_flops_per_token_at_2048: float


@dataclass(frozen=True)
class Hardware:
    device_name: str
    accelerator: Accelerator
    peak_flops_per_gpu: float
    num_gpus: int


@dataclass(frozen=True)
class BenchmarkResult:
    schema_version: int
    created_at: str
    spec: BenchmarkSpec
    optimizer_name: str
    model: ModelDescription
    hardware: Hardware
    rows_per_update: int
    updates: tuple[UpdateRecord, ...]
    summaries: tuple[ProfileSummary, ...]
    # Per-rank ``learner_memory_breakdown`` replies, when the spec asks for them.
    memory_breakdown: tuple[dict[str, Any], ...] = ()


REFERENCE_SEQUENCE_TOKENS = 2048


def describe_model(model_path: str, config: GrugMoeConfig, flops_model: GrugFlopsModel) -> ModelDescription:
    reference = flops_model.sequence_flops(REFERENCE_SEQUENCE_TOKENS, RecomputePolicy.NONE)
    return ModelDescription(
        path=model_path,
        model_type=config.model_type,
        num_layers=config.num_hidden_layers,
        hidden_size=config.hidden_size,
        num_experts=config.num_local_experts,
        experts_per_token=config.num_experts_per_tok,
        latent_dim=config.latent_dim,
        training_flops_per_token_at_2048=reference.training_model / REFERENCE_SEQUENCE_TOKENS,
    )


def result_json(result: BenchmarkResult) -> str:
    return json.dumps(dataclasses.asdict(result), indent=2, sort_keys=True)


def write_result(result: BenchmarkResult, path: str) -> None:
    io.write_bytes_atomic(path, result_json(result).encode())


def wandb_update_metrics(record: UpdateRecord) -> dict[str, float]:
    """Flat W&B metrics for one update, keyed under its profile."""
    prefix = f"learner_bench/{record.profile}"
    metrics = {
        f"{prefix}/wall_seconds": record.wall_seconds,
        f"{prefix}/tokens_per_second_per_gpu": record.tokens_per_second_per_gpu,
        f"{prefix}/mfu": record.mfu,
        f"{prefix}/hfu": record.hfu,
        f"{prefix}/tokens": float(record.tokens),
        f"{prefix}/peak_reserved_gib": record.peak_reserved_bytes / 2**30,
        "learner_bench/warmup": float(record.warmup),
    }
    metrics.update({f"{prefix}/phase/{phase}": seconds for phase, seconds in record.phase_seconds.items()})
    if record.scoring_seconds is not None:
        metrics[f"{prefix}/scoring_seconds"] = record.scoring_seconds
    return metrics


def wandb_summary(summaries: Sequence[ProfileSummary]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for profile in summaries:
        for field in ("wall_seconds", "tokens_per_second_per_gpu", "mfu", "hfu"):
            for stat, value in dataclasses.asdict(getattr(profile, field)).items():
                summary[f"learner_bench_summary/{profile.profile}/{field}/{stat}"] = value
    return summary


def log_to_wandb(result: BenchmarkResult, project: str) -> None:
    run = wandb.init(
        project=project,
        name=result.spec.run_name,
        config={
            "spec": dataclasses.asdict(result.spec),
            "optimizer_name": result.optimizer_name,
            "model": dataclasses.asdict(result.model),
            "hardware": dataclasses.asdict(result.hardware),
            "rows_per_update": result.rows_per_update,
            "schema_version": result.schema_version,
        },
    )
    try:
        for record in result.updates:
            run.log(wandb_update_metrics(record), step=record.index)
        run.summary.update(wandb_summary(result.summaries))
    finally:
        run.finish()


def worker_observation(outputs: Sequence[Any], peaks: Sequence[Mapping[str, Any]]) -> WorkerObservation:
    """Collect ``ppo_train`` per-rank metadata and ``get_cuda_memory_peaks`` replies."""
    phase_seconds = []
    for output in outputs:
        if PPO_TRAIN_PHASE_SECONDS_KEY not in output.metadata:
            raise ValueError("ppo_train returned no phase seconds; trainer.policy_train_spans must be true")
        phase_seconds.append(dict(output.metadata[PPO_TRAIN_PHASE_SECONDS_KEY]))
    return WorkerObservation(
        train_status={key: float(value) for key, value in outputs[0].metadata["train_status"].items()},
        phase_seconds_by_rank=tuple(phase_seconds),
        peak_allocated_by_rank=tuple(int(peak["peak_allocated"]) for peak in peaks),
        peak_reserved_by_rank=tuple(int(peak["peak_reserved"]) for peak in peaks),
    )


def select_entries(
    entries: Sequence[EnsembleEntry], profiles: Sequence[LengthProfile], num_sequences: int | None
) -> list[EnsembleEntry]:
    """Keep the entries of ``profiles`` and, if ``num_sequences`` is set, of that sequence count."""
    return [
        entry
        for entry in entries
        if entry.profile in profiles and (num_sequences is None or entry.num_sequences == num_sequences)
    ]


def ensemble_entries(spec: BenchmarkSpec) -> list[EnsembleEntry]:
    entries = []
    for path in list_ensemble(spec.ensemble_root, spec.ensemble_version):
        manifest = read_manifest(path)
        entries.append(EnsembleEntry(path, manifest.profile, manifest.batch_index, manifest.num_sequences))
    return select_entries(entries, spec.profiles, spec.num_sequences)


def source_num_layers(model_path: str, model_num_layers: int) -> int:
    """Return the layer count of the checkpoint ``model_path`` was truncated from, or its own if not truncated."""
    if not io.exists(f"{model_path.rstrip('/')}/{TRUNCATION_PROVENANCE_FILENAME}"):
        return model_num_layers
    provenance = read_truncation_provenance(model_path)
    if provenance.num_layers != model_num_layers:
        raise ValueError(
            f"{model_path} truncation provenance records {provenance.num_layers} layers, config has {model_num_layers}"
        )
    return provenance.source_num_layers


def replay_routes(routes: RoutedExpertRows, model_num_layers: int, full_num_layers: int) -> RoutedExpertRows:
    """Fit routes to the benchmarked model's decoder layers.

    Routes recorded on the full checkpoint (``full_num_layers`` layers) are sliced to the leading layers of a
    depth-truncated model; routes recorded on the benchmarked model itself are used as they are.
    """
    route_layers = {row.shape[1] for row in routes.rows}
    if len(route_layers) != 1:
        raise ValueError(f"route rows disagree on the layer count: {sorted(route_layers)}")
    (layers,) = route_layers
    if layers == model_num_layers:
        return routes
    if layers == full_num_layers:
        return slice_routes(routes, model_num_layers)
    raise ValueError(
        f"routes carry {layers} layers; expected the model's {model_num_layers} or its source checkpoint's "
        f"{full_num_layers}"
    )


def load_replay_batch(
    path: str, routes_checkpoint: str | None, model_num_layers: int, full_num_layers: int
) -> tuple[TrainingInputBatch, EnsembleBatchManifest]:
    """Read a stored batch with its routes (if ``routes_checkpoint``) fitted to the benchmarked model's layers."""
    batch, manifest = read_ensemble_batch(path, routes_checkpoint=routes_checkpoint)
    if batch.routed_expert_rows is not None:
        batch.routed_expert_rows = replay_routes(batch.routed_expert_rows, model_num_layers, full_num_layers)
    return batch, manifest


def _init_policy_group(cfg: DictConfig, layout: ParallelLayout) -> PPORayActorGroup:
    from skyrl_train.workers.megatron.megatron_worker import PolicyWorker  # optional megatron extra

    bundles = [{"GPU": layout.gpus_per_node, "CPU": layout.gpus_per_node} for _ in range(layout.num_nodes)]
    pg = placement_group(bundles, strategy="PACK")
    get_ray_pg_ready_with_timeout(pg, timeout=PLACEMENT_GROUP_TIMEOUT_SECONDS)
    group = PPORayActorGroup(
        cfg,
        num_nodes=layout.num_nodes,
        num_gpus_per_node=layout.gpus_per_node,
        ray_actor_type=PolicyWorker,
        pg=pg,
        num_gpus_per_actor=1,
        colocate_all=False,
        sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
    )
    ray.get(group.async_init_model(cfg.trainer.policy.model.path))
    return group


def _score_old_logprobs(policy: PPORayActorGroup, batch: TrainingInputBatch) -> float:
    """Replace the batch's old log-probs with the policy's own forward and return its wall seconds."""
    started = time.perf_counter()
    outputs = ray.get(policy.async_run_ray_method("mesh", "forward", data=batch))
    seconds = time.perf_counter() - started
    scores = concatenate_outputs_after_mesh_dispatch(policy.actor_infos, outputs)["output"]
    batch["action_log_probs"] = (scores * batch["response_mask"]).float()
    return seconds


def run_benchmark(spec: BenchmarkSpec) -> BenchmarkResult:
    """Run the benchmark on the current Ray cluster and return its result record."""
    entries = ensemble_entries(spec)
    schedule = schedule_updates(
        entries,
        spec.profiles,
        warmup_updates=spec.warmup_updates,
        measured_updates=spec.measured_updates,
        profile_update=spec.profile_update,
    )
    scheduled_entries = list(dict.fromkeys(update.entry for update in schedule))
    rows = rows_per_update(scheduled_entries, spec.layout, spec.micro_batch_size_per_gpu)
    cfg = build_policy_cfg(spec, rows, compose_base_config())
    model_config = AutoConfig.from_pretrained(spec.model_path)
    if not isinstance(model_config, GrugMoeConfig):
        raise ValueError(f"learner benchmark FLOPs model supports Grug MoE only, got {type(model_config).__name__}")
    flops_model = GrugFlopsModel.from_config(model_config)
    full_num_layers = source_num_layers(spec.model_path, model_config.num_hidden_layers)
    multiple = spec.layout.data_parallel * spec.micro_batch_size_per_gpu

    initialize_ray(cfg)
    try:
        policy = _init_policy_group(cfg, spec.layout)
        records: list[UpdateRecord] = []
        hardware: Hardware | None = None
        for update in schedule:
            batch, _ = load_replay_batch(
                update.entry.path, spec.routes_checkpoint, model_config.num_hidden_layers, full_num_layers
            )
            batch.metadata["global_step"] = update.index
            padded = pad_training_input(batch, multiple)
            scoring_seconds = _score_old_logprobs(policy, padded) if spec.score_old_logprobs else None
            started = time.perf_counter()
            outputs = ray.get(policy.async_run_ray_method("mesh", "ppo_train", padded))
            wall_seconds = time.perf_counter() - started
            peaks = ray.get(policy.async_run_ray_method("pass_through", "get_cuda_memory_peaks"))
            if hardware is None:
                device_name = peaks[0]["device_name"]
                accelerator = accelerator_from_device_name(device_name)
                hardware = Hardware(
                    device_name=device_name,
                    accelerator=accelerator,
                    peak_flops_per_gpu=PEAK_BF16_DENSE_FLOPS[accelerator],
                    num_gpus=spec.layout.world_size,
                )
            record = update_record(
                update,
                batch,
                padded_rows=padded.batch_size,
                wall_seconds=wall_seconds,
                scoring_seconds=scoring_seconds,
                observation=worker_observation(outputs, peaks),
                flops_model=flops_model,
                recompute=spec.recompute,
                num_gpus=spec.layout.world_size,
                accelerator=hardware.accelerator,
            )
            logger.info(
                f"learner bench update {record.index} {record.profile} warmup={record.warmup}: "
                f"{record.wall_seconds:.2f}s {record.tokens_per_second_per_gpu:.0f} tok/s/GPU MFU {record.mfu:.3f}"
            )
            records.append(record)
        memory_breakdown = (
            tuple(ray.get(policy.async_run_ray_method("pass_through", "learner_memory_breakdown")))
            if spec.memory_breakdown
            else ()
        )
    finally:
        ray.shutdown()
    assert hardware is not None
    return BenchmarkResult(
        schema_version=SCHEMA_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),
        spec=spec,
        optimizer_name=spec.optimizer.name,
        model=describe_model(spec.model_path, model_config, flops_model),
        hardware=hardware,
        rows_per_update=rows,
        updates=tuple(records),
        summaries=summarize(records),
        memory_breakdown=memory_breakdown,
    )


def _git_revision() -> str:
    checkout = Path(__file__).resolve().parent
    completed = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise ValueError("cannot resolve the harness revision from git; pass --harness-revision")
    return completed.stdout.strip()


def parse_spec(argv: Sequence[str] | None = None) -> BenchmarkSpec:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--harness-revision", help="defaults to this checkout's git HEAD")
    parser.add_argument("--model-path", required=True, help="full or depth-truncated Hero HF checkpoint")
    for name in ("tp", "pp", "ep", "cp", "etp", "num-nodes", "gpus-per-node"):
        parser.add_argument(f"--{name}", type=int, required=True)
    parser.add_argument("--optimizer", choices=[MuonHSettings.name, AdamWSettings.name], required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--adam-lr", type=float, help="MuonH Adam-routed learning rate")
    parser.add_argument("--offload-momentum", action=argparse.BooleanOptionalAction, help="MuonH momentum on host")
    parser.add_argument("--offload-fraction", type=float, help="AdamW optimizer-state CPU offload fraction")
    parser.add_argument("--recompute", choices=[policy.value for policy in RecomputePolicy], required=True)
    parser.add_argument("--micro-batch-size-per-gpu", type=int, required=True)
    parser.add_argument("--sample-packing", action=argparse.BooleanOptionalAction, required=True)
    parser.add_argument("--flash-attn", action=argparse.BooleanOptionalAction, required=True)
    parser.add_argument("--ensemble-root", required=True)
    parser.add_argument("--ensemble-version", required=True)
    parser.add_argument("--profiles", nargs="+", choices=[profile.value for profile in LengthProfile], required=True)
    parser.add_argument("--routes-checkpoint", help="replay these stored routes; omit to route natively")
    parser.add_argument("--num-sequences", type=int, help="benchmark only the batches with this many sequences")
    parser.add_argument("--warmup-updates", type=int, required=True)
    parser.add_argument("--measured-updates", type=int, required=True)
    parser.add_argument("--score-old-logprobs", action="store_true")
    parser.add_argument("--profiler-dir", help="capture a torch profiler table for the first measured update")
    parser.add_argument("--output", required=True, help="local or s3:// path of the JSON result")
    parser.add_argument("--wandb-project")
    parser.add_argument("--override", action="append", default=[], help="extra Hydra dotlist override")
    parser.add_argument(
        "--memory-breakdown", action="store_true", help="record per-rank memory by optimizer route (MuonH)"
    )
    args = parser.parse_args(argv)

    optimizer: MuonHSettings | AdamWSettings
    if args.optimizer == MuonHSettings.name:
        if args.adam_lr is None or args.offload_momentum is None or args.offload_fraction is not None:
            parser.error("MuonH requires --adam-lr and --[no-]offload-momentum and rejects --offload-fraction")
        optimizer = MuonHSettings(lr=args.lr, adam_lr=args.adam_lr, offload_momentum=args.offload_momentum)
    else:
        if args.offload_fraction is None or args.adam_lr is not None or args.offload_momentum is not None:
            parser.error("AdamW requires --offload-fraction and rejects MuonH options")
        optimizer = AdamWSettings(lr=args.lr, offload_fraction=args.offload_fraction)
    return BenchmarkSpec(
        run_name=args.run_name,
        harness_revision=args.harness_revision or _git_revision(),
        model_path=args.model_path,
        layout=ParallelLayout(
            tensor_parallel=args.tp,
            pipeline_parallel=args.pp,
            expert_parallel=args.ep,
            context_parallel=args.cp,
            expert_tensor_parallel=args.etp,
            num_nodes=args.num_nodes,
            gpus_per_node=args.gpus_per_node,
        ),
        optimizer=optimizer,
        recompute=RecomputePolicy(args.recompute),
        micro_batch_size_per_gpu=args.micro_batch_size_per_gpu,
        sample_packing=args.sample_packing,
        flash_attn=args.flash_attn,
        ensemble_root=args.ensemble_root,
        ensemble_version=args.ensemble_version,
        profiles=tuple(LengthProfile(profile) for profile in args.profiles),
        routes_checkpoint=args.routes_checkpoint,
        warmup_updates=args.warmup_updates,
        measured_updates=args.measured_updates,
        score_old_logprobs=args.score_old_logprobs,
        profile_update=args.profiler_dir is not None,
        profiler_dir=args.profiler_dir,
        output_path=args.output,
        wandb_project=args.wandb_project,
        megatron_overrides=tuple(args.override),
        memory_breakdown=args.memory_breakdown,
        num_sequences=args.num_sequences,
    )


def main(argv: Sequence[str] | None = None) -> None:
    spec = parse_spec(argv)
    result = run_benchmark(spec)
    write_result(result, spec.output_path)
    logger.info(f"learner bench result written to {spec.output_path}")
    if spec.wandb_project is not None:
        log_to_wandb(result, spec.wandb_project)


if __name__ == "__main__":
    main()
