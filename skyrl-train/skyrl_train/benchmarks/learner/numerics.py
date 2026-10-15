"""Post-update numerics check for learner optimizations.

A learner optimization (kernel, layout, overlap, precision plumbing) must not change what one PPO update
computes beyond the bf16 reduction-order noise that any layout change introduces. This module records a
fixed set of signals for a fixed batch, checkpoint, and seed, and compares a candidate's signals against a
stored, calibrated reference:

* ``train_status/<key>``: scalars that ``ppo_train`` returns (loss, gradient norm, clip fractions, log-ratio
  statistics, update counts), one set per update.
* ``logprobs/{pre,post}_update/...``: scoring log-probs on the batch before the first update and after the
  last one, summarized as the global masked mean, the per-sequence masked means, and a deterministic sample
  of token log-probs.
* ``params/<group>/...``: a parameter-delta fingerprint over all updates, per layout-independent parameter
  group (:mod:`skyrl_train.distributed.megatron.parameter_fingerprint`). It is computed on rank-local shards
  and reduced by summation, so no full weight moves.

Calibration procedure:

1. Record the reference at least twice with :func:`record_numerics`, preferably under different layouts
   (TP/EP/PP/CP or micro-batch size), because a layout change reorders bf16 reductions and that noise is
   what a candidate legitimately adds. Runs on one layout can be bitwise identical and underestimate noise.
2. :func:`calibrate_reference` sets each signal's tolerance to ``multiplier`` times the largest pairwise
   deviation observed, but never below the floor for the signal's kind, and, for a parameter group's
   ``delta_norm`` and ``sketch``, never below the group's weight-resolution floor (below). Store it with
   :func:`write_reference`. Calibrate on at least three layouts, one of them a micro-batch change: micro-batching
   reorders gradient accumulation, and two layouts can agree exactly on groups that a third moves.
3. Run the must-fail controls (:class:`ControlPerturbation`) through the same driver with the reference
   provenance and check that :func:`compare_to_reference` rejects them. ``NEGATE_ADVANTAGES`` must fail the
   loss, sketch, and post-update log-prob signals. ``DOUBLE_ADVANTAGES`` must fail ``policy_loss`` and
   ``raw_grad_norm``; scale-invariant optimizers (Adam, MuonH) can leave the parameter delta nearly unchanged,
   which is why the train-status scalars are part of the check. A control that passes means the
   tolerances are too loose for that signal.
4. Compare each candidate with :func:`compare_to_reference`.

Weight-resolution floor: the fingerprint reads the stored (bf16) parameters, so a group's delta is a sum of
rounding steps. When the update is small next to the weights, most elements move by zero or one representable
step, and which ones flip is decided by reduction-order noise in the FP32 masters. A layout change re-draws a
subset of those flips, so the delta norm and sketches carry the shot noise of the flip count: with ``F`` flips
of about one step each, ``||delta|| ~ sqrt(F) * step`` and the relative noise is ``1 / sqrt(F) = step / ||delta||``.
The floor is that shot noise times the calibration multiplier: ``multiplier * weight_resolution * rms(w) /
||delta||``, with ``weight_resolution`` the relative spacing of the stored dtype (``2**-8`` for bf16, the lower
bound of its spacing) and ``rms(w)`` the group's pre-update RMS. Groups whose update spans many steps keep a
tight, spread-based tolerance; a group whose update is a handful of flips gets a wide one, which says that it
cannot be checked at that learning rate. Use a learning rate large enough that the groups that matter flip many
elements; that also makes the post-update log-probs move measurably.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
import hashlib
import itertools
import json
import math
import sys

from loguru import logger
from omegaconf import DictConfig, OmegaConf
import ray
import torch
from transformers import AutoConfig

from marinskyrl.model_manifest import MODEL_MANIFEST_FILENAME, ModelManifest

from skyrl_train.distributed.dispatch import concatenate_outputs_after_mesh_dispatch
from skyrl_train.distributed.megatron.parameter_fingerprint import (
    ParameterGroupStatistics,
    merge_parameter_statistics,
    mix32,
)
from skyrl_train.benchmarks.learner import driver
from skyrl_train.io import io
from skyrl_train.training_batch import TrainingInputBatch, pad_training_input
from skyrl_train.utils.utils import initialize_ray

SCHEMA_VERSION = 1

# Config subtrees that change what an update computes and so must match between reference and candidate.
# Layout and kernel choices are deliberately absent: they are what candidates vary.
NUMERICS_CONFIG_KEYS = (
    "trainer.seed",
    "trainer.algorithm",
    "trainer.policy.optimizer_config",
    "trainer.policy_mini_batch_size",
    "trainer.update_epochs_per_batch",
    "generator.sampling_params.temperature",
)

# ppo_train scalars compared per update. Losses and fractions can be zero, so they use absolute deviations.
TRAIN_STATUS_SIGNAL_KINDS = {
    "policy_loss": "absolute",
    "final_loss": "absolute",
    "policy_kl": "absolute",
    "policy_entropy": "relative",
    "raw_grad_norm": "relative",
    "ppo_clip_ratio": "absolute",
    "ppo_clip_ratio_low": "absolute",
    "ppo_clip_ratio_high": "absolute",
    "log_ratio_mean": "absolute",
    "log_ratio_abs_mean": "absolute",
    "log_ratio_abs_max": "absolute",
    "skipped_steps": "absolute",
    "policy_update_steps": "absolute",
}
REQUIRED_TRAIN_STATUS_KEYS = ("policy_loss", "raw_grad_norm", "policy_update_steps")

LOGPROB_SAMPLE_SIZE = 4096
# Lower bound of bf16's relative spacing between adjacent representable values (8 significand bits).
BF16_RELATIVE_SPACING = 2.0**-8
# Odd 32-bit salt selecting the log-prob sample's hash stream, independent of the fingerprint's sketch salts.
LOGPROB_SAMPLE_SALT = 0x165667B1


class SignalKind(StrEnum):
    """How a signal's deviation from the reference is measured.

    RELATIVE: ``|c - r| / |r|`` for one scalar. ABSOLUTE: ``max_i |c_i - r_i|``. RELATIVE_L2:
    ``||c - r|| / ||r||`` for a vector.
    """

    RELATIVE = "relative"
    ABSOLUTE = "absolute"
    RELATIVE_L2 = "relative_l2"


class ControlPerturbation(StrEnum):
    """Batch perturbations that a calibrated reference must reject."""

    NEGATE_ADVANTAGES = "negate_advantages"
    DOUBLE_ADVANTAGES = "double_advantages"


@dataclass(frozen=True)
class Signal:
    kind: SignalKind
    values: tuple[float, ...]


@dataclass(frozen=True)
class NumericsProvenance:
    """Facts that must match between a reference and a candidate, plus informational layout facts.

    Attributes:
        checkpoint: Model checkpoint identity, including the layer count of a truncated variant.
        batch_sha256: Hash of the batch tensors and the routes they carry.
        config_fingerprint: :func:`numerics_config_fingerprint` of the run's config.
        num_updates: ``ppo_train`` calls between the pre- and post-update scoring passes.
        layout: Informational parallel layout description, e.g. ``"tp1-ep8-pp1-cp1-mbs1"``.
        source_revision: Informational harness revision.
    """

    checkpoint: str
    batch_sha256: str
    config_fingerprint: str
    num_updates: int
    layout: str
    source_revision: str

    def matching_fields(self) -> tuple[str, str, str, int]:
        return (self.checkpoint, self.batch_sha256, self.config_fingerprint, self.num_updates)


@dataclass(frozen=True)
class NumericsRecord:
    provenance: NumericsProvenance
    signals: dict[str, Signal]


@dataclass(frozen=True)
class CalibrationPolicy:
    """Tolerance = max(multiplier x observed spread, floors[kind], multiplier x weight-resolution floor).

    Attributes:
        weight_resolution: Relative spacing of the stored parameter dtype; see the module docstring.
    """

    multiplier: float
    floors: dict[SignalKind, float]
    weight_resolution: float


# A starting point for Hero-scale bf16 runs; check it with the must-fail controls before trusting it.
RECOMMENDED_CALIBRATION = CalibrationPolicy(
    multiplier=4.0,
    floors={SignalKind.RELATIVE: 1e-3, SignalKind.ABSOLUTE: 1e-4, SignalKind.RELATIVE_L2: 1e-3},
    weight_resolution=BF16_RELATIVE_SPACING,
)


@dataclass(frozen=True)
class NumericsReference:
    """A reference record with calibrated tolerances.

    Attributes:
        record: The first calibration run; candidates are compared against its signal values.
        tolerances: Maximum accepted deviation per signal.
        spreads: Largest pairwise deviation per signal across the calibration runs.
        calibration: The policy that turned spreads into tolerances.
        calibration_layouts: Layouts of the calibration runs, in order.
    """

    schema_version: int
    record: NumericsRecord
    tolerances: dict[str, float]
    spreads: dict[str, float]
    calibration: CalibrationPolicy
    calibration_layouts: tuple[str, ...]


@dataclass(frozen=True)
class SignalCheck:
    name: str
    deviation: float
    tolerance: float

    @property
    def passed(self) -> bool:
        return self.deviation <= self.tolerance


@dataclass(frozen=True)
class NumericsComparison:
    checks: tuple[SignalCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> tuple[SignalCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)


def parameter_signals(statistics: Mapping[str, ParameterGroupStatistics]) -> dict[str, Signal]:
    signals = {}
    for group, group_statistics in sorted(statistics.items()):
        signals[f"params/{group}/numel"] = Signal(SignalKind.ABSOLUTE, (group_statistics.numel,))
        signals[f"params/{group}/weight_norm"] = Signal(
            SignalKind.RELATIVE, (math.sqrt(group_statistics.before_square_sum),)
        )
        signals[f"params/{group}/delta_norm"] = Signal(
            SignalKind.RELATIVE, (math.sqrt(group_statistics.delta_square_sum),)
        )
        signals[f"params/{group}/sketch"] = Signal(SignalKind.RELATIVE_L2, group_statistics.sketches)
    total = sum(group_statistics.delta_square_sum for group_statistics in statistics.values())
    signals["params/total_delta_norm"] = Signal(SignalKind.RELATIVE, (math.sqrt(total),))
    total_weight = sum(group_statistics.before_square_sum for group_statistics in statistics.values())
    signals["params/total_weight_norm"] = Signal(SignalKind.RELATIVE, (math.sqrt(total_weight),))
    return signals


# ---------------------------------------------------------------------------------------------------------------
# Log-prob and train-status signals


def logprob_sample_positions(mask: torch.Tensor, sample_size: int) -> torch.Tensor:
    """Flat positions of up to ``sample_size`` masked tokens, chosen by a position hash and sorted."""
    positions = torch.nonzero(mask.reshape(-1), as_tuple=True)[0]
    if positions.numel() <= sample_size:
        return positions
    keys = mix32(positions.to(torch.int64) ^ LOGPROB_SAMPLE_SALT)
    chosen = positions[torch.topk(keys, sample_size, largest=False).indices]
    return torch.sort(chosen).values


def logprob_signals(prefix: str, logprobs: torch.Tensor, mask: torch.Tensor) -> dict[str, Signal]:
    """Summarize response log-probs ``[batch, response_length]`` over the loss mask."""
    if logprobs.shape != mask.shape:
        raise ValueError(f"Log-prob shape {tuple(logprobs.shape)} does not match mask {tuple(mask.shape)}")
    values = logprobs.to(torch.float64)
    weights = mask.to(torch.float64)
    token_count = weights.sum()
    if token_count == 0:
        raise ValueError("Log-prob mask selects no tokens")
    per_sequence_count = weights.sum(dim=1).clamp_min(1.0)
    positions = logprob_sample_positions(mask.bool(), LOGPROB_SAMPLE_SIZE)
    return {
        f"{prefix}/token_mean": Signal(SignalKind.ABSOLUTE, (float((values * weights).sum() / token_count),)),
        f"{prefix}/sequence_mean": Signal(
            SignalKind.ABSOLUTE, tuple(((values * weights).sum(dim=1) / per_sequence_count).tolist())
        ),
        f"{prefix}/sampled_tokens": Signal(SignalKind.ABSOLUTE, tuple(values.reshape(-1)[positions].tolist())),
    }


def train_status_signals(update_index: int, train_status: Mapping[str, float]) -> dict[str, Signal]:
    missing = [key for key in REQUIRED_TRAIN_STATUS_KEYS if key not in train_status]
    if missing:
        raise KeyError(f"train_status lacks required keys {missing}")
    return {
        f"train_status/update{update_index}/{key}": Signal(SignalKind(kind), (float(train_status[key]),))
        for key, kind in TRAIN_STATUS_SIGNAL_KINDS.items()
        if key in train_status
    }


def numerics_record(
    provenance: NumericsProvenance,
    train_statuses: Sequence[Mapping[str, float]],
    pre_update_logprobs: torch.Tensor,
    post_update_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    parameter_statistics: Mapping[str, ParameterGroupStatistics],
) -> NumericsRecord:
    if len(train_statuses) != provenance.num_updates:
        raise ValueError(f"Expected {provenance.num_updates} train statuses, got {len(train_statuses)}")
    signals: dict[str, Signal] = {}
    for index, status in enumerate(train_statuses):
        signals.update(train_status_signals(index, status))
    signals.update(logprob_signals("logprobs/pre_update", pre_update_logprobs, loss_mask))
    signals.update(logprob_signals("logprobs/post_update", post_update_logprobs, loss_mask))
    signals.update(parameter_signals(parameter_statistics))
    return NumericsRecord(provenance=provenance, signals=signals)


def numerics_config_fingerprint(cfg: DictConfig) -> str:
    """Hash the config subtrees in :data:`NUMERICS_CONFIG_KEYS`; a missing key is an error."""
    selected = {}
    for key in NUMERICS_CONFIG_KEYS:
        value = OmegaConf.select(cfg, key, throw_on_missing=True)
        if value is None:
            raise KeyError(f"Config lacks numerics key {key}")
        selected[key] = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    payload = json.dumps(selected, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def perturbed_batch(batch: TrainingInputBatch, control: ControlPerturbation) -> TrainingInputBatch:
    """Return a copy of ``batch`` with the control's advantage perturbation applied."""
    scale = {ControlPerturbation.NEGATE_ADVANTAGES: -1.0, ControlPerturbation.DOUBLE_ADVANTAGES: 2.0}[control]
    tensors = dict(batch)
    tensors["advantages"] = batch["advantages"] * scale
    perturbed = TrainingInputBatch(tensors, routed_expert_rows=batch.routed_expert_rows)
    perturbed.metadata = dict(batch.metadata)
    return perturbed


# ---------------------------------------------------------------------------------------------------------------
# Comparison and calibration


def signal_deviation(reference: Signal, candidate: Signal) -> float:
    """Deviation of ``candidate`` from ``reference``; ``inf`` for a structural mismatch or non-finite value."""
    if reference.kind != candidate.kind or len(reference.values) != len(candidate.values):
        return math.inf
    if not all(math.isfinite(value) for value in candidate.values):
        return math.inf
    if reference.kind == SignalKind.ABSOLUTE:
        return max((abs(c - r) for r, c in zip(reference.values, candidate.values)), default=0.0)
    if reference.kind == SignalKind.RELATIVE:
        (r,), (c,) = reference.values, candidate.values
        return _relative(abs(c - r), abs(r))
    difference = math.sqrt(sum((c - r) ** 2 for r, c in zip(reference.values, candidate.values)))
    return _relative(difference, math.sqrt(sum(r * r for r in reference.values)))


def _relative(difference: float, scale: float) -> float:
    if scale == 0.0:
        return 0.0 if difference == 0.0 else math.inf
    return difference / scale


def _check_provenance(reference: NumericsProvenance, candidate: NumericsProvenance) -> None:
    if reference.matching_fields() != candidate.matching_fields():
        raise ValueError(
            "Numerics provenance differs (checkpoint, batch_sha256, config_fingerprint, num_updates): "
            f"{reference.matching_fields()} != {candidate.matching_fields()}"
        )


def calibrate_reference(runs: Sequence[NumericsRecord], policy: CalibrationPolicy) -> NumericsReference:
    """Build a reference from repeated runs of one configuration, ideally on different layouts."""
    if len(runs) < 2:
        raise ValueError("Calibration needs at least two runs")
    first = runs[0]
    for run in runs[1:]:
        _check_provenance(first.provenance, run.provenance)
        if run.signals.keys() != first.signals.keys():
            raise ValueError(
                f"Calibration runs disagree on signals: {sorted(run.signals.keys() ^ first.signals.keys())}"
            )
    spreads = {
        name: max(signal_deviation(a.signals[name], b.signals[name]) for a, b in itertools.permutations(runs, 2))
        for name in first.signals
    }
    unstable = sorted(name for name, spread in spreads.items() if not math.isfinite(spread))
    if unstable:
        raise ValueError(f"Calibration runs disagree structurally or are non-finite for {unstable}")
    tolerances = {
        name: max(
            policy.multiplier * spread,
            policy.floors[first.signals[name].kind],
            policy.multiplier * weight_resolution_floor(first.signals, name, policy.weight_resolution),
        )
        for name, spread in spreads.items()
    }
    return NumericsReference(
        schema_version=SCHEMA_VERSION,
        record=first,
        tolerances=tolerances,
        spreads=spreads,
        calibration=policy,
        calibration_layouts=tuple(run.provenance.layout for run in runs),
    )


def weight_resolution_floor(signals: Mapping[str, Signal], name: str, weight_resolution: float) -> float:
    """Rounding-flip shot noise ``weight_resolution * rms(w) / ||delta||`` of a parameter delta signal, else 0.

    The caller scales it by the calibration multiplier. A group that did not move gets no floor: any candidate
    movement is then an infinite deviation regardless.
    """
    if name == "params/total_delta_norm":
        weight_norm = signals["params/total_weight_norm"].values[0]
        numel = sum(signal.values[0] for key, signal in signals.items() if key.endswith("/numel"))
        delta_norm = signals[name].values[0]
    elif name.startswith("params/") and name.endswith(("/delta_norm", "/sketch")):
        group = name.removeprefix("params/").rsplit("/", 1)[0]
        weight_norm = signals[f"params/{group}/weight_norm"].values[0]
        numel = signals[f"params/{group}/numel"].values[0]
        delta_norm = signals[f"params/{group}/delta_norm"].values[0]
    else:
        return 0.0
    if delta_norm == 0 or numel == 0:
        return 0.0
    return weight_resolution * weight_norm / math.sqrt(numel) / delta_norm


def compare_to_reference(reference: NumericsReference, candidate: NumericsRecord) -> NumericsComparison:
    """Check every reference signal; a signal missing from or added by the candidate fails."""
    _check_provenance(reference.record.provenance, candidate.provenance)
    checks = []
    for name, expected in sorted(reference.record.signals.items()):
        observed = candidate.signals.get(name)
        deviation = math.inf if observed is None else signal_deviation(expected, observed)
        checks.append(SignalCheck(name=name, deviation=deviation, tolerance=reference.tolerances[name]))
    checks.extend(
        SignalCheck(name=name, deviation=math.inf, tolerance=0.0)
        for name in sorted(candidate.signals.keys() - reference.record.signals.keys())
    )
    return NumericsComparison(checks=tuple(checks))


# ---------------------------------------------------------------------------------------------------------------
# JSON storage


def _record_payload(record: NumericsRecord) -> dict:
    return {
        "provenance": asdict(record.provenance),
        "signals": {
            name: {"kind": str(signal.kind), "values": list(signal.values)} for name, signal in record.signals.items()
        },
    }


def _record_from_payload(payload: Mapping) -> NumericsRecord:
    return NumericsRecord(
        provenance=NumericsProvenance(**payload["provenance"]),
        signals={
            name: Signal(SignalKind(signal["kind"]), tuple(float(value) for value in signal["values"]))
            for name, signal in payload["signals"].items()
        },
    )


def reference_json(reference: NumericsReference) -> str:
    payload = {
        "schema_version": reference.schema_version,
        "record": _record_payload(reference.record),
        "tolerances": reference.tolerances,
        "spreads": reference.spreads,
        "calibration": {
            "multiplier": reference.calibration.multiplier,
            "floors": {str(kind): floor for kind, floor in reference.calibration.floors.items()},
            "weight_resolution": reference.calibration.weight_resolution,
        },
        "calibration_layouts": list(reference.calibration_layouts),
    }
    return json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n"


def parse_reference(text: str) -> NumericsReference:
    payload = json.loads(text)
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported numerics reference schema {payload['schema_version']}")
    calibration = payload["calibration"]
    return NumericsReference(
        schema_version=payload["schema_version"],
        record=_record_from_payload(payload["record"]),
        tolerances={name: float(value) for name, value in payload["tolerances"].items()},
        spreads={name: float(value) for name, value in payload["spreads"].items()},
        calibration=CalibrationPolicy(
            multiplier=float(calibration["multiplier"]),
            floors={SignalKind(kind): float(floor) for kind, floor in calibration["floors"].items()},
            weight_resolution=float(calibration["weight_resolution"]),
        ),
        calibration_layouts=tuple(payload["calibration_layouts"]),
    )


def write_reference(path: str, reference: NumericsReference) -> None:
    io.write_bytes_atomic(path, reference_json(reference).encode())


def read_reference(path: str) -> NumericsReference:
    return parse_reference(io.read_bytes(path).decode())


def record_json(record: NumericsRecord) -> str:
    """Serialize one run, e.g. a calibration run kept for later re-calibration."""
    return json.dumps({"schema_version": SCHEMA_VERSION, **_record_payload(record)}, indent=1, allow_nan=False) + "\n"


def parse_record(text: str) -> NumericsRecord:
    payload = json.loads(text)
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported numerics record schema {payload['schema_version']}")
    return _record_from_payload(payload)


# ---------------------------------------------------------------------------------------------------------------
# Driver (GPU; runs through the production policy worker)


def _response_logprobs(policy, batch: TrainingInputBatch) -> torch.Tensor:
    outputs = ray.get(policy.async_run_ray_method("mesh", "forward", data=batch))
    return concatenate_outputs_after_mesh_dispatch(policy.actor_infos, outputs)["output"].float()


def record_numerics(policy, batch: TrainingInputBatch, provenance: NumericsProvenance) -> NumericsRecord:
    """Score, update ``provenance.num_updates`` times, fingerprint the parameter delta, and score again.

    ``policy`` is the Megatron policy actor group; ``batch`` must already be padded to its data-parallel size
    and carry the replay routes. The batch's ``action_log_probs`` are replaced in place by the pre-update scores,
    so the first update is on-policy: a stored ensemble batch carries the log-probs of the model that generated
    its text, not of the model under test.
    """
    pre_update = _response_logprobs(policy, batch)
    batch["action_log_probs"] = pre_update * batch["response_mask"]
    ray.get(policy.async_run_ray_method("pass_through", "capture_numerics_baseline"))
    statuses = []
    for _ in range(provenance.num_updates):
        outputs = ray.get(policy.async_run_ray_method("mesh", "ppo_train", batch))
        statuses.append(outputs[0].metadata["train_status"])
    per_rank = ray.get(policy.async_run_ray_method("pass_through", "numerics_parameter_statistics"))
    post_update = _response_logprobs(policy, batch)
    return numerics_record(
        provenance,
        statuses,
        pre_update,
        post_update,
        batch["loss_mask"],
        merge_parameter_statistics(per_rank),
    )


# ---------------------------------------------------------------------------------------------------------------
# Command line: record a run with the learner driver's policy settings, calibrate, compare


def batch_identity(manifest_sha256: str, batch: TrainingInputBatch) -> str:
    """Hash a stored batch's tensor digest together with the replay routes it carries, if any."""
    digest = hashlib.sha256(manifest_sha256.encode())
    routes = batch.routed_expert_rows
    if routes is not None:
        digest.update(f"{routes.num_experts}:{routes.response_len}".encode())
        for rows in routes.rows:
            digest.update(str(rows.shape).encode())
            digest.update(rows.tobytes())
    return digest.hexdigest()


def checkpoint_identity(model_path: str, num_layers: int) -> str:
    manifest_path = f"{model_path.rstrip('/')}/{MODEL_MANIFEST_FILENAME}"
    manifest = ModelManifest.from_mapping(json.loads(io.read_bytes(manifest_path)), manifest_path)
    return f"{manifest.identity}/layers{num_layers}"


def layout_name(spec: driver.BenchmarkSpec) -> str:
    layout = spec.layout
    return (
        f"tp{layout.tensor_parallel}-pp{layout.pipeline_parallel}-ep{layout.expert_parallel}"
        f"-cp{layout.context_parallel}-etp{layout.expert_tensor_parallel}-mbs{spec.micro_batch_size_per_gpu}"
    )


def _record_command(args: argparse.Namespace, driver_argv: Sequence[str]) -> None:
    spec = driver.parse_spec(driver_argv)
    entries = driver.ensemble_entries(spec)
    entry = min(entries, key=lambda e: (e.profile, e.batch_index))
    rows = driver.rows_per_update([entry], spec.layout, spec.micro_batch_size_per_gpu)
    cfg = driver.build_policy_cfg(spec, rows, driver.compose_base_config())
    model_config = AutoConfig.from_pretrained(spec.model_path)
    full_layers = driver.source_num_layers(spec.model_path, model_config.num_hidden_layers)
    batch, manifest = driver.load_replay_batch(
        entry.path, spec.routes_checkpoint, model_config.num_hidden_layers, full_layers
    )
    provenance = NumericsProvenance(
        checkpoint=checkpoint_identity(spec.model_path, model_config.num_hidden_layers),
        batch_sha256=batch_identity(manifest.tensors_sha256, batch),
        config_fingerprint=numerics_config_fingerprint(cfg),
        num_updates=args.num_updates,
        layout=layout_name(spec),
        source_revision=spec.harness_revision,
    )
    if args.control is not None:
        batch = perturbed_batch(batch, ControlPerturbation(args.control))
    batch.metadata["global_step"] = 0
    padded = pad_training_input(batch, spec.layout.data_parallel * spec.micro_batch_size_per_gpu)
    initialize_ray(cfg)
    try:
        policy = driver._init_policy_group(cfg, spec.layout)
        record = record_numerics(policy, padded, provenance)
    finally:
        ray.shutdown()
    io.write_bytes_atomic(args.record_output, record_json(record).encode())
    logger.info(f"numerics record ({provenance.layout}, control={args.control}) written to {args.record_output}")


def _calibrate_command(args: argparse.Namespace) -> None:
    runs = [parse_record(io.read_bytes(path).decode()) for path in args.records]
    policy = CalibrationPolicy(
        multiplier=args.multiplier,
        floors=RECOMMENDED_CALIBRATION.floors,
        weight_resolution=RECOMMENDED_CALIBRATION.weight_resolution,
    )
    write_reference(args.reference, calibrate_reference(runs, policy))
    logger.info(f"numerics reference from {len(runs)} runs written to {args.reference}")


def comparison_json(comparison: NumericsComparison) -> str:
    """Every check with its deviation, tolerance and margin (tolerance / deviation), failures first."""
    checks = sorted(comparison.checks, key=lambda check: (check.passed, check.name))
    payload = {
        "passed": comparison.passed,
        "failures": [check.name for check in comparison.failures],
        "checks": [
            {
                "name": check.name,
                "deviation": check.deviation if math.isfinite(check.deviation) else None,
                "tolerance": check.tolerance,
                "margin": check.tolerance / check.deviation if 0 < check.deviation < math.inf else None,
                "passed": check.passed,
            }
            for check in checks
        ],
    }
    return json.dumps(payload, indent=1, allow_nan=False) + "\n"


def _compare_command(args: argparse.Namespace) -> None:
    reference = read_reference(args.reference)
    comparison = compare_to_reference(reference, parse_record(io.read_bytes(args.record).decode()))
    text = comparison_json(comparison)
    if args.output is not None:
        io.write_bytes_atomic(args.output, text.encode())
    logger.info(f"numerics {'PASSED' if comparison.passed else 'FAILED'}: {len(comparison.failures)} failing signals")
    if not comparison.passed:
        sys.exit(1)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Record, calibrate and compare learner post-update numerics.")
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser(
        "record", help="record one run; arguments not listed here are the learner driver's (its first batch is used)"
    )
    record.add_argument("--record-output", required=True)
    record.add_argument("--num-updates", type=int, required=True)
    record.add_argument("--control", choices=[control.value for control in ControlPerturbation])
    calibrate = commands.add_parser("calibrate", help="build a reference from two or more records")
    calibrate.add_argument("--records", nargs="+", required=True)
    calibrate.add_argument("--reference", required=True)
    calibrate.add_argument("--multiplier", type=float, default=RECOMMENDED_CALIBRATION.multiplier)
    compare = commands.add_parser("compare", help="compare a record with a reference; exits 1 on failure")
    compare.add_argument("--reference", required=True)
    compare.add_argument("--record", required=True)
    compare.add_argument("--output", help="write every check as JSON")
    args, rest = parser.parse_known_args(argv)
    if args.command == "record":
        _record_command(args, rest)
        return
    if rest:
        parser.error(f"unrecognized arguments: {rest}")
    if args.command == "calibrate":
        _calibrate_command(args)
    else:
        _compare_command(args)


if __name__ == "__main__":
    main()
