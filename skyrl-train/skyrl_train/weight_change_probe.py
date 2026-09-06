"""Bounded surveys of exact wire-weight changes; no codec or transport behavior.

Capture only selects integer bit representations on the input device. The caller
must complete any CUDA-to-CPU staging before add_cpu_sample, then finish only
after the entire publication succeeds. No CUDA synchronization occurs here.
Estimates describe changed elements and a hypothetical tensor-local uint32 index
plus replacement-value payload, excluding headers, compression and application.
"""

import hashlib
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum

import torch
from omegaconf import DictConfig
from skyrl_train.telemetry import record_event

SAMPLER_VERSION = "wire_bits_v1"
MAX_SNAPSHOT_BYTES = 2 * 1024 * 1024
MAX_TENSORS = 4096
MAX_TENSOR_SAMPLES = 2048
BIT_DTYPES = {torch.bfloat16: torch.int16, torch.float32: torch.int32}
type EventValue = str | int | float


class ProbeStatus(StrEnum):
    BASELINE = "baseline"
    COMPARED = "compared"
    INCOMPLETE = "incomplete"
    FAILED = "failed"


@dataclass(frozen=True)
class TensorSamplePlan:
    name: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: torch.dtype
    element_bytes: int
    elements: int
    family: str
    size_bin: str
    strata: int
    samples_per_stratum: int

    @property
    def sample_count(self) -> int:
        return self.strata * self.samples_per_stratum

    @property
    def sample_bytes(self) -> int:
        return self.sample_count * self.element_bytes


@dataclass(frozen=True)
class CapturedWireSample:
    plan: TensorSamplePlan
    bits: torch.Tensor
    generation: int


@dataclass
class _Sample:
    plan: TensorSamplePlan
    bits: torch.Tensor | None = None


@dataclass
class _Publication:
    publication_id: str
    base_update: int | None
    target_update: int
    generation: int
    samples: dict[str, _Sample] = field(default_factory=dict)
    problems: set[str] = field(default_factory=set)
    reserved_bytes: int = 0
    population_elements: int = 0
    dense_wire_bytes: int = 0
    tensor_count: int = 0


@dataclass(frozen=True)
class ProbeResult:
    """Flat event bodies; groups are bounded by family, dtype and size bin."""

    summary: dict[str, EventValue]
    groups: tuple[dict[str, EventValue], ...] = ()


def _family(name: str) -> str:
    if name.endswith(".mlp.router.bias"):
        return "router_bias"
    for part, family in ((".mlp.experts.", "routed"), (".shared_expert.", "shared")):
        if part in name:
            for projection in ("gate", "up", "down"):
                if name.endswith(f".{projection}_proj.weight"):
                    return f"{family}_{projection}"
            return family
    for part, family in (
        (".self_attn.", "attention"),
        (".router.", "router"),
        ("norm", "norm"),
        ("embed_tokens", "embedding"),
        ("lm_head", "output"),
    ):
        if part in name:
            return family
    return "other"


def _plan(name: str, tensor: torch.Tensor) -> TensorSamplePlan:
    elements = tensor.numel()
    if elements <= 256:
        size_bin, quota = "tiny", elements
    elif elements <= 65536:
        size_bin, quota = "small", 256
    elif elements <= 1048576:
        size_bin, quota = "medium", 512
    else:
        size_bin, quota = "large", MAX_TENSOR_SAMPLES
    family = _family(name)
    if family == "router_bias" and tensor.dtype == torch.float32 and elements <= 4096:
        quota = elements
    strata = tensor.shape[0] if family.startswith("routed_") and tensor.ndim == 3 else 1
    per_stratum = min(elements // strata, max(2, quota // strata)) if strata else 0
    return TensorSamplePlan(
        name,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.element_size(),
        elements,
        family,
        size_bin,
        strata,
        per_stratum,
    )


def _plan_identity(plan: TensorSamplePlan) -> str:
    return json.dumps(
        [plan.name, plan.shape, plan.strides, str(plan.dtype), plan.strata, plan.samples_per_stratum],
        separators=(",", ":"),
    )


class WeightChangeProbe:
    """Commit CPU sample history only after successful, complete publications.

    Every tensor is sampled with private deterministic uniform draws without
    replacement. Grug's stacked expert tensors sample every expert slice. Point
    estimates use each stratum's full population/sample ratio and actual wire
    width. Incomplete coverage suppresses all extrapolated estimates.
    The caller must capture every wire tensor before finishing; this helper
    checks registered receipts and prior inventory, not a separate model graph.
    """

    def __init__(self, *, seed: int = 0, max_sample_bytes: int = MAX_SNAPSHOT_BYTES) -> None:
        if not 0 < max_sample_bytes <= MAX_SNAPSHOT_BYTES:
            raise ValueError(f"max_sample_bytes must be in [1, {MAX_SNAPSHOT_BYTES}]")
        self._seed = seed
        self._max_sample_bytes = max_sample_bytes
        self._previous: _Publication | None = None
        self._pending: _Publication | None = None
        self._generation = 0

    @property
    def retained_sample_bytes(self) -> int:
        return sum(
            sample.bits.numel() * sample.bits.element_size()
            for publication in (self._previous, self._pending)
            if publication is not None
            for sample in publication.samples.values()
            if sample.bits is not None
        )

    def begin(self, *, publication_id: str, base_update: int | None, target_update: int) -> None:
        """Start a publication with the driver's explicit installed/update identities."""
        if self._pending is not None:
            raise RuntimeError("a wire-weight publication is already pending")
        if not publication_id or len(publication_id.encode()) > 256:
            raise ValueError("publication_id must contain 1 to 256 bytes")
        if target_update < 0 or (base_update is not None and not 0 <= base_update <= target_update):
            raise ValueError("invalid base/target update ordering")
        self._generation += 1
        self._pending = _Publication(publication_id, base_update, target_update, self._generation)

    def capture(self, name: str, tensor: torch.Tensor, *, collect_bits: bool = True) -> CapturedWireSample | None:
        """Select independent bit storage on the input device without a CPU read.

        A None result marks incomplete coverage. No full flattening copy, dense
        index permutation, wire dtype cast, or CUDA-to-CPU transfer is allowed.
        """
        pending = self._pending
        if pending is None:
            raise RuntimeError("begin a publication before capturing weights")
        pending.tensor_count += 1
        pending.population_elements += tensor.numel()
        pending.dense_wire_bytes += tensor.numel() * tensor.element_size()
        if name in pending.samples or not name or len(name.encode()) > 512:
            pending.problems.add("invalid_or_duplicate_name")
            return None
        if len(pending.samples) >= MAX_TENSORS:
            pending.problems.add("tensor_limit")
            return None
        plan = _plan(name, tensor)
        pending.samples[name] = _Sample(plan)
        if not collect_bits:
            pending.problems.add("sampling_disabled")
            return None
        if tensor.dtype not in BIT_DTYPES or (plan.family == "router_bias" and tensor.dtype != torch.float32):
            pending.problems.add("unsupported_wire_dtype")
            return None
        if not tensor.is_contiguous() or tensor.numel() == 0:
            pending.problems.add("unsupported_layout")
            return None
        if plan.family.startswith("routed_") and (tensor.ndim != 3 or plan.strata > MAX_TENSOR_SAMPLES // 2):
            pending.problems.add("unsupported_expert_geometry")
            return None
        if pending.reserved_bytes + plan.sample_bytes > self._max_sample_bytes:
            pending.problems.add("sample_budget")
            return None
        pending.reserved_bytes += plan.sample_bytes
        key = f"{SAMPLER_VERSION}:{self._seed}:{_plan_identity(plan)}".encode()
        rng = random.Random(int.from_bytes(hashlib.sha256(key).digest(), "big"))
        population = plan.elements // plan.strata
        indices = [
            expert * population + index
            for expert in range(plan.strata)
            for index in rng.sample(range(population), plan.samples_per_stratum)
        ]
        index_tensor = torch.tensor(indices, dtype=torch.int64, device="cpu", pin_memory=tensor.device.type == "cuda")
        index_tensor = index_tensor.to(device=tensor.device, non_blocking=True)
        bits = tensor.detach().view(-1).view(BIT_DTYPES[tensor.dtype]).index_select(0, index_tensor)
        return CapturedWireSample(plan, bits, pending.generation)

    def add_cpu_sample(self, capture: CapturedWireSample, bits: torch.Tensor, *, take_ownership: bool = False) -> None:
        """Own a CPU sample after the caller has completed its asynchronous staging."""
        if bits.device.type != "cpu":
            raise ValueError("sample staging must complete on CPU before add_cpu_sample")
        if (
            self._pending is None
            or capture.generation != self._pending.generation
            or capture.plan.name not in self._pending.samples
        ):
            raise RuntimeError("sample does not belong to the pending publication")
        sample = self._pending.samples[capture.plan.name]
        if sample.plan != capture.plan or sample.bits is not None:
            raise ValueError("sample plan mismatch or duplicate CPU sample")
        if bits.dtype != BIT_DTYPES[sample.plan.dtype] or tuple(bits.shape) != (sample.plan.sample_count,):
            raise ValueError("sample bits do not match their wire dtype/count")
        # The publication observer relinquishes its pinned staging buffer here;
        # callers retaining access must use the default independent copy.
        sample.bits = bits.detach() if take_ownership else bits.detach().clone()

    def invalidate(self, reason: str) -> None:
        if self._pending is None:
            raise RuntimeError("no wire-weight publication is pending")
        self._pending.problems.add(reason)

    def finish(self, *, success: bool) -> ProbeResult:
        """Promote a successful publication; otherwise retain the committed base."""
        pending = self._pending
        if pending is None:
            raise RuntimeError("no wire-weight publication is pending")
        self._pending = None
        if not pending.samples or any(sample.bits is None for sample in pending.samples.values()):
            pending.problems.add("missing_samples")
        previous = self._previous
        comparable = previous is not None and previous.target_update == pending.base_update
        reason = "" if comparable else "missing_or_mismatched_base"
        if comparable and (
            previous.samples.keys() != pending.samples.keys()
            or any(previous.samples[name].plan != sample.plan for name, sample in pending.samples.items())
        ):
            comparable, reason = False, "inventory_or_layout_changed"
        status = ProbeStatus.COMPARED if comparable else ProbeStatus.BASELINE
        if not success:
            status = ProbeStatus.FAILED
            reason = "publication_failed"
        elif pending.problems:
            status = ProbeStatus.INCOMPLETE
        manifest = "\n".join(_plan_identity(s.plan) for _, s in sorted(pending.samples.items()))
        summary: dict[str, EventValue] = {
            "status": status.value,
            "publication_id": pending.publication_id,
            "target_update": pending.target_update,
            "sampler_version": SAMPLER_VERSION,
            "sample_seed": self._seed,
            "sampling_manifest": hashlib.sha256(manifest.encode()).hexdigest(),
            "tensor_count": pending.tensor_count,
            "population_elements": pending.population_elements,
            "dense_wire_bytes": pending.dense_wire_bytes,
            "sample_bytes": pending.reserved_bytes,
            "coverage_complete": int(not pending.problems),
        }
        if detail := ",".join(sorted(pending.problems)) or reason:
            summary["reason"] = detail
        if pending.base_update is not None:
            summary["base_update"] = pending.base_update
        groups = ()
        if status == ProbeStatus.COMPARED:
            assert previous is not None
            totals: dict[tuple[str, str, str], dict[str, int | float]] = defaultdict(lambda: defaultdict(int))
            for name, sample in pending.samples.items():
                plan = sample.plan
                changed = int(torch.count_nonzero(sample.bits != previous.samples[name].bits).item())
                estimated = changed * (plan.elements // plan.strata) / plan.samples_per_stratum
                group = totals[(plan.family, str(plan.dtype), plan.size_bin)]
                for key, value in {
                    "tensors": 1,
                    "strata": plan.strata,
                    "sampled_elements": plan.sample_count,
                    "sampled_changed_elements": changed,
                    "population_elements": plan.elements,
                    "dense_wire_bytes": plan.elements * plan.element_bytes,
                    "estimated_changed_elements": estimated,
                    "estimated_changed_value_bytes": estimated * plan.element_bytes,
                    "index32_eligible_tensors": int(plan.elements <= 2**32),
                    "estimated_index32_value_bytes": estimated * (plan.element_bytes + 4),
                }.items():
                    group[key] += value
            groups = tuple(
                {"family": family, "dtype": dtype, "size_bin": size, **values}
                for (family, dtype, size), values in sorted(totals.items())
            )
            for key in (
                "sampled_elements",
                "sampled_changed_elements",
                "estimated_changed_elements",
                "estimated_changed_value_bytes",
                "estimated_index32_value_bytes",
            ):
                summary[key] = sum(group[key] for group in groups)
            for group in groups:
                group["sampled_changed_element_fraction"] = (
                    group["sampled_changed_elements"] / group["sampled_elements"]
                )
                group["estimated_changed_element_fraction"] = (
                    group["estimated_changed_elements"] / group["population_elements"]
                )
                group["estimated_changed_value_fraction"] = (
                    group["estimated_changed_value_bytes"] / group["dense_wire_bytes"]
                )
                if group["index32_eligible_tensors"] != group["tensors"]:
                    group.pop("estimated_index32_value_bytes")
                    summary.pop("estimated_index32_value_bytes", None)
            summary["estimated_changed_element_fraction"] = (
                summary["estimated_changed_elements"] / pending.population_elements
            )
            summary["estimated_changed_value_fraction"] = (
                summary["estimated_changed_value_bytes"] / pending.dense_wire_bytes
            )
        if success:
            self._previous = pending if not pending.problems else None
        summary["retained_sample_bytes"] = self.retained_sample_bytes
        return ProbeResult(summary, groups)


def validate_weight_change_probe_config(cfg: DictConfig) -> bool:
    """Validate the observational probe's supported transport before allocation."""
    enabled = cfg.trainer.get("weight_change_probe", False)
    if not isinstance(enabled, bool):
        raise ValueError("trainer.weight_change_probe must be a boolean")
    if not enabled:
        return False
    if (
        cfg.trainer.strategy != "megatron"
        or cfg.trainer.placement.colocate_all
        or cfg.generator.fuse_weights
        or not cfg.generator.run_engines_locally
    ):
        raise ValueError("weight_change_probe requires local, non-colocated, unfused Megatron broadcast")
    return True


class WirePublicationObserver:
    """Stage bounded wire samples, then commit only on the driver's success ack.

    CUDA event pairs cover sample indexing and D2H staging on the publication
    stream. Reading them requires the worker's existing terminal CUDA sync;
    this observer never adds a synchronize. CPU staging ownership transfers to
    the sampler without a second copy (at most two 2 MiB CPU snapshots).
    """

    def __init__(self, *, seed: int) -> None:
        self.probe = WeightChangeProbe(seed=seed)
        self._publication_id: str | None = None
        self._staged: list[tuple[CapturedWireSample, torch.Tensor, object, object]] = []
        self._ready = False
        self._disabled = False
        self._enqueue_seconds = 0.0
        self._staging_seconds = 0.0
        self._cuda_milliseconds = 0.0
        self._device_sample_bytes = 0

    def begin(self, *, publication_id: str, base_update: int | None, target_update: int) -> None:
        self.probe.begin(publication_id=publication_id, base_update=base_update, target_update=target_update)
        self._publication_id = publication_id
        self._ready = False
        self._enqueue_seconds = self._staging_seconds = self._cuda_milliseconds = 0.0
        self._device_sample_bytes = 0
        if self._disabled:
            self.probe.invalidate("observer_resource_failure")

    def capture(self, name: str, tensor: torch.Tensor) -> None:
        if self._disabled:
            # Continue inventorying the full transmitted population even when
            # sample allocation failed; incomplete coverage still forbids estimates.
            self.probe.capture(name, tensor, collect_bits=False)
            return
        started = time.perf_counter()
        try:
            start = end = None
            if tensor.device.type == "cuda":
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
            sample = self.probe.capture(name, tensor)
            if sample is not None:
                cpu = torch.empty_like(sample.bits, device="cpu", pin_memory=tensor.device.type == "cuda")
                cpu.copy_(sample.bits, non_blocking=True)
                if end is not None:
                    end.record()
                self._staged.append((sample, cpu, start, end))
                if tensor.device.type == "cuda":
                    self._device_sample_bytes += sample.plan.sample_bytes
        except (torch.OutOfMemoryError, MemoryError):
            self._disabled = True
            self.probe.invalidate("observer_resource_failure")
        finally:
            self._enqueue_seconds += time.perf_counter() - started

    def staging_complete(self) -> None:
        """Called after the existing publication synchronization, before any commit."""
        started = time.perf_counter()
        for sample, cpu, start, end in self._staged:
            if end is not None:
                if not end.query():
                    raise RuntimeError("wire probe staging was not covered by publication synchronization")
                self._cuda_milliseconds += start.elapsed_time(end)
            self.probe.add_cpu_sample(sample, cpu, take_ownership=True)
        self._staged.clear()
        self._staging_seconds = time.perf_counter() - started
        self._ready = True

    def finish(self, *, publication_id: str, success: bool) -> ProbeResult:
        if publication_id != self._publication_id:
            raise ValueError("wire probe acknowledgment does not match the pending publication")
        if success and not self._ready:
            raise RuntimeError("wire probe cannot commit before publication staging completes")
        started = time.perf_counter()
        result = self.probe.finish(success=success)
        result.summary.update(
            capture_enqueue_seconds=self._enqueue_seconds,
            staging_receipt_seconds=self._staging_seconds,
            sample_cuda_milliseconds=self._cuda_milliseconds,
            compare_commit_seconds=time.perf_counter() - started,
            staged_device_sample_bytes=self._device_sample_bytes,
            device_sample_budget_bytes=MAX_SNAPSHOT_BYTES,
            cpu_snapshot_budget_bytes=2 * MAX_SNAPSHOT_BYTES,
            wire_population="all_rank0_converted_broadcast_tensors",
        )
        self._staged.clear()
        self._publication_id = None
        record_event("weight_change_probe", result.summary)
        for group in result.groups:
            record_event("weight_change_probe_stratum", {"publication_id": publication_id, **group})
        return result
