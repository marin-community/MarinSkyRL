"""Per-collective NCCL inventory read from torch's flight recorder.

The policy spans cannot bound NCCL's share of a training step. FSDP2 issues its all-gather and
reduce-scatter inside forward and backward, so ``policy_entry_barrier``, ``policy_entropy_allreduce``,
``policy_metric_allreduce`` and ``policy_final_barrier`` measure only the *explicit* collectives; the
ones that scale with the model are inside ``policy_backward`` and are counted as compute there. Torch
already records every collective in a ring buffer with its shape, dtype and process group, and --
under ``TORCH_NCCL_ENABLE_TIMING``, which the distributed debug preset sets -- its device duration.
This publishes the buffer's per-step delta as a few labelled counters carrying the same ``rank`` and
``step`` attributes as the spans, so the decomposition is queryable beside them.

``duration_ms`` is the NCCL kernel's device time, so it bounds exposed collective time from above
rather than measuring it: a collective that waits behind compute on its own stream still accrues
duration. Element counts carry no such caveat, and they are the baseline that expert parallelism and
grouped matmul move.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, astuple, dataclass, field
from typing import Any

import torch
from loguru import logger

from skyrl_train.telemetry import WORKER_ROLE, telemetry

LOG_PREFIX = "FLIGHT_RECORDER_SUMMARY "
UNKNOWN = "unknown"
# torch writes a process group as (name, desc) and prints "undefined" for a group nobody named;
# torch/distributed/flight_recorder/components/types.py joins them the same way, so a label here
# matches what `fr_trace` prints for the same run.
UNDEFINED_GROUP_DESCRIPTION = "undefined"

collectives = telemetry.counter("nccl_collectives", unit="{collective}")
collective_elements = telemetry.counter("nccl_collective_elements", unit="{element}")
collective_seconds = telemetry.counter("nccl_collective_seconds", unit="s")


@dataclass(frozen=True)
class CollectiveBucket:
    process_group: str
    collective: str
    dtype: str
    state: str


@dataclass
class CollectiveTotals:
    count: int = 0
    input_elements: int = 0
    duration_seconds: float = 0.0
    timed_count: int = 0


@dataclass
class FlightRecorderDelta:
    """Everything the recorder holds that the previous capture had not already reported."""

    covers_global_step: int | None = None
    first_record_id: int | None = None
    last_record_id: int | None = None
    dropped_records: int = 0
    totals: dict[CollectiveBucket, CollectiveTotals] = field(default_factory=dict)


@dataclass
class _CaptureState:
    global_step: int | None = None
    last_record_id: int = -1


_state = _CaptureState()


# torch exposes TWO flight recorders and they are NOT the same object. Proven on an H100
# (job atqamar-fr-probe): after 8 NCCL all_reduces, `_dump_nccl_trace_json` held 8 entries and
# `_dump_fr_trace_json` held 0.
#   _dump_nccl_trace_json  ProcessGroupNCCL's own FlightRecorder<at::cuda::CUDAEvent>
#                          (ProcessGroupNCCL.hpp:1520) -- where NCCL collectives actually go.
#   _dump_fr_trace_json    the generic FlightRecorder<c10::Event> (FlightRecorder.hpp:329) --
#                          gloo and CPU. torch's own debug server lists them as two views,
#                          "FlightRecorder NCCL" vs "FlightRecorder CPU".
# ⚠️ The NCCL symbol is absent on a CPU wheel (USE_C10D_NCCL not compiled), which is exactly why
# probing for it on a laptop "proved" the generic one was correct. Twice. A CUDA-only symbol
# cannot be checked on a CPU build.
_DUMP_PREFERENCE = ("_dump_nccl_trace_json", "_dump_fr_trace_json")
_dump_binding: tuple[str, Any] | None = None
_dump_logged = False


def _resolve_dump() -> tuple[str, Any] | None:
    global _dump_binding
    if _dump_binding is None:
        for name in _DUMP_PREFERENCE:
            fn = getattr(torch._C._distributed_c10d, name, None)
            if fn is not None:
                _dump_binding = (name, fn)
                break
        else:
            logger.error(
                f"{LOG_PREFIX}no dump symbol on this torch: tried {', '.join(_DUMP_PREFERENCE)}. "
                "NCCL collective telemetry is unavailable."
            )
            _dump_binding = ("", None)
    return _dump_binding if _dump_binding[1] is not None else None


def _dump_trace() -> Mapping[str, Any] | None:
    """Read the recorder, or None when this torch or this backend has nothing to read."""
    global _dump_logged
    bound = _resolve_dump()
    if bound is None:
        return None
    name, dump = bound
    try:
        payload = json.loads(dump(True, False))
    except Exception:  # noqa: BLE001 - a diagnostic must never fail the step it measures
        logger.opt(exception=True).warning(f"Could not read the NCCL flight recorder via {name}")
        return None
    if not _dump_logged:
        _dump_logged = True
        n = len(payload.get("entries") or ())
        # Emptiness here is the ONLY symptom this bug ever had: no rows, no error, for four steps
        # and a full cluster run. Say it out loud, once, rather than returning quietly.
        detail = "" if n else f" -- recorder is EMPTY; payload keys {sorted(payload)}"
        (logger.info if n else logger.warning)(
            f"{LOG_PREFIX}bound {name}, first read returned {n} entries{detail}"
        )
    return payload


def _element_count(shapes: Sequence[Sequence[int]] | None) -> int:
    return sum(math.prod(int(extent) for extent in shape) for shape in shapes or ())


def _bucket(entry: Mapping[str, Any]) -> CollectiveBucket:
    group = entry.get("process_group") or ()
    name = str(group[0]) if len(group) > 0 else UNKNOWN
    description = str(group[1]) if len(group) > 1 else UNDEFINED_GROUP_DESCRIPTION
    dtypes = sorted({str(dtype) for dtype in entry.get("input_dtypes") or ()})
    return CollectiveBucket(
        process_group=name if description == UNDEFINED_GROUP_DESCRIPTION else f"{name}:{description}",
        collective=str(entry.get("profiling_name") or UNKNOWN),
        dtype=",".join(dtypes) or UNKNOWN,
        state=str(entry.get("state") or UNKNOWN),
    )


def summarize(
    entries: Iterable[Mapping[str, Any]],
    *,
    after_record_id: int,
    covers_global_step: int | None,
) -> FlightRecorderDelta:
    """Fold the entries newer than ``after_record_id`` into one bucket per group and collective."""
    delta = FlightRecorderDelta(covers_global_step=covers_global_step)
    oldest_held: int | None = None
    for entry in entries:
        record_id = int(entry.get("record_id", -1))
        oldest_held = record_id if oldest_held is None else min(oldest_held, record_id)
        if record_id <= after_record_id:
            continue
        if delta.first_record_id is None or record_id < delta.first_record_id:
            delta.first_record_id = record_id
        if delta.last_record_id is None or record_id > delta.last_record_id:
            delta.last_record_id = record_id
        totals = delta.totals.setdefault(_bucket(entry), CollectiveTotals())
        totals.count += 1
        totals.input_elements += _element_count(entry.get("input_sizes"))
        duration_ms = entry.get("duration_ms")
        if duration_ms is not None:
            totals.duration_seconds += float(duration_ms) / 1000.0
            totals.timed_count += 1
    # The buffer is a ring, so anything between the cursor and its oldest surviving record was
    # overwritten before this capture read it. Reported rather than inferred from a suspicious total.
    if after_record_id >= 0 and oldest_held is not None:
        delta.dropped_records = max(0, oldest_held - after_record_id - 1)
    return delta


def publish(delta: FlightRecorderDelta, *, rank: int) -> None:
    step = UNKNOWN if delta.covers_global_step is None else str(delta.covers_global_step)
    ordered = sorted(delta.totals.items(), key=lambda item: astuple(item[0]))
    buckets = [{**asdict(bucket), **asdict(totals)} for bucket, totals in ordered]
    logger.info(
        LOG_PREFIX
        + json.dumps(
            {
                "rank": rank,
                "covers_global_step": delta.covers_global_step,
                "first_record_id": delta.first_record_id,
                "last_record_id": delta.last_record_id,
                "dropped_records": delta.dropped_records,
                "buckets": buckets,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    for bucket, totals in delta.totals.items():
        attributes = {
            "role": WORKER_ROLE,
            "rank": str(rank),
            "step": step,
            "process_group": bucket.process_group,
            "collective": bucket.collective,
            "dtype": bucket.dtype,
            "collective_state": bucket.state,
        }
        collectives.add(totals.count, attributes=attributes)
        collective_elements.add(totals.input_elements, attributes=attributes)
        if totals.timed_count:
            collective_seconds.add(totals.duration_seconds, attributes=attributes)


def capture_at_step_boundary(rank: int, global_step: int | None) -> None:
    """Publish the collectives recorded since the previous global step, once per step.

    Called on entry to every training-step region, which is per micro-step: at E6's geometry that is
    64 regions per step per rank, and one recorder read each would cost more than it explains.
    """
    if global_step == _state.global_step:
        return
    covered = _state.global_step
    _state.global_step = global_step
    trace = _dump_trace()
    if trace is None:
        return
    delta = summarize(
        trace.get("entries") or (),
        after_record_id=_state.last_record_id,
        covers_global_step=covered,
    )
    if delta.last_record_id is not None:
        _state.last_record_id = delta.last_record_id
    # A process with no NCCL communicator holds an empty recorder for the whole run, and a summary
    # of nothing would then be published at every step boundary for as long as it lives.
    if delta.totals or delta.dropped_records:
        publish(delta, rank=rank)
