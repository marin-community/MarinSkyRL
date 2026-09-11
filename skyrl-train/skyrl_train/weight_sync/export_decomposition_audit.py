"""Accept or reject export-decomposition receipts; fold sub-stage costs into fixed and per-layer parts.

Device-independent: the GPU entrypoint records rows, this module judges them. It never
imports Megatron, so the CPU suite runs where the export path itself cannot.
"""

import math
import re

from skyrl_train.weight_sync.pipeline_timing import merged_intervals, overlap_seconds

ARMS = ("pp1ep2", "pp2ep1", "pp2ep1-nonexpert")
ARM_PARALLELISM = {"pp1ep2": (1, 2), "pp2ep1": (2, 1), "pp2ep1-nonexpert": (2, 1)}
LAYERS = 2
CONNECTIONS = 8
PASSES = 3
REQUIRED_ENVIRONMENT = {
    "NCCL_CUMEM_ENABLE": "0",
    "VLLM_BATCH_INVARIANT": "0",
    "NCCL_DEBUG": "INFO",
    "NCCL_DEBUG_SUBSYS": "INIT,NET",
}
# CUDA-bracketed sub-stages that run inside the sender's ``export`` bracket.
CONTAINED_SUBSTAGES = (
    "task_build",
    "pp_broadcast",
    "pp_broadcast_obj",
    "ep_gather",
    "expert_stack",
    "wire_cast",
)
# ``_next_source`` wraps the export bracket: its host source-release wait precedes the bracket,
# so each call encloses one export span rather than sitting inside it.
ENCLOSING_SUBSTAGES = ("next_source",)
CUDA_SUBSTAGES = (*CONTAINED_SUBSTAGES, *ENCLOSING_SUBSTAGES)
# Host-only wait that precedes the export bracket and is otherwise only a gap.
HOST_SUBSTAGES = ("source_release_wait",)
EXPERT_NAME_FRAGMENT = ".mlp.experts."
PASS_LINE = (
    "K11_EXPORT_DECOMPOSITION_PASS arms=pp1ep2,pp2ep1,pp2ep1-nonexpert layers=2 "
    "tensor_inventory_exact=true events_complete=true stage_times_are_observations=true"
)
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_CONTAINMENT_TOLERANCE = 1e-6


def layer_of(hf_name):
    """Layer index carried by an HF parameter name, or None for shared tensors."""
    match = _LAYER.search(hf_name or "")
    return int(match.group(1)) if match else None


def inventory_key(entry):
    name, shape, dtype = entry
    return (name, tuple(int(d) for d in shape), dtype)


def _finite_ordered(intervals):
    for start, end in intervals:
        if not (math.isfinite(start) and math.isfinite(end)) or start < 0 or end < start:
            raise ValueError("Sub-stage interval must be finite, ordered and nonnegative")
    return intervals


def _union_seconds(intervals):
    return sum(end - start for start, end in merged_intervals(intervals))


def validate_pass(record, arm, layers):
    """One timed or warm-up export pass of one rank."""
    timing = record["timing"]
    assert timing["events_complete"] and record["operation_complete"]
    intervals = timing["intervals"]
    assert intervals.get("export"), "Every pass must carry the sender export bracket"
    for values in intervals.values():
        merged_intervals(values)  # Rejects malformed or nonfinite CUDA clocks.
    export = intervals["export"]
    substages = record["substages"]
    assert set(substages) == set(CUDA_SUBSTAGES) | set(HOST_SUBSTAGES), "Sub-stage set must be complete"
    for stage in CUDA_SUBSTAGES:
        detail = substages[stage]
        cuda = intervals.get(stage, [])
        assert detail["calls"] == len(cuda) == len(detail["host_intervals"]) == len(detail["layers"])
        assert len(detail["bytes"]) == detail["calls"]
        _finite_ordered(detail["host_intervals"])
        assert all(layer is None or 0 <= layer < layers for layer in detail["layers"])
        if stage in ENCLOSING_SUBSTAGES:
            assert detail["calls"] == len(export), f"{stage} must wrap every export span once"
            enclosed = overlap_seconds(export, cuda)
            assert enclosed >= _union_seconds(export) - _CONTAINMENT_TOLERANCE, f"export escapes {stage}"
        else:
            inside = overlap_seconds(cuda, export)
            assert inside >= _union_seconds(cuda) - _CONTAINMENT_TOLERANCE, f"{stage} escapes the export bracket"
    for stage in HOST_SUBSTAGES:
        detail = substages[stage]
        assert detail["calls"] == len(detail["host_intervals"])
        _finite_ordered(detail["host_intervals"])
    assert record["peak_extra_bytes"] >= 0
    expert_calls = substages["expert_stack"]["calls"]
    gather_calls = substages["ep_gather"]["calls"]
    assert substages["pp_broadcast"]["calls"] > 0 and substages["wire_cast"]["calls"] > 0
    assert substages["next_source"]["calls"] > 0 and substages["task_build"]["calls"] == 1
    # The Bridge calls gather_from_ep_ranks and the grouped accumulate for every expert task at any
    # EP width (1,536 calls at EP1 on the v2 run); the width shows in seconds, not in counts.
    if arm.endswith("-nonexpert"):
        assert expert_calls == 0 and gather_calls == 0, "The remainder arm must not touch expert tasks"
    else:
        assert expert_calls > 0 and gather_calls > 0


def validate_arm_rows(rows, arm, layers=LAYERS):
    """Two per-rank rows of one fresh-process arm."""
    assert len(rows) == 2 and {row["rank"] for row in rows} == {0, 1}
    pp, ep = ARM_PARALLELISM[arm]
    assert len({row["manifest_id"] for row in rows}) == 1
    assert len({row["identity"]["gpu_uuid"] for row in rows}) == 2
    assert len({row["identity"]["pid"] for row in rows}) == 2
    for row in rows:
        assert row["arm"] == arm and row["layers"] == layers and (row["pp"], row["ep"]) == (pp, ep)
        assert row["environment"]["CUDA_DEVICE_MAX_CONNECTIONS"] == str(CONNECTIONS)
        assert all(row["environment"].get(key) == value for key, value in REQUIRED_ENVIRONMENT.items())
        expected = [inventory_key(entry) for entry in row["expected_inventory"]]
        yielded = [inventory_key(entry) for entry in row["yielded_inventory"]]
        assert expected and len(set(expected)) == len(expected) and len(set(yielded)) == len(yielded)
        assert set(expected) == set(yielded), "Yielded tensor inventory must equal the HF state dict"
        assert row["tensor_inventory_exact"] is True
        has_experts = any(EXPERT_NAME_FRAGMENT in name for name, _, _ in expected)
        assert has_experts != arm.endswith("-nonexpert")
        passes = row["passes"]
        assert [record["pass_index"] for record in passes] == list(range(PASSES))
        assert [record["warmup"] for record in passes] == [True] + [False] * (PASSES - 1)
        for record in passes:
            validate_pass(record, arm, layers)
            assert record["substages"]["next_source"]["calls"] == len(yielded)
            assert record["source_count"] == len(yielded)


def validate_receipt_set(cells, source_commit, layers=LAYERS):
    """All three arms, each in a fresh process pair on the same two GPUs and attempt."""
    assert len(cells) == len(ARMS)
    seen = {}
    processes = set()
    gpu_sets = set()
    attempts = set()
    for cell in cells:
        assert cell["source_commit"] == source_commit
        rows = cell["rows"]
        arm = rows[0]["arm"]
        assert arm in ARMS and arm not in seen, "Every arm exactly once"
        seen[arm] = rows
        validate_arm_rows(rows, arm, layers)
        for row in rows:
            process = (row["identity"]["host"], row["identity"]["pid"])
            assert process not in processes, "Arms must run in fresh processes"
            processes.add(process)
            attempts.add((row["identity"]["task_id"], row["identity"]["attempt_uid"]))
        gpu_sets.add(tuple(sorted(row["identity"]["gpu_uuid"] for row in rows)))
    assert set(seen) == set(ARMS)
    assert len(gpu_sets) == 1 and len(attempts) == 1
    return seen


def fold_costs(cells, layers=LAYERS):
    """Per arm, rank and sub-stage: calls, bytes, CUDA union seconds split into fixed and per-layer parts.

    Only non-warm-up passes are folded. The decomposed union covers the contained sub-stages
    only; an enclosing stage is reported but never counted against the export bracket. Fixed
    cost is the share attributed to tensors outside any decoder layer plus whatever the contained
    sub-stages leave unattributed inside the export bracket;
    per-layer cost is the mean over decoder layers. Two layers under-amortise fixed cost, so both
    parts are reported rather than a single total.
    """
    summaries = []
    for cell in cells:
        for row in cell["rows"]:
            kept = [record for record in row["passes"] if not record["warmup"]]
            assert kept, "Fold requires at least one timed pass"
            arm, rank = row["arm"], row["rank"]
            export_union = 0.0
            substage_union = 0.0
            release_wait = 0.0
            peak_extra = 0
            per_stage = {}
            for record in kept:
                intervals = record["timing"]["intervals"]
                export_union += _union_seconds(intervals["export"])
                all_cuda = [span for stage in CONTAINED_SUBSTAGES for span in intervals.get(stage, [])]
                substage_union += _union_seconds(all_cuda)
                release_wait += sum(e - s for s, e in record["substages"]["source_release_wait"]["host_intervals"])
                peak_extra = max(peak_extra, record["peak_extra_bytes"])
                for stage in CUDA_SUBSTAGES:
                    detail = record["substages"][stage]
                    entry = per_stage.setdefault(
                        stage,
                        {
                            "calls": 0,
                            "bytes": 0,
                            "cuda_seconds": 0.0,
                            "host_seconds": 0.0,
                            "layer_seconds": [0.0] * layers,
                            "fixed_seconds": 0.0,
                        },
                    )
                    cuda = intervals.get(stage, [])
                    entry["calls"] += detail["calls"]
                    entry["bytes"] += sum(detail["bytes"])
                    entry["cuda_seconds"] += _union_seconds(cuda)
                    entry["host_seconds"] += sum(e - s for s, e in detail["host_intervals"])
                    for (start, end), layer in zip(cuda, detail["layers"]):
                        if layer is None:
                            entry["fixed_seconds"] += end - start
                        else:
                            entry["layer_seconds"][layer] += end - start
            count = len(kept)
            stages = {}
            for stage, entry in per_stage.items():
                stages[stage] = {
                    "share_of_export_union": entry["cuda_seconds"] / export_union if export_union else 0.0,
                    "calls_per_pass": entry["calls"] / count,
                    "bytes_per_pass": entry["bytes"] / count,
                    "cuda_seconds_per_pass": entry["cuda_seconds"] / count,
                    "host_seconds_per_pass": entry["host_seconds"] / count,
                    "fixed_seconds_per_pass": entry["fixed_seconds"] / count,
                    "per_layer_seconds_per_pass": sum(entry["layer_seconds"]) / count / layers,
                    "layer_seconds_per_pass": [value / count for value in entry["layer_seconds"]],
                }
            summaries.append(
                {
                    "arm": arm,
                    "rank": rank,
                    "timed_passes": count,
                    "export_union_seconds_per_pass": export_union / count,
                    "substage_union_seconds_per_pass": substage_union / count,
                    "unattributed_export_seconds_per_pass": max(0.0, (export_union - substage_union) / count),
                    "source_release_wait_seconds_per_pass": release_wait / count,
                    "peak_extra_bytes": peak_extra,
                    "stages": stages,
                    "scope": "two intra-node H100, two layers, random init; op costs and counts, not Snowball's absolute split",
                }
            )
    return summaries
