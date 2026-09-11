"""Synthetic export-decomposition receipts: acceptance, rejection of confounds, and the cost fold."""

import copy

import pytest
from skyrl_train.weight_sync.export_decomposition_audit import (
    ARM_PARALLELISM,
    ARMS,
    CUDA_SUBSTAGES,
    LAYERS,
    PASSES,
    REQUIRED_ENVIRONMENT,
    fold_costs,
    layer_of,
    validate_arm_rows,
    validate_receipt_set,
)

EXPORT = [[0.0, 10.0]]
FULL_INVENTORY = [
    ["model.embed_tokens.weight", [4, 8], "bfloat16"],
    ["model.layers.0.mlp.experts.down_proj.weight", [2, 3, 4], "bfloat16"],
    ["model.layers.1.mlp.router.bias", [2], "float32"],
]
NONEXPERT_INVENTORY = [entry for entry in FULL_INVENTORY if ".mlp.experts." not in entry[0]]


def _detail(intervals, layers, byte=8):
    return {
        "host_intervals": [[start, end] for start, end in intervals],
        "layers": list(layers),
        "bytes": [byte] * len(intervals),
        "calls": len(intervals),
    }


def export_pass(arm, pass_index):
    _pp, ep = ARM_PARALLELISM[arm]
    nonexpert = arm.endswith("-nonexpert")
    inventory = NONEXPERT_INVENTORY if nonexpert else FULL_INVENTORY
    cuda = {
        "task_build": _detail([[0.0, 0.5]], [None]),
        "pp_broadcast": _detail([[1.0, 1.2], [2.0, 2.2], [3.0, 3.2], [4.0, 4.2]], [None, 0, 1, None]),
        "pp_broadcast_obj": _detail([[1.2, 1.3], [2.2, 2.3]], [0, 1], byte=0),
        "ep_gather": _detail([[5.0, 5.5], [6.0, 6.5]], [0, 1]) if ep > 1 else _detail([], []),
        "expert_stack": _detail([], []) if nonexpert else _detail([[7.0, 7.5], [8.0, 8.5]], [0, 1]),
        "wire_cast": _detail([[8.6, 8.7], [8.8, 8.9], [9.0, 9.1]], [None, 0, 1], byte=0),
        "next_source": _detail([[0.6, 0.9], [1.5, 4.5], [5.5, 9.5]][: len(inventory)], [None, 0, 1][: len(inventory)]),
    }
    intervals = {"export": copy.deepcopy(EXPORT), "pack": [[10.0, 11.0]]}
    intervals.update(
        {stage: copy.deepcopy(detail["host_intervals"]) for stage, detail in cuda.items() if detail["calls"]}
    )
    return {
        "pass_index": pass_index,
        "warmup": pass_index == 0,
        "timing": {"events_complete": True, "intervals": intervals},
        "substages": {**cuda, "source_release_wait": _detail([[0.95, 0.97], [4.6, 4.7]], [])},
        "source_count": len(inventory),
        "peak_extra_bytes": 123,
        "operation_complete": True,
    }


def arm_rows(arm, pid_base=0):
    pp, ep = ARM_PARALLELISM[arm]
    inventory = NONEXPERT_INVENTORY if arm.endswith("-nonexpert") else FULL_INVENTORY
    rows = []
    for rank in (0, 1):
        rows.append(
            {
                "arm": arm,
                "layers": LAYERS,
                "pp": pp,
                "ep": ep,
                "rank": rank,
                "identity": {
                    "host": "host",
                    "pid": pid_base + rank,
                    "gpu_uuid": f"gpu-{rank}",
                    "task_id": "task:0",
                    "attempt_uid": "original",
                },
                "environment": {**REQUIRED_ENVIRONMENT, "CUDA_DEVICE_MAX_CONNECTIONS": "8"},
                "manifest_id": f"manifest-{arm}",
                "expected_inventory": copy.deepcopy(inventory),
                "yielded_inventory": copy.deepcopy(inventory[::-1]),  # Order is the sender's business, not the audit's.
                "tensor_inventory_exact": True,
                "passes": [export_pass(arm, index) for index in range(PASSES)],
            }
        )
    return rows


def receipt_cells():
    return [{"source_commit": "frozen", "rows": arm_rows(arm, pid_base=10 * index)} for index, arm in enumerate(ARMS)]


def test_layer_attribution_from_hf_names():
    assert layer_of("model.layers.7.mlp.experts.down_proj.weight") == 7
    assert layer_of("model.embed_tokens.weight") is None
    assert layer_of(None) is None


def test_accepts_complete_synthetic_arms():
    for arm in ARMS:
        validate_arm_rows(arm_rows(arm), arm)
    assert set(validate_receipt_set(receipt_cells(), "frozen")) == set(ARMS)


@pytest.mark.parametrize(
    "failure",
    [
        "missing_arm",
        "duplicate_arm",
        "substage_outside_export",
        "incomplete_events",
        "reused_pid",
        "inventory_mismatch",
        "nonfinite_clock",
        "expert_work_in_remainder_arm",
        "warmup_not_first",
        "call_count_mismatch",
        "connections",
        "changed_attempt",
        "changed_source",
    ],
)
def test_rejects_corrupt_receipt_set(failure):
    cells = receipt_cells()
    timed = cells[0]["rows"][0]["passes"][1]
    if failure == "missing_arm":
        cells.pop()
    elif failure == "duplicate_arm":
        cells[2] = copy.deepcopy(cells[0])
    elif failure == "substage_outside_export":
        timed["timing"]["intervals"]["pp_broadcast"][0] = [9.9, 10.5]
    elif failure == "incomplete_events":
        timed["timing"]["events_complete"] = False
    elif failure == "reused_pid":
        for row in cells[1]["rows"]:
            row["identity"]["pid"] = row["rank"]
    elif failure == "inventory_mismatch":
        cells[0]["rows"][1]["yielded_inventory"][0][1] = [4, 9]
    elif failure == "nonfinite_clock":
        timed["substages"]["source_release_wait"]["host_intervals"][0][1] = float("nan")
    elif failure == "expert_work_in_remainder_arm":
        remainder = cells[2]["rows"][0]["passes"][2]
        remainder["substages"]["expert_stack"] = _detail([[7.0, 7.5]], [0])
        remainder["timing"]["intervals"]["expert_stack"] = [[7.0, 7.5]]
    elif failure == "warmup_not_first":
        passes = cells[1]["rows"][0]["passes"]
        passes[0]["warmup"], passes[1]["warmup"] = False, True
    elif failure == "call_count_mismatch":
        timed["substages"]["wire_cast"]["calls"] = 2
    elif failure == "connections":
        cells[1]["rows"][1]["environment"]["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    elif failure == "changed_attempt":
        for row in cells[1]["rows"]:
            row["identity"]["attempt_uid"] = "replacement"
    else:
        cells[1]["source_commit"] = "different"
    with pytest.raises((AssertionError, ValueError)):
        validate_receipt_set(cells, "frozen")


def test_accepts_yield_order_that_differs_from_the_state_dict():
    rows = arm_rows("pp2ep1")
    assert rows[0]["yielded_inventory"] != rows[0]["expected_inventory"]
    validate_arm_rows(rows, "pp2ep1")


def test_fold_separates_fixed_from_per_layer_cost_and_skips_warmup():
    cells = receipt_cells()
    # A pathological warm-up pass must not leak into the fold.
    cells[0]["rows"][0]["passes"][0]["timing"]["intervals"]["export"] = [[0.0, 1000.0]]
    summaries = {(row["arm"], row["rank"]): row for row in fold_costs(cells)}
    assert set(summaries) == {(arm, rank) for arm in ARMS for rank in (0, 1)}
    first = summaries[("pp1ep2", 0)]
    assert first["timed_passes"] == PASSES - 1
    assert first["export_union_seconds_per_pass"] == pytest.approx(10.0)
    assert first["substage_union_seconds_per_pass"] == pytest.approx(8.6)
    assert first["unattributed_export_seconds_per_pass"] == pytest.approx(1.4)
    assert first["source_release_wait_seconds_per_pass"] == pytest.approx(0.12)
    assert first["peak_extra_bytes"] == 123
    broadcast = first["stages"]["pp_broadcast"]
    assert broadcast["calls_per_pass"] == 4
    assert broadcast["cuda_seconds_per_pass"] == pytest.approx(0.8)
    assert broadcast["fixed_seconds_per_pass"] == pytest.approx(0.4)
    assert broadcast["per_layer_seconds_per_pass"] == pytest.approx(0.2)
    assert broadcast["layer_seconds_per_pass"] == pytest.approx([0.2, 0.2])
    assert set(first["stages"]) == set(CUDA_SUBSTAGES)
    remainder = summaries[("pp2ep1-nonexpert", 1)]
    assert remainder["stages"]["expert_stack"]["calls_per_pass"] == 0
    assert remainder["stages"]["ep_gather"]["cuda_seconds_per_pass"] == 0
