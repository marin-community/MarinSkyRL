from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from skyrl_train.weight_sync.frozen_source_plan import frozen_source_plan
from skyrl_train.weight_sync.frozen_source_views import local_source_slices, source_view
from skyrl_train.weight_sync.manifest import TensorSpec, build_manifest, pack_bucket
from tests.cpu.weight_sync.test_frozen_source_views import task, config


def plan_fixture():
    manifest = build_manifest(
        [TensorSpec("expert", (4, 2, 2), "bfloat16", True), TensorSpec("bias", (2,), "float32")], 24
    )
    full = {
        "expert": torch.arange(16, dtype=torch.bfloat16).reshape(4, 2, 2),
        "bias": torch.tensor([-0.0, float("nan")]),
    }
    rows, all_sources = [], []
    for rank in range(2):
        tasks = [
            task("GrugStackedExpertMapping", f"experts.linear_fc2.weight{expert}", "expert", full["expert"][expert])
            for expert in range(rank, 4, 2)
        ]
        tasks.append(task("ReplicatedMapping", "router.expert_bias", "bias", full["bias"]))
        pieces, sources = local_source_slices(tasks, config())
        rows.append(
            {
                "rank": rank,
                "dense_owner": rank == 0,
                "expert_owner": True,
                "slices": [asdict(piece) for piece in pieces],
            }
        )
        all_sources.append(sources)
    return SimpleNamespace(manifest=manifest, full=full, rows=rows, sources=all_sources)


def test_two_policy_owners_cover_every_wire_byte_once():
    case = plan_fixture()
    plan = frozen_source_plan(case.manifest, case.rows)
    assert {part.owner_rank for bucket in plan for part in bucket} == {0, 1}
    for bucket, parts in enumerate(plan):
        actual = torch.empty(24, dtype=torch.uint8)
        expected = torch.empty(24, dtype=torch.uint8)
        count = pack_bucket(case.manifest, bucket, case.full, expected)
        for part in parts:
            value = source_view(part.source, case.sources[part.owner_rank]).view(torch.uint8)
            actual[part.bucket_offset : part.bucket_offset + value.numel()].copy_(value)
        assert torch.equal(actual[:count], expected[:count])


@pytest.mark.parametrize("failure", ["rank", "owner", "missing", "overlap", "dtype"])
def test_ambiguous_or_incomplete_source_ownership_rejects(failure):
    case = plan_fixture()
    if failure == "rank":
        case.rows.pop()
    elif failure == "owner":
        case.rows[1]["dense_owner"] = True
    elif failure == "missing":
        case.rows[0]["slices"].pop(0)
    elif failure == "overlap":
        case.rows[0]["slices"][1]["hf_offset"] = 0
    else:
        case.rows[0]["slices"][0]["wire_dtype"] = "float32"
    with pytest.raises(ValueError):
        frozen_source_plan(case.manifest, case.rows)
