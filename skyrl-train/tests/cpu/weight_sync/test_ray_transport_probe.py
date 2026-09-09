"""Actual Ray/default-world/custom-Gloo regression for the transport fixture."""

import copy

import pytest
import ray

from skyrl_train.entrypoints.probe_weight_sync_ray_transport import ENV_KEYS, run_probe, validate_worlds


def test_native_ray_custom_group_and_all_payload_bytes(tmp_path):
    ray.init(num_cpus=4, include_dashboard=False)
    try:
        result = run_probe(str(tmp_path), "gloo", (257, 8193), repeats=2)
    finally:
        ray.shutdown()
    assert result["error"] is None, result["error"]
    assert len(result["payloads"]) == 8
    assert len(result["initialization"]) == len(result["cleanup"]) == 2
    assert {row["default_world"] for row in result["worlds"]} == {1}
    assert {row["custom_world"] for row in result["worlds"]} == {2}
    assert len({row["pid"] for row in result["worlds"]}) == 2
    for rank in (0, 1):
        assert (tmp_path / f"rank-{rank}.json").is_file()
        assert result["rank_events"][f"rank-{rank}.json"][0]["stage"] == "before_default_group"


@pytest.mark.parametrize("mutation", ["rank", "default_world", "custom_world", "environment", "invariance"])
def test_pre_group_evidence_rejects_wrong_world_or_environment(mutation):
    rows = [
        {
            "custom_rank": rank,
            "default_rank": 0,
            "default_world": 1,
            "custom_world": 2,
            "environment": dict.fromkeys(ENV_KEYS),
        }
        for rank in (0, 1)
    ]
    validate_worlds(rows)
    rows = copy.deepcopy(rows)
    if mutation == "rank":
        rows[1]["custom_rank"] = 0
    elif mutation == "environment":
        rows[1]["environment"]["NCCL_PROTO"] = "Simple"
    elif mutation == "invariance":
        for row in rows:
            row["environment"]["VLLM_BATCH_INVARIANT"] = "1"
    else:
        rows[1][mutation] = 3
    with pytest.raises(ValueError):
        validate_worlds(rows)
