"""Actual Ray/default-world/custom-Gloo regression for the transport fixture."""

import copy

import pytest
import ray
from ray.cluster_utils import Cluster

from skyrl_train.entrypoints.probe_weight_sync_ray_transport import (
    ENV_KEYS,
    attempt_receipt_prefix,
    run_probe,
    validate_worlds,
)


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


def test_native_attempt_receipts_cannot_overwrite_another_attempt():
    assert attempt_receipt_prefix("s3://bucket/probe", "attempt1") != attempt_receipt_prefix(
        "s3://bucket/probe", "attempt2"
    )
    for uid in ("", "../escape", "a/b"):
        with pytest.raises(ValueError):
            attempt_receipt_prefix("s3://bucket/probe", uid)
    assert "NCCL_NTHREADS" in ENV_KEYS and "NCCL_P2P_NET_DISABLE" in ENV_KEYS


@pytest.mark.parametrize("mutation", ["physical_host", "ray_node", "missing_host"])
def test_transport_rejects_two_pods_without_distinct_physical_host_evidence(mutation):
    rows = [
        {
            "custom_rank": rank,
            "default_rank": 0,
            "default_world": 1,
            "custom_world": 2,
            "environment": dict.fromkeys(ENV_KEYS),
            "physical_node": f"physical-{rank}",
            "ray_node_id": f"ray-{rank}",
        }
        for rank in (0, 1)
    ]
    validate_worlds(rows, require_distinct_hosts=True)
    if mutation == "physical_host":
        rows[1]["physical_node"] = rows[0]["physical_node"]
    elif mutation == "ray_node":
        rows[1]["ray_node_id"] = rows[0]["ray_node_id"]
    else:
        rows[1]["physical_node"] = None
    with pytest.raises(ValueError):
        validate_worlds(rows, require_distinct_hosts=True)


def test_actual_two_ray_nodes_on_one_host_fail_before_custom_group_and_measurements(tmp_path, monkeypatch):
    monkeypatch.setenv("IRIS_NODE_NAME", "actual-cpu-test-host")
    cluster = Cluster()
    try:
        for _ in range(2):
            cluster.add_node(num_cpus=2, include_dashboard=False, object_store_memory=80 * 1024**2)
        ray.init(address=cluster.address)
        result = run_probe(str(tmp_path), "gloo", (257,), require_distinct_hosts=True)
    finally:
        ray.shutdown()
        cluster.shutdown()
    assert "two distinct physical hosts" in result["error"]
    assert len({row["ray_node_id"] for row in result["worlds"]}) == 2
    assert len({row["physical_node"] for row in result["worlds"]}) == 1
    assert result["initialization"] == result["payloads"] == []
    assert len(result["rank_events"]) == 2
    assert all(rows[-1]["stage"] == "pre_custom_group" for rows in result["rank_events"].values())
