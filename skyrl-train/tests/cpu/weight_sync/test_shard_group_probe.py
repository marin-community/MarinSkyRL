"""Actual Ray and Gloo coverage for the expert-block native fixture."""

import hashlib

import pytest
import ray
from ray.cluster_utils import Cluster

from skyrl_train.weight_sync.shard_group_probe import port_counters, run_group_probe, tiny_schedule, validate_role_hosts
from skyrl_train.entrypoints.probe_shard_groups import persist, require_unmeasured


@pytest.fixture
def native_ray():
    ray.init(num_cpus=8, include_dashboard=False)
    yield
    ray.shutdown()


@pytest.mark.parametrize("receivers,ep", [(2, 1), (4, 1), (2, 2)])
def test_native_groups_deliver_every_byte_from_alternating_pp_roots(tmp_path, native_ray, receivers, ep):
    schedule = tiny_schedule(receivers, 257, ep=ep)
    markers = []

    def measured(state):
        markers.append((len(state["ready"]), len(state["groups"]), len(state["broadcasts"])))

    result = run_group_probe(schedule, "gloo", str(tmp_path), measured)
    assert result["error"] is None, result["error"]
    assert "cleanup_error" not in result, result.get("cleanup_error")
    count = (2 + receivers) * ep
    assert markers == [(count, count, 0)]
    assert len(result["ready"]) == len(result["groups"]) == len(result["cleanup"]) == count
    assert len({row["pid"] for row in result["ready"]}) == count
    assert result["rendezvous"]["phase"] == "listening"
    assert result["rendezvous_cleanup"]["phase"] == "closed"
    assert result["rendezvous_cleanup"]["state_was_present"]
    assert result["rendezvous_cleanup"]["closed_monotonic"] >= max(row["monotonic"] for row in result["cleanup"])
    assert len({row["endpoint"]["store_namespace"] for row in result["groups"]}) == ep
    assert all(row["endpoint"]["store_namespace"].startswith("shard/tiny-") for row in result["groups"])
    assert len({row["endpoint"]["init_method"] for row in result["groups"]}) == 1
    assert all(row["workspace_bytes"] == 257 for row in result["ready"])
    for row in result["groups"]:
        assert row["readiness"]["phase"] == "groups-ready"
        assert row["readiness"]["new_explicit_tensor_storage_bytes"] == 0
        assert row["readiness"]["groups"][0]["members"] == row["members"]
        assert row["readiness"]["groups"][0]["payload_bytes"] == 4
    for index, item in enumerate(schedule.broadcasts):
        rows = [row for row in result["broadcasts"] if row["index"] == index]
        assert {row["rank"] for row in rows} == set(schedule.groups[item.group_ep].members)
        assert len(rows) == 2 + receivers
        expected = hashlib.sha256(bytes([(index * 17 + 3) % 251]) * 257).hexdigest()
        assert {row["sha256"] for row in rows} == {expected}
        assert {row["received_bytes"] for row in rows} == {257}
    assert {row["stage"] for row in result["cleanup"]} == {"closed"}
    # Both PP owners really became roots; no trainer or receiver was skipped.
    assert {row["entry"]["root"] for row in result["broadcasts"]} == set(range(2 * ep))


def test_port_readback_retains_raw_units_and_missing_counter_error(tmp_path):
    directory = tmp_path / "mlx5_0/ports/1/counters"
    directory.mkdir(parents=True)
    (directory / "port_xmit_data").write_text("19\n")
    (directory / "port_rcv_data").write_text("37\n")
    result = port_counters(tmp_path)
    assert result["ports"] == [
        {"device": "mlx5_0", "port": "1", "port_xmit_data": 19, "port_rcv_data": 37, "counter_unit_bytes": 4}
    ]
    (directory / "port_rcv_data").unlink()
    assert "FileNotFoundError" in port_counters(tmp_path)["ports"][0]["error"]


@pytest.mark.parametrize("fault", ["same_host", "same_ray_node", "missing", "split_role"])
def test_role_placement_rejects_incomplete_or_shared_physical_host(fault):
    rows = [
        {"physical_node": "sender-host" if rank < 2 else "receiver-host", "ray_node_id": "a" if rank < 2 else "b"}
        for rank in range(6)
    ]
    validate_role_hosts(rows, 2)
    if fault == "same_host":
        for row in rows:
            row["physical_node"] = "one-host"
    elif fault == "same_ray_node":
        for row in rows:
            row["ray_node_id"] = "one-node"
    elif fault == "missing":
        rows[0]["physical_node"] = None
    else:
        rows[3]["physical_node"] = "third-host"
    with pytest.raises(ValueError):
        validate_role_hosts(rows, 2)


def test_two_actual_ray_nodes_on_one_physical_host_fail_before_group_init(tmp_path, monkeypatch):
    monkeypatch.setenv("IRIS_NODE_NAME", "one-physical-cpu-host")
    cluster = Cluster()
    try:
        for _ in range(2):
            cluster.add_node(num_cpus=4, include_dashboard=False, object_store_memory=80 * 1024**2)
        ray.init(address=cluster.address)
        result = run_group_probe(tiny_schedule(4, 257), "gloo", str(tmp_path), two_hosts=True)
    finally:
        ray.shutdown()
        cluster.shutdown()
    assert "distinct physical hosts" in result["error"]
    assert result["groups"] == result["broadcasts"] == []
    assert len(result["cleanup"]) == 6
    assert len({row["ray_node_id"] for row in result["ready"][:2]}) == 1
    assert len({row["ray_node_id"] for row in result["ready"][2:]}) == 1
    assert len({row["ray_node_id"] for row in result["ready"]}) == 2


def test_prior_native_attempt_marker_prevents_remeasurement(tmp_path):
    persist(str(tmp_path / "attempts/startup-only/entered.json"), {"attempt": "startup-only"})
    require_unmeasured(str(tmp_path))
    receipt = persist(str(tmp_path / "attempts/measured/measurement-started.json"), {"attempt": "measured"})
    assert receipt["bytes"] == len((tmp_path / "attempts/measured/measurement-started.json").read_bytes())
    with pytest.raises(ValueError, match="measured attempt"):
        require_unmeasured(str(tmp_path))


def test_measurement_marker_lookup_fails_closed_on_storage_error(tmp_path, monkeypatch):
    def unavailable(path):
        raise PermissionError("storage unavailable")

    monkeypatch.setattr("skyrl_train.entrypoints.probe_shard_groups.find_files", unavailable)
    with pytest.raises(PermissionError):
        require_unmeasured(str(tmp_path))


@pytest.mark.parametrize("ready", [True, False])
def test_native_entrypoint_persists_warmup_failure_before_interpretation(tmp_path, monkeypatch, capsys, ready):
    import json
    import sys
    from skyrl_train.entrypoints import probe_shard_groups as entry

    monkeypatch.setenv("IRIS_ATTEMPT_UID", "0123456789abcdef")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe",
            "--durable-prefix",
            str(tmp_path / "durable"),
            "--output",
            str(tmp_path / "local"),
            "--source-commit",
            "a" * 40,
        ],
    )
    monkeypatch.setattr(entry.ray, "init", lambda **kwargs: None)
    monkeypatch.setattr(entry.ray, "shutdown", lambda: None)
    calls = []

    def run(schedule, backend, output, marker, *, two_hosts):
        assert backend == "nccl" and two_hosts
        calls.append(len(schedule.receiver_global_ranks))
        result = {
            "ready": [],
            "groups": [{"readiness": {"phase": "groups-ready" if ready else "missing"}}],
            "error": None,
        }
        marker(result)
        return result

    monkeypatch.setattr(entry, "run_group_probe", run)
    if ready:
        entry.main()
        assert calls == [2, 4]
        assert "K10_TINY_NATIVE_GROUP_PASS senders=2 receivers=2,4 hosts=2 warmup=true" in capsys.readouterr().out
    else:
        with pytest.raises(RuntimeError, match="omitted acknowledged"):
            entry.main()
        assert calls == [2]
        result = json.loads((tmp_path / "durable/attempts/0123456789abcdef/receivers-2.json").read_bytes())
        assert "omitted acknowledged" in result["error"]
        assert not (tmp_path / "durable/attempts/0123456789abcdef/complete.json").exists()
