"""Actual Ray and Gloo coverage for the expert-block native fixture."""

import hashlib

import pytest
import ray

from skyrl_train.weight_sync.shard_group_probe import port_counters, run_group_probe, tiny_schedule


@pytest.fixture(scope="module")
def native_ray():
    ray.init(num_cpus=8, include_dashboard=False)
    yield
    ray.shutdown()


@pytest.mark.parametrize("receivers,ep", [(2, 1), (4, 1), (2, 2)])
def test_native_groups_deliver_every_byte_from_alternating_pp_roots(tmp_path, native_ray, receivers, ep):
    schedule = tiny_schedule(receivers, 257, ep=ep)
    result = run_group_probe(schedule, "gloo", str(tmp_path))
    assert result["error"] is None, result["error"]
    assert "cleanup_error" not in result, result.get("cleanup_error")
    count = (2 + receivers) * ep
    assert len(result["ready"]) == len(result["groups"]) == len(result["cleanup"]) == count
    assert len({row["pid"] for row in result["ready"]}) == count
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
