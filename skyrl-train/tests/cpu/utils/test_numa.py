import pytest

from skyrl_train import numa_policy
from skyrl_train.numa_policy import (
    allowed_memory_nodes,
    host_memory_nodes,
    install_host_memory_policy,
    parse_numactl_hardware,
)
from skyrl_train.utils.numa import physical_gpu_id_for_worker


def test_host_memory_nodes_exclude_memory_only_nodes():
    cpu_topology = {0: tuple(range(72)), 1: tuple(range(72, 144)), 2: tuple(range(144, 216)), 3: tuple(range(216, 288))}

    assert host_memory_nodes(cpu_topology, {0, 1, 2, 3, 4, 12, 20, 28}) == (0, 1, 2, 3)


def test_host_memory_nodes_fail_when_slurm_disallows_lpddr():
    with pytest.raises(RuntimeError, match="no CPU-bearing NUMA nodes are allowed"):
        host_memory_nodes({0: (0, 1), 1: (2, 3)}, {4, 12})


def test_numactl_topology_ignores_memory_only_nodes():
    output = """node 0 cpus: 0 1 2 3
node 1 cpus: 4 5 6 7
node 4 cpus:
node 12 cpus:
"""

    assert parse_numactl_hardware(output) == {0: (0, 1, 2, 3), 1: (4, 5, 6, 7)}


def test_allowed_memory_nodes_read_cpuset_mask(tmp_path):
    status_path = tmp_path / "status"
    status_path.write_text("Name:\tpython\nMems_allowed_list:\t0-3,12\n")

    assert allowed_memory_nodes(status_path) == {0, 1, 2, 3, 12}


def test_host_memory_policy_binds_to_allowed_cpu_nodes(monkeypatch):
    monkeypatch.setattr(numa_policy, "cpu_numa_topology", lambda: {0: (0, 1), 1: (2, 3)})
    monkeypatch.setattr(numa_policy, "allowed_memory_nodes", lambda: {0, 1, 4, 12})
    installed_policies = []
    monkeypatch.setattr(numa_policy, "_set_membind", installed_policies.append)

    assert install_host_memory_policy() == (0, 1)
    assert installed_policies == [(0, 1)]


@pytest.mark.parametrize(
    ("cuda_visible_devices", "launcher_local_rank", "physical_gpu_id"),
    [
        (None, 2, 2),
        ("", 3, 3),
        ("3", 0, 3),
        ("2,0,3,1", 2, 3),
    ],
)
def test_physical_gpu_id_follows_visible_device_mapping(cuda_visible_devices, launcher_local_rank, physical_gpu_id):
    assert physical_gpu_id_for_worker(cuda_visible_devices, launcher_local_rank) == physical_gpu_id


def test_physical_gpu_id_rejects_rank_outside_visible_devices():
    with pytest.raises(RuntimeError, match="outside CUDA_VISIBLE_DEVICES"):
        physical_gpu_id_for_worker("2,3", 2)
