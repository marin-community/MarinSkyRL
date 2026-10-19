import os

import ray

from skyrl_train.distributed.dispatch import MeshRank
from skyrl_train.workers.worker import PPORayActorGroup


@ray.remote
class EnvironmentActor:
    def __init__(self, *, rank, **kwargs):
        self.rank = rank

    def get_master_addr_port(self):
        return "127.0.0.1", 12345

    def init_worker_process_group(self, master_addr, master_port):
        self.buffer_size_at_init = os.environ.get("NCCL_BUFFSIZE")

    def get_mesh_rank(self):
        return MeshRank(dp=self.rank, sp=0, tp=0, pp=0, world_size=2, dp_size=2, pp_size=1)

    def nccl_buffer_size(self):
        return self.buffer_size_at_init

    def close_telemetry(self):
        pass


def test_actor_environment_is_set_before_distributed_initialization(ray_init):
    group = PPORayActorGroup(
        cfg=None,
        num_nodes=2,
        num_gpus_per_node=1,
        ray_actor_type=EnvironmentActor,
        num_gpus_per_actor=0,
        actor_env_vars={"NCCL_BUFFSIZE": "262144"},
    )
    try:
        assert ray.get([info.handle.nccl_buffer_size.remote() for info in group.actor_infos]) == ["262144", "262144"]
    finally:
        group.kill_actors()
