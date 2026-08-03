"""Validate FSDP2 CPU-offload placement through the production policy worker.

Run this file explicitly on one four-GPU GH200 node. It is intentionally outside
pytest discovery because it starts a local Ray cluster and materializes a real
FSDP2 policy on every GPU.
"""

import os
import sys

import ray
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from ray.util.placement_group import placement_group

from skyrl_train.entrypoints.main_base import config_dir
from skyrl_train.numa_policy import MEMORY_POLICY_BIND, NUMA_AFFINITY_ENV
from skyrl_train.utils.utils import get_ray_pg_ready_with_timeout, initialize_ray
from skyrl_train.workers.fsdp.fsdp_worker import PolicyWorker
from skyrl_train.workers.worker import PPORayActorGroup


MODEL = os.environ.get("NUMA_TEST_MODEL", "Qwen/Qwen3-0.6B")
WORLD_SIZE = 4


def policy_config():
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_base_config")
    OmegaConf.set_struct(cfg, False)
    cfg.trainer.strategy = "fsdp2"
    cfg.trainer.policy.model.path = MODEL
    cfg.trainer.policy.fsdp_config.cpu_offload = True
    cfg.trainer.policy.fsdp_config.fsdp_size = WORLD_SIZE
    cfg.trainer.policy.fsdp_config.expert_model_parallel_size = 1
    cfg.trainer.policy.sequence_parallel_size = 1
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.policy_num_gpus_per_node = WORLD_SIZE
    cfg.trainer.placement.policy_per_gpu_bundles = True
    cfg.trainer.placement.policy_force_cvd_mask = True
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_kl_in_reward = False
    cfg.trainer.gradient_checkpointing = False
    cfg.trainer.flash_attn = False
    cfg.trainer.use_sample_packing = False
    cfg.trainer.logger = "console"
    return cfg


def main() -> int:
    os.environ[NUMA_AFFINITY_ENV] = "1"
    cfg = policy_config()
    initialize_ray(cfg)
    try:
        group = placement_group([{"GPU": 1, "CPU": 1} for _ in range(WORLD_SIZE)], strategy="PACK")
        get_ray_pg_ready_with_timeout(group, timeout=120)

        policy = PPORayActorGroup(
            cfg,
            num_nodes=1,
            num_gpus_per_node=WORLD_SIZE,
            ray_actor_type=PolicyWorker,
            pg=group,
            num_gpus_per_actor=1,
            colocate_all=False,
            sequence_parallel_size=1,
            pin_to_ray_gpu_id=True,
            force_cvd_mask=True,
        )
        ray.get(policy.async_init_model(MODEL))
        diagnostics = ray.get(policy.async_run_ray_method("pass_through", "get_cpu_offload_numa_diagnostics"))
    finally:
        ray.shutdown()

    for diagnostic in sorted(diagnostics, key=lambda item: item.rank):
        print(diagnostic, flush=True)
        cpu_nodes = set(diagnostic.cpu_nodes)
        policy_nodes = set(diagnostic.memory_policy.nodes)
        page_nodes = set(diagnostic.page_nodes)
        affinity_nodes = diagnostic.affinity_nodes
        assert diagnostic.memory_policy.mode == MEMORY_POLICY_BIND, diagnostic
        assert policy_nodes == cpu_nodes, diagnostic
        assert diagnostic.sampled_pages > 0, diagnostic
        assert page_nodes <= cpu_nodes, diagnostic
        assert len(affinity_nodes) == 1, diagnostic
        local_pages = diagnostic.page_nodes.get(affinity_nodes[0], 0)
        assert local_pages / diagnostic.sampled_pages >= 0.95, diagnostic

    print(f"FSDP2_CPU_OFFLOAD_NUMA_OK world={WORLD_SIZE} model={MODEL}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
