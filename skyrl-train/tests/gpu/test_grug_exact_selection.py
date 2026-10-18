import ray
from ray.util.placement_group import placement_group

from skyrl_train.models.grug_megatron import GrugGPTModel
from skyrl_train.utils import get_ray_pg_ready_with_timeout, initialize_ray
from skyrl_train.utils.utils import validate_cfg
from skyrl_train.workers.megatron.megatron_worker import MegatronPolicyWorkerBase
from skyrl_train.workers.worker import PPORayActorGroup
from tests.gpu.grug_exact_checkpoint import exact_numerics_config
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.test_hero_megatron import write_tiny_hero_checkpoint


class ModelClassWorkerBase(MegatronPolicyWorkerBase):
    def model_class(self) -> str:
        model = self.actor_module[0]
        while hasattr(model, "module"):  # the DDP and the precision wrappers
            model = model.module
        return type(model).__name__


ModelClassWorker = ray.remote(num_gpus=1)(ModelClassWorkerBase)


def test_exact_numerics_build_the_stock_model_for_hero(tmp_path):
    require_hoppers(1)
    model_path = tmp_path / "model"
    model_path.mkdir()
    write_tiny_hero_checkpoint(model_path)
    cfg = exact_numerics_config(str(model_path), policy_gpus=1, pipeline_stages=1, expert_parallel=1, engine_ranks=1)
    validate_cfg(cfg)
    initialize_ray(cfg)
    try:
        pg = placement_group([{"GPU": 1, "CPU": 1}], strategy="PACK")
        get_ray_pg_ready_with_timeout(pg, timeout=30)
        policy = PPORayActorGroup(
            cfg,
            num_nodes=1,
            num_gpus_per_node=1,
            ray_actor_type=ModelClassWorker,
            pg=pg,
            num_gpus_per_actor=0.75,
            colocate_all=False,
            sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
            record_memory=cfg.trainer.policy.record_memory,
        )
        ray.get(policy.async_init_model(str(model_path)))
        assert ray.get(policy.async_run_ray_method("pass_through", "model_class")) == [GrugGPTModel.__name__]
    finally:
        ray.shutdown()
