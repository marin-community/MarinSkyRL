import pytest
import ray
from ray.util.placement_group import placement_group

from skyrl_train.config.numerics import NumericsMode, read_numerics
from skyrl_train.utils import get_ray_pg_ready_with_timeout, initialize_ray
from skyrl_train.utils.utils import validate_cfg
from skyrl_train.workers.megatron.megatron_worker import MegatronPolicyWorkerBase
from skyrl_train.workers.worker import PPORayActorGroup
from tests.gpu.grug_exact_checkpoint import exact_numerics_config, write_snowball_checkpoint
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.test_hero_megatron import write_tiny_hero_checkpoint

# The module classes the worker builds under trainer.numerics.mode=exact.
EXACT_CLASSES = {
    "model": "GrugExactGPTModel",
    "embed_norm": "ExactEmbeddingGatedNorm",
    "layer": "ExactTransformerLayer",
    "input_layernorm": "ExactInputGatedNorm",
    "self_attention": "ExactSelfAttention",
    "pre_mlp_layernorm": "ExactGatedNorm",
    "mlp": "ExactMoELayer",
    "router": "ExactTopKRouter",
    "experts": "ExactGroupedMLP",
    "shared_experts": "ExactSharedExpertMLP",
    "final_layernorm": "ExactFinalGatedNorm",
}
HERO_CLASSES = {
    "model": "GrugGPTModel",
    "embed_norm": "GrugGatedRMSNorm",
    "layer": "GrugTransformerLayer",
    "input_layernorm": "GrugGatedRMSNorm",
    "self_attention": "GrugSelfAttention",
    "pre_mlp_layernorm": "GrugGatedRMSNorm",
    "mlp": "GrugMoELayer",
    "router": "GrugTopKRouter",
    "experts": "TEGroupedMLP",
    "shared_experts": "GrugSharedExperts",
    "final_layernorm": "GrugGatedRMSNorm",
}


class ModuleReportWorkerBase(MegatronPolicyWorkerBase):
    def module_classes(self) -> dict[str, str]:
        model = self.actor_module[0]
        while hasattr(model, "module"):  # the DDP and the precision wrappers
            model = model.module
        layer = model.decoder.layers[0]
        modules = {
            "model": model,
            "embed_norm": model.embed_norm,
            "layer": layer,
            "input_layernorm": layer.input_layernorm,
            "self_attention": layer.self_attention,
            "pre_mlp_layernorm": layer.pre_mlp_layernorm,
            "mlp": layer.mlp,
            "router": layer.mlp.router,
            "experts": layer.mlp.experts,
            "shared_experts": layer.mlp.shared_experts,
            "final_layernorm": model.decoder.final_layernorm,
        }
        return {name: type(module).__name__ for name, module in modules.items()}


ModuleReportWorker = ray.remote(num_gpus=1)(ModuleReportWorkerBase)


@pytest.mark.parametrize(
    ("hero", "mode", "classes"),
    [(False, NumericsMode.EXACT, EXACT_CLASSES), (True, NumericsMode.NATIVE, HERO_CLASSES)],
    ids=["snowball_exact", "hero_native"],
)
def test_exact_numerics_select_the_exact_modules_for_snowball_and_the_stock_modules_for_hero(
    tmp_path, hero, mode, classes
):
    require_hoppers(1)
    model_path = tmp_path / "model"
    model_path.mkdir()
    if hero:
        write_tiny_hero_checkpoint(model_path)
    else:
        write_snowball_checkpoint(model_path, num_hidden_layers=2, num_local_experts=8)
    cfg = exact_numerics_config(str(model_path), policy_gpus=1, pipeline_stages=1, expert_parallel=1, engine_ranks=1)
    validate_cfg(cfg)
    resolution = read_numerics(cfg)
    assert resolution.mode is mode
    if mode is NumericsMode.NATIVE:
        assert any("Hero" in reason for reason in resolution.fallback_reasons), resolution
    initialize_ray(cfg)
    try:
        pg = placement_group([{"GPU": 1, "CPU": 1}], strategy="PACK")
        get_ray_pg_ready_with_timeout(pg, timeout=30)
        policy = PPORayActorGroup(
            cfg,
            num_nodes=1,
            num_gpus_per_node=1,
            ray_actor_type=ModuleReportWorker,
            pg=pg,
            num_gpus_per_actor=0.75,
            colocate_all=False,
            sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
            record_memory=cfg.trainer.policy.record_memory,
        )
        ray.get(policy.async_init_model(str(model_path)))
        assert ray.get(policy.async_run_ray_method("pass_through", "module_classes"))[0] == classes
    finally:
        ray.shutdown()
