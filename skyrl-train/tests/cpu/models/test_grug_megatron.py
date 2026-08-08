"""CPU construction and conversion coverage for the public Grug Bridge seam."""

from __future__ import annotations

from types import SimpleNamespace

from omegaconf import OmegaConf
import torch
import torch.distributed as dist
import pytest
from transformers import AutoModelForCausalLM


pytest.importorskip("megatron.bridge")

from megatron.bridge import AutoBridge  # noqa: E402
from megatron.core import parallel_state  # noqa: E402

from skyrl_train.models.grug_megatron import GrugMoeBridge, GrugMoeMegatronModel  # noqa: E402
from skyrl_train.models.grug_moe import GrugMoeRouter  # noqa: E402
from skyrl_train.distributed.megatron.megatron_strategy import MegatronStrategy  # noqa: E402
from skyrl_train.workers.megatron.megatron_worker import MegatronWorker  # noqa: E402
from tests.grug_training_parity import ORACLE_FIXTURE_DIR  # noqa: E402


@pytest.fixture
def model_parallel_world(tmp_path):
    initialized_dist = not dist.is_initialized()
    if initialized_dist:
        dist.init_process_group(
            "gloo",
            init_method=f"file://{tmp_path / 'process-group'}",
            rank=0,
            world_size=1,
        )
    initialized_model_parallel = not parallel_state.is_initialized()
    if initialized_model_parallel:
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=1,
        )
    try:
        yield
    finally:
        if initialized_model_parallel:
            parallel_state.destroy_model_parallel()
        if initialized_dist:
            dist.destroy_process_group()


def test_grug_bridge_constructs_and_round_trips_exact_hf_state(model_parallel_world):
    source = AutoModelForCausalLM.from_pretrained(
        ORACLE_FIXTURE_DIR,
        local_files_only=True,
        dtype=torch.float32,
    )
    source_state = source.state_dict()
    bridge = AutoBridge.from_hf_pretrained(
        ORACLE_FIXTURE_DIR,
        local_files_only=True,
        dtype=torch.float32,
    )
    provider = bridge.to_megatron_provider()
    provider.apply_overrides_and_finalize(
        dtype=torch.float32,
        overrides={
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 1,
            "expert_model_parallel_size": 1,
            "expert_tensor_parallel_size": 1,
            "virtual_pipeline_model_parallel_size": None,
            "persist_layer_norm": False,
        },
    )
    models = provider.provide_distributed_model(
        wrap_with_ddp=False,
        use_cpu_initialization=True,
        mixed_precision_wrapper=None,
    )

    assert len(models) == 1
    assert isinstance(models[0], GrugMoeMegatronModel)
    exported = dict(bridge.export_hf_weights(models, cpu=True, show_progress=False))
    assert exported.keys() == source_state.keys()
    for name, expected in source_state.items():
        torch.testing.assert_close(exported[name], expected, rtol=0, atol=0)

    config = GrugMoeBridge.megatron_to_hf_config(provider)
    assert config["model_type"] == "grug_moe"
    assert config["dtype"] == "float32"

    sharded_state = models[0].sharded_state_dict()
    assert set(sharded_state) == {f"model.{name}" for name in source_state}

    models[0].bfloat16()
    router_biases = [module.bias for module in models[0].modules() if isinstance(module, GrugMoeRouter)]
    assert router_biases
    assert all(bias.dtype == torch.float32 for bias in router_biases)


def test_grug_provider_rejects_non_unit_topology_before_construction(model_parallel_world):
    bridge = AutoBridge.from_hf_pretrained(
        ORACLE_FIXTURE_DIR,
        local_files_only=True,
        dtype=torch.float32,
    )
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.tensor_model_parallel_size = 2

    with pytest.raises(ValueError, match="world=TP=EP=ETP=PP=CP=VP=1"):
        provider.provide()


@pytest.mark.parametrize(
    ("is_policy_worker", "use_sample_packing", "transformer_config_kwargs", "message"),
    (
        (False, False, {}, "policy rank only"),
        (True, True, {}, "use_sample_packing=false"),
        (True, False, {"virtual_pipeline_model_parallel_size": 2}, "world=TP=EP=ETP=PP=CP=VP=1"),
    ),
)
def test_worker_rejects_unadmitted_grug_before_tokenizer(
    is_policy_worker,
    use_sample_packing,
    transformer_config_kwargs,
    message,
):
    worker = object.__new__(MegatronWorker)
    worker._world_size = 1
    worker.cfg = OmegaConf.create(
        {"trainer": {"use_sample_packing": use_sample_packing, "gradient_checkpointing": False}}
    )
    worker.strategy = SimpleNamespace(hf_config=None)
    megatron_config = OmegaConf.create(
        {
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 1,
            "expert_model_parallel_size": 1,
            "expert_tensor_parallel_size": None,
        }
    )

    with pytest.raises(ValueError, match=message):
        worker.init_configs(
            ORACLE_FIXTURE_DIR,
            megatron_config,
            {},
            OmegaConf.create(transformer_config_kwargs),
            is_policy_worker=is_policy_worker,
            bf16=False,
            flash_attn=False,
        )


class _OutcomeOptimizer:
    def __init__(self, successful: bool):
        self.successful = successful
        self.zeroed = False

    def step(self):
        return self.successful, torch.tensor(3.0), None

    def zero_grad(self):
        self.zeroed = True


class _StepScheduler:
    def __init__(self):
        self.steps = 0

    def step(self, increment: int):
        self.steps += increment


@pytest.mark.parametrize("successful", [False, True])
def test_megatron_strategy_propagates_optimizer_outcome(successful):
    strategy = object.__new__(MegatronStrategy)
    strategy.last_optimizer_step_succeeded = not successful
    optimizer = _OutcomeOptimizer(successful)
    scheduler = _StepScheduler()

    grad_norm = strategy.optimizer_step(optimizer, model=None, scheduler=scheduler)

    assert strategy.last_optimizer_step_succeeded is successful
    assert scheduler.steps == int(successful)
    assert optimizer.zeroed
    torch.testing.assert_close(grad_norm, torch.tensor(3.0), rtol=0, atol=0)
