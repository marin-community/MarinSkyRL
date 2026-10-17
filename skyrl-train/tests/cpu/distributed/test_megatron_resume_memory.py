"""Resuming a Megatron checkpoint must not hold the gradient buffers while the optimizer state is rebuilt.

Building the optimizer's sharded state dict for loading allocates a full set of moments, and
optimizer.load_state_dict allocates the checkpointed ones before the first set is released. On a
policy that trains at the edge of device memory that second copy is what runs out, so the
gradients, which are not checkpointed, must be off the device for both calls.
"""

import contextlib
from types import SimpleNamespace

import pytest
import torch

from tests.cpu.util import stub_megatron_modules

stub_megatron_modules()

from skyrl_train.distributed.megatron import megatron_strategy  # noqa: E402
from skyrl_train.distributed.megatron.grug_muonh import MegatronGrugMuonH  # noqa: E402


class _Grads:
    resident = True


class _Optimizer:
    def __init__(self, grads: _Grads):
        self.grads = grads
        self.optimizer = object()
        self.grads_resident_during = {}

    def sharded_state_dict(self, model_sharded_state_dict, *, is_loading, metadata):
        self.grads_resident_during["sharded_state_dict"] = self.grads.resident
        return {}

    def load_state_dict(self, state_dict):
        self.grads_resident_during["load_state_dict"] = self.grads.resident


def _load_with_fakes(monkeypatch, tmp_path):
    grads = _Grads()
    monkeypatch.setattr(
        megatron_strategy, "offload_megatron_grads_to_cpu", lambda model: setattr(grads, "resident", False)
    )
    monkeypatch.setattr(megatron_strategy, "load_megatron_grads_to_gpu", lambda model: setattr(grads, "resident", True))
    source = "s3://bucket/checkpoint"
    monkeypatch.setattr(megatron_strategy.io, "exists", lambda path: path == source)
    monkeypatch.setattr(megatron_strategy, "remote_checkpoint_metadata", lambda path: contextlib.nullcontext(tmp_path))
    monkeypatch.setattr(megatron_strategy, "DirectS3TorchDistLoadShardedStrategy", lambda path: None)
    monkeypatch.setattr(
        megatron_strategy.dist_checkpointing,
        "load_common_state_dict",
        lambda read_dir: {"optimizer": {"param_state_sharding_type": "fully_reshardable"}},
    )
    monkeypatch.setattr(
        megatron_strategy.dist_checkpointing,
        "load",
        lambda **kwargs: {"model": {}, "optimizer": {}, "lr_scheduler": {}},
    )
    monkeypatch.setattr(megatron_strategy, "FullyParallelLoadStrategyWrapper", lambda strategy, group: strategy)
    monkeypatch.setattr(
        megatron_strategy.mpu, "get_data_parallel_group", lambda with_context_parallel: None, raising=False
    )
    strategy = megatron_strategy.MegatronStrategy.__new__(megatron_strategy.MegatronStrategy)
    strategy.optimizer_config = None
    monkeypatch.setattr(strategy, "log", lambda *msg: None)
    module = SimpleNamespace(sharded_state_dict=lambda: {}, load_state_dict=lambda state, strict: None)
    model = SimpleNamespace(actor_module=[module])
    optimizer = _Optimizer(grads)
    scheduler = SimpleNamespace(state_dict=lambda: {}, load_state_dict=lambda state: None)
    strategy.load_checkpoint(model, source, optimizer=optimizer, scheduler=scheduler)
    return optimizer, grads


def test_resume_rebuilds_the_optimizer_state_with_the_gradients_off_the_device(monkeypatch, tmp_path):
    optimizer, grads = _load_with_fakes(monkeypatch, tmp_path)
    assert optimizer.grads_resident_during == {"sharded_state_dict": False, "load_state_dict": False}
    assert grads.resident, "training needs its gradient buffers back after the optimizer state is restored"


def test_rank_rng_selection_preserves_exact_rank_and_maps_resized_dp(monkeypatch):
    rank_states = [
        {"coordinates": (0, 0, 0, 0, 0, 0), "generic": "dp0"},
        {"coordinates": (0, 0, 0, 0, 0, 1), "generic": "dp1"},
    ]
    monkeypatch.setattr(megatron_strategy.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(megatron_strategy, "_rng_parallel_coordinates", lambda: (0, 0, 0, 0, 0, 1))
    assert megatron_strategy._select_rank_rng_state(rank_states, 1)["generic"] == "dp1"

    monkeypatch.setattr(megatron_strategy.dist, "get_world_size", lambda: 3)
    monkeypatch.setattr(megatron_strategy, "_rng_parallel_coordinates", lambda: (0, 0, 0, 0, 0, 2))
    assert megatron_strategy._select_rank_rng_state(rank_states, 2)["generic"] == "dp0"


def test_rank_rng_selection_keeps_common_state_for_changed_model_parallel_slice(monkeypatch):
    monkeypatch.setattr(megatron_strategy.dist, "get_world_size", lambda: 3)
    monkeypatch.setattr(megatron_strategy, "_rng_parallel_coordinates", lambda: (0, 1, 0, 0, 0, 0))
    assert (
        megatron_strategy._select_rank_rng_state([{"coordinates": (0, 0, 0, 0, 0, 0), "generic": "saved"}], 0) is None
    )


class _CheckpointOptimizer:
    """CPU boundary for Megatron's wrapper around a real lazy-state optimizer."""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.config = None
        self.init_state_fn = lambda inner, config: inner.initialize_state()

    def sharded_state_dict(self, model_state, *, metadata):
        return [
            {key: self.optimizer.state[parameter][key] for key in ("step", "exp_avg", "exp_avg_sq")}
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        ]


class _CheckpointChain(megatron_strategy.ChainedOptimizer):
    def __init__(self, children):
        self.chained_optimizers = children

    def sharded_state_dict(self, model_state, *, metadata):
        return [child.sharded_state_dict(model_state, metadata=metadata) for child in self.chained_optimizers]


@pytest.mark.parametrize("chained", [False, True])
def test_save_before_first_update_initializes_missing_moments_without_changing_weights(monkeypatch, tmp_path, chained):
    fresh = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    existing = torch.nn.Parameter(torch.tensor([4.0, 5.0]))
    inner = MegatronGrugMuonH(
        [{"params": [fresh, existing], "optimizer": "grug_adamh"}],
        lr=0.01,
        betas=(0.9, 0.95),
        momentum=0.9,
        nesterov=True,
        ns_steps=5,
        eps=1e-8,
        muon_eps=1e-8,
        qkv_split_shapes=(1, 1, 1),
    )
    inner.state[existing] = {"step": torch.tensor(3), "exp_avg": torch.ones(2), "exp_avg_sq": torch.full((2,), 7.0)}
    optimizer = _CheckpointOptimizer(inner)
    if chained:
        optimizer = _CheckpointChain([optimizer])

    def save(**kwargs):
        torch.save(kwargs["sharded_state_dict"], tmp_path / "state.pt")

    monkeypatch.setattr(megatron_strategy.dist, "barrier", lambda: None)
    monkeypatch.setattr(megatron_strategy.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(megatron_strategy.dist, "all_gather_object", lambda states, state: states.__setitem__(0, state))
    for dimension in ("tensor_model", "pipeline_model", "context", "expert_model", "expert_tensor", "data"):
        monkeypatch.setattr(megatron_strategy.mpu, f"get_{dimension}_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(
        megatron_strategy.tensor_parallel,
        "get_cuda_rng_tracker",
        lambda: SimpleNamespace(get_states=lambda: {}),
        raising=False,
    )
    monkeypatch.setattr(megatron_strategy.dist_checkpointing, "save", save)
    monkeypatch.setattr(megatron_strategy, "get_default_save_sharded_strategy", lambda name: None)
    monkeypatch.setattr(megatron_strategy, "FullyParallelSaveStrategyWrapper", lambda strategy, group: strategy)
    monkeypatch.setattr(megatron_strategy.mpu, "get_data_parallel_group", lambda **kwargs: None, raising=False)
    strategy = megatron_strategy.MegatronStrategy(
        SimpleNamespace(optimizer_checkpoint_sharding_type="fully_reshardable")
    )
    monkeypatch.setattr(megatron_strategy.ckpt_base, "async_calls", SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(strategy, "get_rng_state", lambda: {})
    monkeypatch.setattr(strategy, "is_rank_0", lambda: False)
    monkeypatch.setattr(strategy, "log", lambda *args: None)
    module = SimpleNamespace(sharded_state_dict=lambda: {"fresh": fresh, "existing": existing})
    strategy.save_checkpoint(SimpleNamespace(actor_module=[module]), str(tmp_path), 0, optimizer=optimizer)

    saved = torch.load(tmp_path / "state.pt", weights_only=True)
    moments = saved["optimizer"][0] if chained else saved["optimizer"]
    torch.testing.assert_close(saved["model"]["fresh"], torch.tensor([2.0, 3.0]), rtol=0, atol=0)
    torch.testing.assert_close(saved["model"]["existing"], torch.tensor([4.0, 5.0]), rtol=0, atol=0)
    assert moments[0]["step"].item() == 0
    assert torch.equal(moments[0]["exp_avg"], torch.zeros(2))
    assert torch.equal(moments[0]["exp_avg_sq"], torch.zeros(2))
    assert moments[1]["step"].item() == 3
    assert torch.equal(moments[1]["exp_avg"], torch.ones(2))
    assert torch.equal(moments[1]["exp_avg_sq"], torch.full((2,), 7.0))
