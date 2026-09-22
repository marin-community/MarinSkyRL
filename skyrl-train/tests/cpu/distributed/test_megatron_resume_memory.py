"""Resuming a Megatron checkpoint must not hold the gradient buffers while the optimizer state is rebuilt.

Building the optimizer's sharded state dict for loading allocates a full set of moments, and
optimizer.load_state_dict allocates the checkpointed ones before the first set is released. On a
policy that trains at the edge of device memory that second copy is what runs out, so the
gradients, which are not checkpointed, must be off the device for both calls.
"""

import contextlib
from types import SimpleNamespace

from tests.cpu.util import stub_megatron_modules

stub_megatron_modules()

from skyrl_train.distributed.megatron import megatron_strategy  # noqa: E402


class _Grads:
    resident = True


class _Optimizer:
    def __init__(self, grads: _Grads):
        self.grads = grads
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
    monkeypatch.setattr(megatron_strategy.io, "exists", lambda path: path == str(tmp_path))
    monkeypatch.setattr(megatron_strategy.io, "node_cached_read_dir", lambda path, cache: contextlib.nullcontext(path))
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
    monkeypatch.setattr(megatron_strategy, "get_default_load_sharded_strategy", lambda read_dir: None)
    monkeypatch.setattr(megatron_strategy, "FullyParallelLoadStrategyWrapper", lambda strategy, group: strategy)
    monkeypatch.setattr(
        megatron_strategy.mpu, "get_data_parallel_group", lambda with_context_parallel: None, raising=False
    )
    strategy = megatron_strategy.MegatronStrategy.__new__(megatron_strategy.MegatronStrategy)
    monkeypatch.setattr(strategy, "log", lambda *msg: None)
    module = SimpleNamespace(sharded_state_dict=lambda: {}, load_state_dict=lambda state, strict: None)
    model = SimpleNamespace(actor_module=[module])
    optimizer = _Optimizer(grads)
    scheduler = SimpleNamespace(state_dict=lambda: {}, load_state_dict=lambda state: None)
    strategy.load_checkpoint(model, str(tmp_path), optimizer=optimizer, scheduler=scheduler)
    return optimizer, grads


def test_resume_rebuilds_the_optimizer_state_with_the_gradients_off_the_device(monkeypatch, tmp_path):
    optimizer, grads = _load_with_fakes(monkeypatch, tmp_path)
    assert optimizer.grads_resident_during == {"sharded_state_dict": False, "load_state_dict": False}
    assert grads.resident, "training needs its gradient buffers back after the optimizer state is restored"
