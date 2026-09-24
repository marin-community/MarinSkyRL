"""Resuming a Megatron checkpoint must not hold the gradient buffers while the optimizer state is rebuilt.

Building the optimizer's sharded state dict for loading allocates a full set of moments, and
optimizer.load_state_dict allocates the checkpointed ones before the first set is released. On a
policy that trains at the edge of device memory that second copy is what runs out, so the
gradients, which are not checkpointed, must be off the device for both calls.
"""

import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import fsspec
import pytest
import torch
from loguru import logger
from torch.distributed import checkpoint
from torch.distributed.checkpoint import DefaultSavePlanner

from tests.cpu.util import stub_megatron_modules

stub_megatron_modules()

from skyrl_train.distributed.megatron import direct_checkpoint, megatron_strategy  # noqa: E402


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
    monkeypatch.setattr(megatron_strategy.io, "exists", lambda path: Path(path).exists())
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
    _, states = strategy.load_checkpoint(model, str(tmp_path), optimizer=optimizer, scheduler=scheduler)
    return optimizer, grads, states


def test_resume_rebuilds_the_optimizer_state_with_the_gradients_off_the_device(monkeypatch, tmp_path):
    optimizer, grads, _ = _load_with_fakes(monkeypatch, tmp_path)
    assert optimizer.grads_resident_during == {"sharded_state_dict": False, "load_state_dict": False}
    assert grads.resident, "training needs its gradient buffers back after the optimizer state is restored"


def test_resume_returns_replicated_client_state_in_worker_expected_shape(monkeypatch, tmp_path):
    client_state = {"z_clip_state": {"warmup_buffer": [1.0]}, "stale_clip_state": {"ema": 0.5}}
    torch.save({"client_state": client_state, "tag": "step-1"}, tmp_path / "extra_state.pt")

    _, _, states = _load_with_fakes(monkeypatch, tmp_path)

    assert states == {"client_state": client_state}


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


def test_rank_rng_selection_rejects_changed_model_parallel_slice(monkeypatch):
    monkeypatch.setattr(megatron_strategy.dist, "get_world_size", lambda: 3)
    monkeypatch.setattr(megatron_strategy, "_rng_parallel_coordinates", lambda: (0, 1, 0, 0, 0, 0))
    with pytest.raises(ValueError, match="matching model-parallel geometry"):
        megatron_strategy._select_rank_rng_state([{"coordinates": (0, 0, 0, 0, 0, 0), "generic": "saved"}], 0)


def test_direct_save_reports_translation_subphases_and_preserves_tensor(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "global_step_4" / "policy"
    checkpoint_dir.mkdir(parents=True)
    filesystem = fsspec.filesystem("file")
    monkeypatch.setattr(direct_checkpoint.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(direct_checkpoint, "get_s3_fs", lambda: filesystem)
    monkeypatch.setattr(direct_checkpoint, "s3_refresh_if_expiring", lambda filesystem: None)
    monkeypatch.setattr(direct_checkpoint, "MCoreSavePlanner", DefaultSavePlanner)
    monkeypatch.setattr(
        direct_checkpoint, "_replace_state_dict_keys_with_sharded_keys", lambda state, _: (state, {}, {})
    )
    monkeypatch.setattr(direct_checkpoint, "mcore_to_pyt_state_dict", lambda state, _: state)

    output = io.StringIO()
    sink = logger.add(output, format="{message}")
    try:
        strategy = direct_checkpoint.DirectS3TorchDistSaveShardedStrategy(str(checkpoint_dir))
        strategy.keep_only_main_replica = False
        expected = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        strategy.save({"tensor": expected}, checkpoint_dir)
    finally:
        logger.remove(sink)

    restored = {"tensor": torch.zeros_like(expected)}
    checkpoint.load(restored, checkpoint_id=str(checkpoint_dir))
    torch.testing.assert_close(restored["tensor"], expected, rtol=0, atol=0)

    observations = [
        json.loads(line.removeprefix("checkpoint_observation "))
        for line in output.getvalue().splitlines()
        if line.startswith("checkpoint_observation ")
    ]
    phases = {item["phase"]: item for item in observations}
    assert {"dcp_key_replacement", "dcp_mcore_to_pyt", "dcp_translate"} <= phases.keys()
    assert all(phases[name]["outcome"] == "success" for name in phases)
    assert all(phases[name]["rank"] == "0" and phases[name]["step"] == "4" for name in phases)
    assert phases["dcp_translate"]["duration_seconds"] >= (
        phases["dcp_key_replacement"]["duration_seconds"] + phases["dcp_mcore_to_pyt"]["duration_seconds"]
    )
