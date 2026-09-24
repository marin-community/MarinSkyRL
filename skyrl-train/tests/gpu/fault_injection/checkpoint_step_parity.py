"""Opt-in four-H100 Megatron checkpoint-to-next-step numerical parity gate.

Run the two cases in order as separate pytest processes on one node. The first
process records rank-local reference state before and after an uninterrupted
step; the second restores the committed S3 checkpoint into fresh actors and
requires exact state equality before and after replaying the SHA256-pinned batch.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from megatron.core import tensor_parallel
import pytest
import ray
import torch

from skyrl_train.checkpoint_generation import resolve_checkpoint_payload
from skyrl_train.distributed.megatron.megatron_utils import materialize_megatron_params
from skyrl_train.io import io
from skyrl_train.workers.megatron.megatron_worker import MegatronPolicyWorkerBase
from tests.checkpoint_parity_state import assert_state_equal, count_tensors, snapshot_value
from tests.gpu.gpu_ci.test_trainer_full_checkpointing import create_minimal_trainer, get_test_trainer_config
from tests.gpu.test_megatron_worker import get_test_training_batch
from tests.gpu.utils import import_worker


WORLD_SIZE = 4
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


class ParityMegatronPolicyWorker(MegatronPolicyWorkerBase):
    """Expose rank-local training state to the opt-in parity harness only."""

    def _rng_snapshot(self):
        return snapshot_value(
            {
                "generic": self.strategy.get_rng_state(),
                "cuda_tracker": tensor_parallel.get_cuda_rng_tracker().get_states(),
            }
        )

    def parity_rng_snapshot(self):
        return self._rank, self._rng_snapshot()

    def _state_snapshot(self):
        materialize_megatron_params(self.model.actor_module)
        model = self.model.actor_module[0]
        if hasattr(model, "module"):
            model = model.module
        model_state = {
            "parameters": {name: snapshot_value(value) for name, value in model.named_parameters()},
            "buffers": {name: snapshot_value(value) for name, value in model.named_buffers()},
        }
        model_shards = model.sharded_state_dict()
        optimizer_shards = self.optimizer.sharded_state_dict(
            model_shards,
            metadata={"distrib_optim_sharding_type": "dp_reshardable"},
        )
        return {
            "model": model_state,
            "optimizer": snapshot_value(optimizer_shards),
            "scheduler": snapshot_value(self.scheduler.state_dict()),
            "rng": self._rng_snapshot(),
        }

    def parity_write_snapshot(self, directory: str):
        state = self._state_snapshot()
        path = Path(directory) / f"rank-{self._rank}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(path)
        return self._rank, len(state["model"]["parameters"]), count_tensors(state["optimizer"])

    def parity_compare_snapshot(self, directory: str):
        path = Path(directory) / f"rank-{self._rank}.pt"
        expected = torch.load(path, map_location="cpu", weights_only=False)
        actual = self._state_snapshot()
        assert_state_equal(expected, actual, f"rank[{self._rank}]")
        return self._rank, len(actual["model"]["parameters"]), count_tensors(actual["optimizer"])

    def parity_model_changed_since(self, directory: str):
        path = Path(directory) / f"rank-{self._rank}.pt"
        previous = torch.load(path, map_location="cpu", weights_only=False)
        model = self.model.actor_module[0]
        if hasattr(model, "module"):
            model = model.module
        current = dict(model.named_parameters())
        assert current.keys() == previous["model"]["parameters"].keys()
        changed = sum(
            not torch.equal(value.detach().cpu(), previous["model"]["parameters"][name])
            for name, value in current.items()
        )
        return self._rank, changed


def _paths() -> tuple[Path, str]:
    local_input = Path(os.environ["CHECKPOINT_PARITY_LOCAL_ROOT"])
    if not local_input.is_absolute():
        raise ValueError("CHECKPOINT_PARITY_LOCAL_ROOT must be absolute")
    local = local_input.resolve()
    checkpoint_root = os.environ["CHECKPOINT_PARITY_S3_ROOT"].rstrip("/")
    if not local.is_absolute() or local == Path("/"):
        raise ValueError("CHECKPOINT_PARITY_LOCAL_ROOT must name one unique absolute directory")
    if not checkpoint_root.startswith("s3://marin-us-east-02a/tmp/ttl=14d/skyrl/users/atqamar/"):
        raise ValueError("CHECKPOINT_PARITY_S3_ROOT must be a unique east-region TTL prefix")
    return local, checkpoint_root


def _config(checkpoint_root: str, *, resume: bool):
    cfg = get_test_trainer_config("megatron", optimizer_checkpoint_sharding_type="dp_reshardable")
    cfg.trainer.policy.model.revision = MODEL_REVISION
    cfg.trainer.ckpt_path = f"{checkpoint_root}/checkpoints"
    cfg.trainer.export_path = f"{checkpoint_root}/exports"
    cfg.trainer.max_ckpts_to_keep = -1
    cfg.trainer.policy.megatron_config.checkpoint_plan_cache = True
    cfg.trainer.resume_mode = "latest" if resume else "none"
    return cfg


def _model_workers():
    PolicyWorker = ray.remote(num_gpus=1)(ParityMegatronPolicyWorker)
    return PolicyWorker, import_worker("megatron", "critic"), import_worker("megatron", "ref")


def _rank_results(trainer, method: str, argument: str | None = None):
    args = () if argument is None else (argument,)
    results = ray.get(trainer.policy_model.async_run_ray_method("pass_through", method, *args))
    assert sorted(rank for rank, *_ in results) == list(range(WORLD_SIZE))
    return results


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.megatron
def test_megatron_checkpoint_reference_records_uninterrupted_step(ray_init_fixture):
    local, checkpoint_root = _paths()
    assert not local.exists(), "Use a fresh CHECKPOINT_PARITY_LOCAL_ROOT"
    assert not io.exists(f"{checkpoint_root}/checkpoints/latest_ckpt_global_step.txt"), (
        "Use a fresh CHECKPOINT_PARITY_S3_ROOT"
    )
    local.mkdir(parents=True)
    cfg = _config(checkpoint_root, resume=False)
    trainer = create_minimal_trainer(cfg)
    try:
        trainer.build_models(*_model_workers())
        batch = get_test_training_batch(batch_size=WORLD_SIZE)
        batch_path = local / "fixed-batch.pt"
        torch.save(batch, batch_path)
        ray.get(trainer.policy_model.async_run_ray_method("mesh", "ppo_train", batch))
        trainer.global_step = 1

        before_save_rng = _rank_results(trainer, "parity_rng_snapshot")
        trainer.save_checkpoints()
        after_save_rng = _rank_results(trainer, "parity_rng_snapshot")
        for before, after in zip(sorted(before_save_rng), sorted(after_save_rng), strict=True):
            assert_state_equal(before[1], after[1], f"rank[{before[0]}].save_rng")

        checkpoint = f"{checkpoint_root}/checkpoints/global_step_1"
        assert resolve_checkpoint_payload(checkpoint, verify_files=True)
        pre = _rank_results(trainer, "parity_write_snapshot", str(local / "pre-step"))
        assert all(parameters > 0 and optimizer_tensors > 0 for _, parameters, optimizer_tensors in pre)

        replay_batch = torch.load(batch_path, map_location="cpu", weights_only=False)
        ray.get(trainer.policy_model.async_run_ray_method("mesh", "ppo_train", replay_batch))
        changed = _rank_results(trainer, "parity_model_changed_since", str(local / "pre-step"))
        assert any(count > 0 for _, count in changed), "The reference optimizer step did not change model weights"
        post = _rank_results(trainer, "parity_write_snapshot", str(local / "post-step"))
        assert [(rank, parameters, tensors) for rank, parameters, tensors in pre] == post

        manifest = {
            "fixture": "Qwen3-0.6B, four H100s, TP2/PP2/CP1/EP1; not Snowball 32-rank parity",
            "checkpoint": checkpoint,
            "batch_sha256": _sha256(batch_path),
            "model_revision": MODEL_REVISION,
            "source_step": 1,
            "replay_step": 2,
            "optimizer_format": "dp_reshardable",
        }
        (local / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    finally:
        trainer.cleanup_ray_actors()


@pytest.mark.megatron
def test_megatron_checkpoint_fresh_actors_match_next_step(ray_init_fixture):
    local, checkpoint_root = _paths()
    manifest = json.loads((local / "manifest.json").read_text(encoding="utf-8"))
    checkpoint = f"{checkpoint_root}/checkpoints/global_step_1"
    assert manifest["checkpoint"] == checkpoint
    assert manifest["model_revision"] == MODEL_REVISION
    batch_path = local / "fixed-batch.pt"
    assert _sha256(batch_path) == manifest["batch_sha256"]
    batch = torch.load(batch_path, map_location="cpu", weights_only=False)

    cfg = _config(checkpoint_root, resume=True)
    trainer = create_minimal_trainer(cfg)
    try:
        trainer.build_models(*_model_workers())
        loaded_step, loaded_path = trainer.load_checkpoints()
        assert loaded_step == manifest["source_step"]
        assert manifest["replay_step"] == loaded_step + 1
        assert loaded_path == resolve_checkpoint_payload(checkpoint, verify_files=True)
        pre = _rank_results(trainer, "parity_compare_snapshot", str(local / "pre-step"))
        assert all(parameters > 0 and optimizer_tensors > 0 for _, parameters, optimizer_tensors in pre)

        ray.get(trainer.policy_model.async_run_ray_method("mesh", "ppo_train", batch))
        post = _rank_results(trainer, "parity_compare_snapshot", str(local / "post-step"))
        assert post == pre
    finally:
        trainer.cleanup_ray_actors()
