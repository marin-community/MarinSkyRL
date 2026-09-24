"""Opt-in real-GPU/S3 checkpoint failure, retry, and fresh-process recovery test.

Run the two tests as separate pytest processes, in order, with one unique
CHECKPOINT_TEST_ROOT under an east-region TTL S3 prefix. The first test injects
an error after a distributed checkpoint save has completed, leaving an
uncommitted trainer generation; the second loads the retry through `latest`.
"""

from __future__ import annotations

import hashlib
import json
import os

import pytest
import ray

from skyrl_train.checkpoint_generation import COMMIT_FILENAME, resolve_checkpoint_payload
from skyrl_train.io import io
from skyrl_train.workers.megatron.megatron_worker import MegatronPolicyWorkerBase
from tests.gpu.gpu_ci.test_trainer_full_checkpointing import create_minimal_trainer, get_test_trainer_config
from tests.gpu.test_megatron_worker import get_test_training_batch
from tests.gpu.utils import import_worker


MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


class FailingOnceMegatronPolicyWorker(MegatronPolicyWorkerBase):
    def fail_after_next_distributed_save(self) -> int:
        """Fail after all ranks finish DCP, before trainer generation publication."""
        from skyrl_train.distributed.megatron import direct_checkpoint

        real_save = direct_checkpoint.checkpoint.save

        def save_then_fail(*args, **kwargs):
            real_save(*args, **kwargs)
            direct_checkpoint.checkpoint.save = real_save
            raise OSError("injected post-DCP save failure")

        direct_checkpoint.checkpoint.save = save_then_fail
        return self._rank

    def plan_cache_invalidated(self) -> tuple[int, bool]:
        from torch.distributed.checkpoint.planner import SavePlanner

        key = self.strategy._checkpoint_plan_cache_key
        caches = (
            SavePlanner._cached_save_plan,
            SavePlanner._cached_all_plans,
            SavePlanner._cached_global_plan,
            SavePlanner._cached_metadata,
            SavePlanner._cached_final_save_plan,
        )
        return self._rank, key is not None and all(key not in cache for cache in caches)


def _test_root() -> str:
    root = os.environ["CHECKPOINT_TEST_ROOT"].rstrip("/")
    if not root.startswith("s3://marin-us-east-02a/tmp/ttl=14d/skyrl/users/atqamar/"):
        raise ValueError("CHECKPOINT_TEST_ROOT must be a unique east-region TTL prefix")
    return root


def _config(root: str, *, resume: bool = False):
    cfg = get_test_trainer_config("megatron", optimizer_checkpoint_sharding_type="dp_reshardable")
    cfg.trainer.policy.model.revision = MODEL_REVISION
    cfg.trainer.ckpt_path = os.path.join(root, "checkpoints")
    cfg.trainer.export_path = os.path.join(root, "exports")
    cfg.trainer.max_ckpts_to_keep = -1
    cfg.trainer.policy.megatron_config.checkpoint_plan_cache = True
    cfg.trainer.resume_mode = "latest" if resume else "none"
    return cfg


def _step_path(root: str, step: int) -> str:
    return os.path.join(root, "checkpoints", f"global_step_{step}")


def _latest_path(root: str) -> str:
    return os.path.join(root, "checkpoints", "latest_ckpt_global_step.txt")


def _attempt_ids(step_path: str) -> set[str]:
    names = io.find_files(step_path)
    return {name.split("/_attempts/", 1)[1].split("/", 1)[0] for name in names if "/_attempts/" in name}


@pytest.mark.megatron
def test_megatron_failed_save_preserves_latest_and_retry_commits(ray_init_fixture) -> None:
    root = _test_root()
    assert not io.exists(_latest_path(root)), "Use a fresh CHECKPOINT_TEST_ROOT for the first phase"
    cfg = _config(root)
    trainer = create_minimal_trainer(cfg)
    FaultWorker = ray.remote(num_gpus=1)(FailingOnceMegatronPolicyWorker)
    try:
        trainer.build_models(FaultWorker, import_worker("megatron", "critic"), import_worker("megatron", "ref"))
        batch = get_test_training_batch(batch_size=4)
        ray.get(trainer.policy_model.async_run_ray_method("mesh", "ppo_train", batch))
        trainer.global_step = 1
        trainer.save_checkpoints()

        step_one = _step_path(root, 1)
        previous_commit = io.read_bytes(os.path.join(step_one, COMMIT_FILENAME))
        previous_pointer = io.read_bytes(_latest_path(root))
        assert previous_pointer == b"1"
        assert resolve_checkpoint_payload(step_one, verify_files=True)

        ray.get(trainer.policy_model.async_run_ray_method("mesh", "ppo_train", batch))
        trainer.global_step = 2
        armed = ray.get(trainer.policy_model.async_run_ray_method("pass_through", "fail_after_next_distributed_save"))
        assert sorted(armed) == list(range(4))
        with pytest.raises(Exception, match="injected post-DCP save failure"):
            trainer.save_checkpoints()
        cache_state = ray.get(trainer.policy_model.async_run_ray_method("pass_through", "plan_cache_invalidated"))
        assert sorted(cache_state) == [(rank, True) for rank in range(4)]

        step_two = _step_path(root, 2)
        assert io.read_bytes(_latest_path(root)) == previous_pointer
        assert io.read_bytes(os.path.join(step_one, COMMIT_FILENAME)) == previous_commit
        assert not io.exists(os.path.join(step_two, COMMIT_FILENAME))
        with pytest.raises(FileNotFoundError):
            resolve_checkpoint_payload(step_two, verify_files=True)
        failed_attempts = _attempt_ids(step_two)
        assert len(failed_attempts) == 1
        failed_attempt = next(iter(failed_attempts))
        assert not io.exists(os.path.join(step_two, "_attempts", failed_attempt, "checkpoint_manifest.json"))

        trainer.save_checkpoints()
        assert io.read_bytes(_latest_path(root)) == b"2"
        retry_commit = json.loads(io.read_bytes(os.path.join(step_two, COMMIT_FILENAME)))
        assert retry_commit["attempt_id"] != failed_attempt
        assert _attempt_ids(step_two) == {failed_attempt, retry_commit["attempt_id"]}
        retry_payload = resolve_checkpoint_payload(step_two, verify_files=True)
        assert retry_payload.endswith(retry_commit["attempt_id"])
        assert io.read_bytes(os.path.join(step_one, COMMIT_FILENAME)) == previous_commit

        evidence = {
            "failed_attempt": failed_attempt,
            "retry_attempt": retry_commit["attempt_id"],
            "step_one_commit_sha256": hashlib.sha256(previous_commit).hexdigest(),
            "step_two_commit_sha256": hashlib.sha256(
                io.read_bytes(os.path.join(step_two, COMMIT_FILENAME))
            ).hexdigest(),
        }
        io.write_bytes_atomic(os.path.join(root, "fault-evidence.json"), json.dumps(evidence, sort_keys=True).encode())
    finally:
        trainer.cleanup_ray_actors()


@pytest.mark.megatron
def test_megatron_fresh_process_resumes_retry_and_saves_next_step(ray_init_fixture) -> None:
    root = _test_root()
    evidence = json.loads(io.read_bytes(os.path.join(root, "fault-evidence.json")))
    cfg = _config(root, resume=True)
    trainer = create_minimal_trainer(cfg)
    try:
        trainer.build_models(
            import_worker("megatron", "policy"),
            import_worker("megatron", "critic"),
            import_worker("megatron", "ref"),
        )
        loaded_step, loaded_payload = trainer.load_checkpoints()
        assert loaded_step == 2
        assert loaded_payload.endswith(evidence["retry_attempt"])
        assert evidence["failed_attempt"] not in loaded_payload

        batch = get_test_training_batch(batch_size=4)
        ray.get(trainer.policy_model.async_run_ray_method("mesh", "ppo_train", batch))
        trainer.global_step = 3
        trainer.save_checkpoints()

        assert io.read_bytes(_latest_path(root)) == b"3"
        assert resolve_checkpoint_payload(_step_path(root, 3), verify_files=True)
        step_one_commit = io.read_bytes(os.path.join(_step_path(root, 1), COMMIT_FILENAME))
        step_two_commit = io.read_bytes(os.path.join(_step_path(root, 2), COMMIT_FILENAME))
        assert hashlib.sha256(step_one_commit).hexdigest() == evidence["step_one_commit_sha256"]
        assert hashlib.sha256(step_two_commit).hexdigest() == evidence["step_two_commit_sha256"]
    finally:
        trainer.cleanup_ray_actors()
