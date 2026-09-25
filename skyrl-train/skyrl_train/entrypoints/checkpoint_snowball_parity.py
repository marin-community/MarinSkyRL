"""Opt-in 32-rank Snowball checkpoint-to-next-step parity entrypoint.

Run reference and resume in distinct Iris task-runtime processes against the
same unique S3 root. This is a numerical recovery gate, not a save benchmark.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from enum import StrEnum
from unittest.mock import MagicMock

from megatron.core import tensor_parallel
import numpy as np
from omegaconf import DictConfig, OmegaConf
import ray
import torch
from torch.utils.data import Dataset

from skyrl_train.checkpoint_parity_digest import digest_value
from skyrl_train.checkpoint_generation import resolve_checkpoint_payload
from skyrl_train.distributed.megatron.megatron_utils import materialize_megatron_params
from skyrl_train.io import io
from skyrl_train.tokenizer import create_tokenizer
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils.tracking import Tracking
from skyrl_train.workers.megatron.megatron_worker import CriticWorker, MegatronPolicyWorkerBase, RefWorker


WORLD_SIZE = 32
SEQUENCE_LENGTH = 128
RESPONSE_LENGTH = 16
EXPECTED_GEOMETRY = (1, 2, 2, 8)
S3_PREFIX = "s3://marin-us-east-02a/tmp/ttl=14d/skyrl/users/atqamar/"
REFERENCE_NAME = "checkpoint-parity-reference.json"


class Phase(StrEnum):
    REFERENCE = "reference"
    RESUME = "resume"


class FixedDataset(Dataset):
    """Provide a stable dataloader state for the full trainer checkpoint."""

    def __len__(self) -> int:
        return WORLD_SIZE

    def __getitem__(self, index: int) -> tuple[list[dict[str, str]], None]:
        return [{"role": "user", "content": f"Fixed parity prompt {index}"}], None

    def collate_fn(self, batch: list[object]) -> list[object]:
        return batch


class SnowballParityPolicyWorker(MegatronPolicyWorkerBase):
    """Read-only, rank-local numerical state inspection for the parity gate."""

    def parity_digest_rng(self) -> tuple[int, dict[str, int | str]]:
        return self._rank, digest_value(
            {"generic": self.strategy.get_rng_state(), "cuda_tracker": tensor_parallel.get_cuda_rng_tracker().get_states()}
        )

    def parity_digest_state(self) -> tuple[int, dict[str, dict[str, int | str]]]:
        materialize_megatron_params(self.model.actor_module)
        model = self.model.actor_module[0]
        if hasattr(model, "module"):
            model = model.module
        model_shards = model.sharded_state_dict()
        optimizer_shards = self.optimizer.sharded_state_dict(
            model_shards, metadata={"distrib_optim_sharding_type": "dp_reshardable"}
        )
        return self._rank, {
            "model": digest_value(
                {"parameters": dict(model.named_parameters()), "buffers": dict(model.named_buffers())}
            ),
            "optimizer": digest_value(optimizer_shards),
            "scheduler": digest_value(self.scheduler.state_dict()),
            "rng": self.parity_digest_rng()[1],
        }


def _root() -> str:
    root = os.environ["CHECKPOINT_PARITY_S3_ROOT"].rstrip("/")
    if not root.startswith(S3_PREFIX) or root == S3_PREFIX.rstrip("/"):
        raise ValueError("CHECKPOINT_PARITY_S3_ROOT must be a unique east-region TTL prefix")
    return root


def _phase() -> Phase:
    return Phase(os.environ["CHECKPOINT_PARITY_PHASE"])


def _prepare_config(cfg: DictConfig, root: str) -> tuple[DictConfig, str]:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    geometry = cfg.trainer.policy.megatron_config
    actual = (
        geometry.tensor_model_parallel_size,
        geometry.pipeline_model_parallel_size,
        geometry.context_parallel_size,
        geometry.expert_model_parallel_size,
    )
    if cfg.trainer.strategy != "megatron" or actual != EXPECTED_GEOMETRY:
        raise ValueError(f"Snowball parity requires Megatron TP1/PP2/CP2/EP8, got {actual}")
    if cfg.trainer.placement.policy_num_nodes != 4 or cfg.trainer.placement.policy_num_gpus_per_node != 8:
        raise ValueError("Snowball parity requires four policy nodes with eight GPUs each")
    if geometry.optimizer_checkpoint_sharding_type != "dp_reshardable":
        raise ValueError("Snowball parity requires the benchmark's dp_reshardable optimizer format")
    if cfg.trainer.critic.model.path or cfg.trainer.algorithm.use_kl_loss or cfg.trainer.algorithm.use_kl_in_reward:
        raise ValueError("Snowball parity starts policy workers only; critic and reference must be disabled")
    cfg.trainer.train_batch_size = WORLD_SIZE
    cfg.trainer.policy_mini_batch_size = WORLD_SIZE
    OmegaConf.update(cfg, "trainer.algorithm.max_seq_len", SEQUENCE_LENGTH, force_add=True)
    cfg.trainer.ckpt_path = f"{root}/checkpoints"
    cfg.trainer.export_path = f"{root}/exports"
    cfg.trainer.max_ckpts_to_keep = -1
    cfg.trainer.ckpt_interval = 1
    cfg.trainer.logger = "console"
    cfg.generator.run_engines_locally = False
    fingerprint = hashlib.sha256(OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True).encode()).hexdigest()
    return cfg, fingerprint


def _batch() -> TrainingInputBatch:
    positions = torch.arange(SEQUENCE_LENGTH, dtype=torch.long)
    sequences = (positions[None, :] + torch.arange(WORLD_SIZE)[:, None]) % 1000 + 100
    action_shape = (WORLD_SIZE, RESPONSE_LENGTH)
    batch = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": torch.ones_like(sequences),
            "action_log_probs": torch.full(action_shape, 0.1),
            "base_action_log_probs": torch.full(action_shape, 0.2),
            "rollout_logprobs": torch.full(action_shape, 0.11),
            "values": torch.full(action_shape, 0.1),
            "returns": torch.full(action_shape, 0.1),
            "advantages": torch.full(action_shape, 0.5),
            "loss_mask": torch.ones(action_shape),
            "response_mask": torch.ones(action_shape),
        }
    )
    batch.metadata = {"response_length": RESPONSE_LENGTH, "global_step": 1}
    return batch


def _trainer(cfg: DictConfig) -> RayPPOTrainer:
    tokenizer = create_tokenizer(
        model_path=cfg.trainer.policy.model.tokenizer_path,
        disable_fast_tokenizer=cfg.trainer.disable_fast_tokenizer,
        revision=cfg.trainer.policy.model.get("tokenizer_revision"),
    )
    tracker = Tracking(cfg.trainer.project_name, cfg.trainer.run_name, backends="console", config=cfg)
    return RayPPOTrainer(
        cfg=cfg,
        tracker=tracker,
        tokenizer=tokenizer,
        train_dataset=FixedDataset(),
        eval_dataset=None,
        inference_engine_client=None,
        trajectory_runner=MagicMock(),
    )


def _rank_digests(trainer: RayPPOTrainer, method: str) -> dict[str, object]:
    results = ray.get(trainer.policy_model.async_run_ray_method("pass_through", method))
    ranks = sorted(rank for rank, _ in results)
    if ranks != list(range(WORLD_SIZE)):
        raise AssertionError(f"Expected exactly {WORLD_SIZE} distinct policy ranks, got {ranks}")
    return {str(rank): value for rank, value in results}


def _compare_rank_digests(expected: dict[str, object], actual: dict[str, object], stage: str) -> None:
    if expected.keys() != actual.keys():
        raise AssertionError(f"{stage}: policy rank set differs")
    for rank in sorted(expected, key=int):
        before = expected[rank]
        after = actual[rank]
        if not isinstance(before, dict) or not isinstance(after, dict):
            raise AssertionError(f"{stage}: rank {rank} has malformed digest")
        if before.keys() != after.keys() or before.keys() != {"model", "optimizer", "scheduler", "rng"}:
            raise AssertionError(f"{stage}: rank {rank} state components differ")
        for component in ("model", "optimizer", "scheduler", "rng"):
            if before[component] != after[component]:
                raise AssertionError(f"{stage}: rank {rank} {component} differs: {before[component]} != {after[component]}")


def _driver_rng_digest() -> dict[str, int | str]:
    return digest_value({"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state()})


def _train_step(trainer: RayPPOTrainer, batch: TrainingInputBatch) -> None:
    status = trainer.train_critic_and_policy(batch)
    if status["policy_update_steps"] != 1:
        raise AssertionError(f"Expected one policy optimizer update, got {status['policy_update_steps']}")


def _reference(trainer: RayPPOTrainer, root: str, fingerprint: str) -> None:
    record_path = f"{root}/{REFERENCE_NAME}"
    latest_path = f"{root}/checkpoints/latest_ckpt_global_step.txt"
    if io.exists(record_path) or io.exists(latest_path):
        raise RuntimeError("Reference run requires a fresh checkpoint and record prefix")
    batch = _batch()
    batch_digest = digest_value({"data": dict(batch), "metadata": batch.metadata})
    batch.metadata["global_step"] = 0
    _train_step(trainer, batch)
    trainer.global_step = 1
    before_save_rng = _rank_digests(trainer, "parity_digest_rng")
    driver_before_save = _driver_rng_digest()
    trainer.save_checkpoints()
    after_save_rng = _rank_digests(trainer, "parity_digest_rng")
    if before_save_rng != after_save_rng:
        raise AssertionError("Checkpoint save changed worker RNG before the uninterrupted step")
    if driver_before_save != _driver_rng_digest():
        raise AssertionError("Checkpoint save changed driver RNG before the uninterrupted step")
    checkpoint = f"{root}/checkpoints/global_step_1"
    payload = resolve_checkpoint_payload(checkpoint, verify_files=True)
    pre = _rank_digests(trainer, "parity_digest_state")
    if any(pre[rank]["optimizer"]["tensor_count"] == 0 for rank in pre):
        raise AssertionError("Checkpoint source has an uninitialized optimizer shard")
    driver_pre = _driver_rng_digest()
    replay_batch = _batch()
    _train_step(trainer, replay_batch)
    post = _rank_digests(trainer, "parity_digest_state")
    driver_post = _driver_rng_digest()
    if all(pre[rank]["model"] == post[rank]["model"] for rank in pre):
        raise AssertionError("The uninterrupted optimizer step did not change any model shard")
    record = {
        "batch": batch_digest,
        "checkpoint": checkpoint,
        "config_sha256": fingerprint,
        "driver_pre": driver_pre,
        "driver_post": driver_post,
        "geometry": list(EXPECTED_GEOMETRY),
        "payload": payload,
        "pre": pre,
        "post": post,
        "source_step": 1,
        "world_size": WORLD_SIZE,
    }
    with io.open_file(record_path, "w") as output:
        json.dump(record, output, sort_keys=True)
    print(f"SNOWBALL_CHECKPOINT_PARITY_REFERENCE_OK ranks={WORLD_SIZE} record={record_path}", flush=True)


def _resume(trainer: RayPPOTrainer, root: str, fingerprint: str) -> None:
    record_path = f"{root}/{REFERENCE_NAME}"
    with io.open_file(record_path, "r") as source:
        record = json.load(source)
    batch = _batch()
    if record["config_sha256"] != fingerprint or record["batch"] != digest_value(
        {"data": dict(batch), "metadata": batch.metadata}
    ):
        raise AssertionError("Reference and resume configuration or fixed batch differ")
    if record["world_size"] != WORLD_SIZE or record["geometry"] != list(EXPECTED_GEOMETRY):
        raise AssertionError("Reference and resume policy geometry differ")
    loaded_step, payload = trainer.load_checkpoints()
    checkpoint = f"{root}/checkpoints/global_step_1"
    if loaded_step != 1 or record["checkpoint"] != checkpoint or payload != record["payload"]:
        raise AssertionError("Resume did not load the reference run's own committed checkpoint")
    trainer.global_step = loaded_step
    if record["driver_pre"] != _driver_rng_digest():
        raise AssertionError("Driver Python/NumPy/Torch RNG differs immediately after load")
    _compare_rank_digests(record["pre"], _rank_digests(trainer, "parity_digest_state"), "after load")
    _train_step(trainer, batch)
    if record["driver_post"] != _driver_rng_digest():
        raise AssertionError("Driver Python/NumPy/Torch RNG differs after resumed step")
    _compare_rank_digests(record["post"], _rank_digests(trainer, "parity_digest_state"), "after replay")
    print(f"SNOWBALL_CHECKPOINT_PARITY_RESUME_OK ranks={WORLD_SIZE} record={record_path}", flush=True)


def run(cfg: DictConfig) -> None:
    """Execute one phase in a dedicated 32-H100 Iris Ray gang."""
    root = _root()
    phase = _phase()
    cfg, fingerprint = _prepare_config(cfg, root)
    cfg.trainer.resume_mode = "latest" if phase is Phase.RESUME else "none"
    ray.init(address="auto")
    trainer = _trainer(cfg)
    built = False
    try:
        policy = ray.remote(num_gpus=1)(SnowballParityPolicyWorker)
        trainer.build_models(policy, CriticWorker, RefWorker)
        built = True
        if phase is Phase.REFERENCE:
            _reference(trainer, root, fingerprint)
        else:
            _resume(trainer, root, fingerprint)
    finally:
        if built:
            trainer.cleanup_ray_actors()
        ray.shutdown()
