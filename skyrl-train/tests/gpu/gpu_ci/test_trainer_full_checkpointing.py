"""
Integration test for full trainer checkpointing functionality.

This test validates that the RayPPOTrainer can save and restore ALL training state,
ensuring that training can resume exactly where it left off.

Run with:
uv run --group dev --extra vllm --extra megatron pytest tests/gpu/gpu_ci/test_trainer_full_checkpointing.py
"""

import asyncio
from types import SimpleNamespace

import ray
import pytest
import hydra
import os
from uuid import uuid4
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset
from unittest.mock import AsyncMock, MagicMock
from transformers import AutoTokenizer

from skyrl_train.rollouts.context import TrainingContextState
from skyrl_train.rollouts.loader import PromptLoaderState
from skyrl_train.checkpoint_generation import resolve_checkpoint_payload
from skyrl_train.distributed.megatron.checkpoint_metadata import remote_checkpoint_metadata
from skyrl_train.io import io
from skyrl_train.utils.tracking import Tracking
from skyrl_train.trainer import RayPPOTrainer
from tests.gpu.utils import import_worker, ray_init_for_tests
from skyrl_train.entrypoints.main_base import config_dir

MODEL_NAME = "Qwen/Qwen3-0.6B"
NUM_GPUS = 2
ROLLOUT_STATE = TrainingContextState(
    loader=PromptLoaderState(order={"epoch": 0, "position": 2}, retries=[]), ready=[], object_store_root=None
)


class DummyDataset(Dataset):
    """Minimal dataset for testing"""

    def __init__(self, size=10):
        self.data = [([{"role": "user", "content": f"Question {i}"}], None) for i in range(size)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        return batch


def get_test_trainer_config(strategy: str, optimizer_checkpoint_sharding_type: str | None = None) -> DictConfig:
    """Create minimal trainer config for testing"""
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")

    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.trainer.strategy = strategy

    # Use minimal settings for faster testing
    cfg.trainer.placement.policy_num_gpus_per_node = NUM_GPUS
    cfg.trainer.placement.critic_num_gpus_per_node = NUM_GPUS
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.critic_num_nodes = 1
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.placement.colocate_all = False  # Disable colocation for simpler testing
    cfg.trainer.train_batch_size = NUM_GPUS
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.update_epochs_per_batch = 1
    cfg.trainer.epochs = 1
    cfg.trainer.logger = "console"
    cfg.generator.n_samples_per_prompt = 1
    cfg.generator.num_inference_engines = NUM_GPUS // 2
    cfg.generator.inference_engine_tensor_parallel_size = 2

    # Megatron-specific
    if strategy == "megatron":
        OmegaConf.update(cfg, "trainer.algorithm.max_seq_len", 128, force_add=True)
        cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 2
        cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 2
        cfg.trainer.policy.megatron_config.optimizer_checkpoint_sharding_type = optimizer_checkpoint_sharding_type
        cfg.trainer.placement.policy_num_gpus_per_node = 4
        cfg.trainer.train_batch_size = 4
        cfg.trainer.policy_mini_batch_size = 4

    prefix = os.environ.get("MARIN_TEMP_PREFIX", os.environ.get("MARIN_PREFIX", ""))
    if not prefix.startswith("s3://"):
        raise ValueError("Run the Megatron checkpoint test on Iris with CoreWeave object storage configured")
    cfg.trainer.ckpt_path = os.path.join(prefix, "tests", "trainer-checkpoint", uuid4().hex)
    cfg.trainer.export_path = cfg.trainer.ckpt_path

    # Enable checkpointing with correct config names
    cfg.trainer.ckpt_interval = 1  # Save every step
    cfg.trainer.resume_mode = "none"  # Initially false, will be set to True for resume

    return cfg


def create_minimal_trainer(cfg: DictConfig):
    """Create a minimal trainer setup for testing"""
    # Create minimal tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Create dummy dataset
    train_dataset = DummyDataset(size=4)  # Small dataset for quick testing

    mock_trajectory_runner = MagicMock()

    # Create tracker
    tracker = Tracking(
        project_name=cfg.trainer.project_name,
        experiment_name=cfg.trainer.run_name,
        backends=cfg.trainer.logger,
        config=cfg,
    )

    # Create trainer (no inference engine needed for checkpointing tests)
    trainer = RayPPOTrainer(
        cfg=cfg,
        tracker=tracker,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=None,
        inference_engine_client=None,
        trajectory_runner=mock_trajectory_runner,
        context=SimpleNamespace(
            config=SimpleNamespace(max_staleness_steps=0, batch_size=cfg.trainer.train_batch_size),
            state_dict=AsyncMock(return_value=ROLLOUT_STATE),
        ),
    )

    return trainer


def saved_optimizer_format(checkpoint_dir: str) -> str:
    from megatron.core import dist_checkpointing
    from skyrl_train.distributed.megatron.megatron_strategy import _saved_optimizer_sharding_type

    payload = resolve_checkpoint_payload(checkpoint_dir, verify_files=True)
    with remote_checkpoint_metadata(os.path.join(payload, "policy")) as metadata_dir:
        common_state = dist_checkpointing.load_common_state_dict(metadata_dir)
    return _saved_optimizer_sharding_type(common_state)


@pytest.mark.parametrize(
    ("initial_sharding_type, resumed_sharding_type"),
    [
        ("fully_reshardable", "dp_reshardable"),
        ("dp_reshardable", "dp_reshardable"),
    ],
)
def test_trainer_full_checkpointing(ray_init_fixture, initial_sharding_type, resumed_sharding_type):
    from tests.gpu.test_megatron_worker import get_test_training_batch

    cfg = get_test_trainer_config("megatron", initial_sharding_type)
    workers = tuple(import_worker("megatron", role) for role in ("policy", "critic", "ref"))
    trainers = []
    try:
        trainer1 = create_minimal_trainer(cfg)
        trainers.append(trainer1)
        trainer1.build_models(*workers)
        batch = get_test_training_batch(batch_size=4)
        ray.get(trainer1.policy_model.async_run_ray_method("mesh", "ppo_train", batch))
        trainer1.global_step = 2
        asyncio.run(trainer1.save_checkpoints())
        checkpoint_dir = os.path.join(cfg.trainer.ckpt_path, "global_step_2")
        payload_dir = resolve_checkpoint_payload(checkpoint_dir, verify_files=True)
        latest_ckpt_file = os.path.join(cfg.trainer.ckpt_path, "latest_ckpt_global_step.txt")
        assert io.read_bytes(latest_ckpt_file) == b"2"
        assert saved_optimizer_format(checkpoint_dir) == initial_sharding_type
        trainer1.cleanup_ray_actors()
        ray.shutdown()
        ray_init_for_tests()

        cfg_resume = get_test_trainer_config("megatron", resumed_sharding_type)
        cfg_resume.trainer.resume_mode = "from_path"
        cfg_resume.trainer.resume_path = checkpoint_dir
        cfg_resume.trainer.export_path = cfg.trainer.export_path
        cfg_resume.trainer.ckpt_path = cfg.trainer.ckpt_path
        trainer2 = create_minimal_trainer(cfg_resume)
        trainers.append(trainer2)
        trainer2.build_models(*workers)
        loaded_global_step, loaded_checkpoint_dir = trainer2.load_checkpoints()
        assert loaded_global_step == 2
        assert loaded_checkpoint_dir == payload_dir
        assert trainer2._restored_rollout_state == ROLLOUT_STATE
        ray.get(trainer2.policy_model.async_run_ray_method("mesh", "ppo_train", batch))
        trainer2.global_step = 3
        asyncio.run(trainer2.save_checkpoints())
        assert saved_optimizer_format(os.path.join(cfg.trainer.ckpt_path, "global_step_3")) == resumed_sharding_type
        assert io.read_bytes(latest_ckpt_file) == b"3"
    finally:
        for trainer in trainers:
            trainer.cleanup_ray_actors()
        if io.exists(cfg.trainer.ckpt_path):
            io.remove(cfg.trainer.ckpt_path)
