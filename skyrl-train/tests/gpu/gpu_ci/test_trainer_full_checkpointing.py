"""
Integration test for full trainer checkpointing functionality.

This test validates that the RayPPOTrainer can save and restore ALL training state,
ensuring that training can resume exactly where it left off.

Run with:
For FSDP and DeepSpeed, run:
uv run --isolated --group dev --extra deepspeed --extra vllm pytest tests/gpu/gpu_ci/test_trainer_full_checkpointing.py -m "not megatron"

For Megatron, run:
uv run --isolated --group dev --extra vllm --extra megatron pytest tests/gpu/gpu_ci/test_trainer_full_checkpointing.py -m "megatron"
"""

import ray
import pytest
import hydra
import hashlib
import pickle
import torch
import os
import shutil
import tempfile
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset
from unittest.mock import MagicMock
from transformers import AutoTokenizer

from skyrl_train.utils.tracking import Tracking
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.checkpoint_generation import resolve_checkpoint_payload
from skyrl_train.distributed.dispatch import concatenate_outputs_after_mesh_dispatch
from tests.gpu.utils import import_worker, ray_init_for_tests
from skyrl_train.entrypoints.main_base import config_dir

MODEL_NAME = "Qwen/Qwen3-0.6B"
NUM_GPUS = 2


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


def get_test_trainer_config(
    strategy: str, fsdp2_cpu_offload: bool = False, optimizer_checkpoint_sharding_type: str | None = None
) -> DictConfig:
    """Create minimal trainer config for testing"""
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")

    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.trainer.critic.model.path = MODEL_NAME  # Enable critic for testing
    cfg.trainer.strategy = strategy
    if strategy == "fsdp2":
        cfg.trainer.policy.fsdp_config.cpu_offload = fsdp2_cpu_offload

    # Use minimal settings for faster testing
    cfg.trainer.placement.policy_num_gpus_per_node = NUM_GPUS
    cfg.trainer.placement.critic_num_gpus_per_node = NUM_GPUS
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.critic_num_nodes = 1
    cfg.trainer.algorithm.use_kl_loss = (
        False  # disable ref model so we just have policy and critic (NUM_GPUS total GPUs)
    )
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
        # Disable critic for megatron
        cfg.trainer.critic.model.path = ""

    # Use temporary directories
    cfg.trainer.export_path = tempfile.mkdtemp(prefix="trainer_ckpt_test_")
    cfg.trainer.ckpt_path = cfg.trainer.export_path

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
    )

    return trainer


def saved_optimizer_format(checkpoint_dir: str) -> str:
    # Megatron is optional for the other strategies in this module.
    from megatron.core import dist_checkpointing
    from skyrl_train.distributed.megatron.megatron_strategy import _saved_optimizer_sharding_type

    common_state = dist_checkpointing.load_common_state_dict(
        os.path.join(resolve_checkpoint_payload(checkpoint_dir), "policy")
    )
    return _saved_optimizer_sharding_type(common_state)


def megatron_policy_logprobs(trainer: RayPPOTrainer) -> torch.Tensor:
    """Probe the policy on a fixed batch without changing its training state."""
    from tests.gpu.test_megatron_worker import get_test_training_batch

    probe_batch = get_test_training_batch(batch_size=4)
    probe_results = ray.get(trainer.policy_model.async_run_ray_method("mesh", "forward", data=probe_batch))
    logprobs = concatenate_outputs_after_mesh_dispatch(trainer.policy_model.actor_infos, probe_results)["output"]
    assert torch.isfinite(logprobs).all()
    return logprobs.detach().cpu()


def megatron_next_step_logprobs(trainer: RayPPOTrainer) -> torch.Tensor:
    """Run the same post-save optimizer step and probe the resulting policy."""
    from tests.gpu.test_megatron_worker import get_test_training_batch

    train_batch = get_test_training_batch(batch_size=4)
    train_batch.metadata["global_step"] = 3
    train_results = ray.get(trainer.policy_model.async_run_ray_method("mesh", "ppo_train", train_batch))
    assert all(result.metadata["train_status"]["policy_update_steps"] == 1 for result in train_results)
    return megatron_policy_logprobs(trainer)


@pytest.mark.megatron
def test_megatron_full_checkpoint_restores_per_rank_rng_and_cuda_tracker(ray_init_fixture, tmp_path):
    """A saved rank must recover both ordinary RNG and Megatron's separate CUDA tracker."""
    from skyrl_train.workers.megatron.megatron_worker import MegatronPolicyWorkerBase

    class RngPolicyWorker(MegatronPolicyWorkerBase):
        def rng_fingerprint(self, advance: bool = False):
            from megatron.core import tensor_parallel

            tracker = tensor_parallel.get_cuda_rng_tracker()
            names = sorted(tracker.get_states())
            assert names, "Megatron CUDA RNG tracker was not initialized"
            if advance:
                with tracker.fork(names[0]):
                    torch.rand(self._rank + 1, device="cuda")
                torch.rand(self._rank + 1, device="cuda")
            generic = self.strategy.get_rng_state()
            return {
                "rank": self._rank,
                "generic": hashlib.sha256(pickle.dumps(generic)).hexdigest(),
                "tracker": {
                    name: hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()
                    for name, state in tracker.get_states().items()
                },
            }

    cfg = get_test_trainer_config("megatron", optimizer_checkpoint_sharding_type="dp_reshardable")
    cfg.trainer.export_path = str(tmp_path)
    cfg.trainer.ckpt_path = str(tmp_path)
    trainer = create_minimal_trainer(cfg)
    trainer.build_models(
        ray.remote(num_gpus=1)(RngPolicyWorker), import_worker("megatron", "critic"), import_worker("megatron", "ref")
    )

    def fingerprints(advance: bool):
        results = ray.get(trainer.policy_model.async_run_ray_method("pass_through", "rng_fingerprint", advance))
        return {result["rank"]: result for result in results}

    expected = fingerprints(advance=True)
    trainer.global_step = 1
    trainer.save_checkpoints()
    changed = fingerprints(advance=True)
    assert changed != expected, "The test must advance the tracked CUDA RNG after saving"
    checkpoint_dir = os.path.join(resolve_checkpoint_payload(str(tmp_path / "global_step_1")), "policy")
    ray.get(trainer.policy_model.async_run_ray_method("pass_through", "load_checkpoint", checkpoint_dir))
    assert fingerprints(advance=False) == expected


@pytest.mark.parametrize(
    ("strategy, fsdp2_cpu_offload, initial_sharding_type, resumed_sharding_type"),
    [
        ("deepspeed", False, None, None),
        ("fsdp", False, None, None),
        ("fsdp2", False, None, None),
        ("fsdp2", True, None, None),
        pytest.param("megatron", False, "fully_reshardable", "dp_reshardable", marks=pytest.mark.megatron),
        pytest.param("megatron", False, "dp_reshardable", "dp_reshardable", marks=pytest.mark.megatron),
    ],
)
def test_trainer_full_checkpointing(
    ray_init_fixture, strategy, fsdp2_cpu_offload, initial_sharding_type, resumed_sharding_type
):
    """
    Test full trainer checkpointing by:
    1. Creating trainer and setting it up
    2. Saving checkpoint
    3. Capturing training state
    4. Destroying trainer
    5. Creating new trainer with resume enabled
    6. Loading checkpoint
    7. Verifying all state matches
    8. Continuing training to ensure it works
    """
    cfg = get_test_trainer_config(strategy, fsdp2_cpu_offload, initial_sharding_type)

    checkpoint_dir = None
    try:
        # ============= PHASE 1: Initial Training and Save =============
        print("Phase 1: Initial training and checkpoint save")

        trainer1 = create_minimal_trainer(cfg)

        # Get worker classes
        PolicyWorker = import_worker(strategy, "policy")
        CriticWorker = import_worker(strategy, "critic")
        RefWorker = import_worker(strategy, "ref")
        if strategy == "megatron":
            from skyrl_train.workers.megatron.megatron_worker import MegatronPolicyWorkerBase

            class FingerprintPolicyWorker(MegatronPolicyWorkerBase):
                def checkpoint_parameter_fingerprints(self):
                    module = self.actor_module[0]
                    tensors = {
                        **dict(module.named_parameters()),
                        **{f"buffer:{name}": tensor for name, tensor in module.named_buffers()},
                    }
                    return {
                        "rank": self._rank,
                        "hashes": {
                            name: hashlib.sha256(
                                tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
                            ).hexdigest()
                            for name, tensor in tensors.items()
                        },
                    }

            PolicyWorker = ray.remote(num_gpus=1)(FingerprintPolicyWorker)

        # Build models
        trainer1.build_models(PolicyWorker, CriticWorker, RefWorker)
        initial_logprobs = None
        initial_parameter_fingerprints = None
        if strategy == "megatron":
            initial_logprobs = megatron_policy_logprobs(trainer1)
            initial_parameter_fingerprints = ray.get(
                trainer1.policy_model.async_run_ray_method("pass_through", "checkpoint_parameter_fingerprints")
            )
            # A real optimizer step initializes Adam moments, which the checkpoint must preserve.
            # Keep this Megatron-only fixture out of non-Megatron test collection.
            from tests.gpu.test_megatron_worker import get_test_training_batch

            batch = get_test_training_batch(batch_size=4)
            batch.metadata["global_step"] = 1
            ray.get(trainer1.policy_model.async_run_ray_method("mesh", "ppo_train", batch))

        # Set initial global step as if 2 steps were completed
        trainer1.global_step = 2

        pre_save_logprobs = megatron_policy_logprobs(trainer1) if strategy == "megatron" else None

        # Save checkpoint
        trainer1.save_checkpoints()

        # Capture state before teardown
        saved_global_step = trainer1.global_step
        checkpoint_dir = os.path.join(cfg.trainer.export_path, f"global_step_{trainer1.global_step}")
        payload_dir = resolve_checkpoint_payload(checkpoint_dir, verify_files=True)

        # Verify checkpoint structure was created
        expected_files = [
            os.path.join(payload_dir, "policy"),
            os.path.join(payload_dir, "trainer_state.pt"),
            os.path.join(payload_dir, "data.pt"),
        ]
        # Only expect critic dir for non-megatron strategies
        if strategy != "megatron":
            expected_files.append(os.path.join(payload_dir, "critic"))
        for expected_file in expected_files:
            assert os.path.exists(expected_file), f"Expected checkpoint file/dir not found: {expected_file}"
        if strategy == "megatron":
            assert saved_optimizer_format(checkpoint_dir) == initial_sharding_type

        # Verify atomic tracking file
        latest_ckpt_file = os.path.join(cfg.trainer.ckpt_path, "latest_ckpt_global_step.txt")
        assert os.path.exists(latest_ckpt_file)
        with open(latest_ckpt_file, "r") as f:
            latest_step = int(f.read())
        assert latest_step == trainer1.global_step, "Atomic tracking file has incorrect step after first save"

        # Verify trainer state content
        print("Verifying checkpoint content...")
        loaded_trainer_state = torch.load(
            os.path.join(payload_dir, "trainer_state.pt"), map_location="cpu", weights_only=False
        )

        # Check key configuration values are preserved
        assert loaded_trainer_state["config"]["trainer"]["train_batch_size"] == cfg.trainer.train_batch_size, (
            "train_batch_size not preserved in checkpoint"
        )
        assert loaded_trainer_state["config"]["trainer"]["strategy"] == strategy, "strategy not preserved in checkpoint"
        assert loaded_trainer_state["global_step"] == saved_global_step, "global_step not preserved in checkpoint"

        # A second save alone does not exercise restored optimizer moments.
        expected_pre_step_logprobs = None
        expected_next_step_logprobs = None
        expected_parameter_fingerprints = None
        if strategy == "megatron":
            assert pre_save_logprobs is not None
            expected_pre_step_logprobs = megatron_policy_logprobs(trainer1)
            torch.testing.assert_close(expected_pre_step_logprobs, pre_save_logprobs, rtol=1e-3, atol=1e-3)
            repeated_pre_step_logprobs = megatron_policy_logprobs(trainer1)
            torch.testing.assert_close(repeated_pre_step_logprobs, expected_pre_step_logprobs, rtol=1e-3, atol=1e-3)
            expected_parameter_fingerprints = ray.get(
                trainer1.policy_model.async_run_ray_method("pass_through", "checkpoint_parameter_fingerprints")
            )
            expected_next_step_logprobs = megatron_next_step_logprobs(trainer1)
            # A same-process reload separates checkpoint-load effects from fresh-worker effects.
            ray.get(
                trainer1.policy_model.async_run_ray_method(
                    "pass_through", "load_checkpoint", os.path.join(payload_dir, "policy")
                )
            )
            reloaded_same_worker_logprobs = megatron_policy_logprobs(trainer1)
            torch.testing.assert_close(reloaded_same_worker_logprobs, expected_pre_step_logprobs, rtol=1e-3, atol=1e-3)
            replayed_same_worker_logprobs = megatron_next_step_logprobs(trainer1)
            torch.testing.assert_close(replayed_same_worker_logprobs, expected_next_step_logprobs, rtol=1e-3, atol=1e-3)

        # Cleanup first trainer
        del trainer1
        ray.shutdown()

        # ============= PHASE 2: Resume from Checkpoint =============
        print("Phase 2: Resume from checkpoint")
        ray_init_for_tests()
        # Create new config with resume enabled
        cfg_resume = get_test_trainer_config(strategy, fsdp2_cpu_offload, resumed_sharding_type)
        cfg_resume.trainer.resume_mode = "from_path"  # Enable resume
        cfg_resume.trainer.resume_path = checkpoint_dir  # Set resume path
        cfg_resume.trainer.export_path = cfg.trainer.export_path  # Use same export path
        cfg_resume.trainer.ckpt_path = cfg.trainer.ckpt_path

        trainer2 = create_minimal_trainer(cfg_resume)

        # Build models again
        trainer2.build_models(PolicyWorker, CriticWorker, RefWorker)
        if strategy == "megatron":
            assert initial_logprobs is not None and initial_parameter_fingerprints is not None
            fresh_logprobs = megatron_policy_logprobs(trainer2)
            fresh_parameter_fingerprints = ray.get(
                trainer2.policy_model.async_run_ray_method("pass_through", "checkpoint_parameter_fingerprints")
            )
            initial_by_rank = {result["rank"]: result["hashes"] for result in initial_parameter_fingerprints}
            fresh_by_rank = {result["rank"]: result["hashes"] for result in fresh_parameter_fingerprints}
            fresh_diff = (fresh_logprobs.float() - initial_logprobs.float()).abs()
            fresh_mismatches = ~torch.isclose(fresh_logprobs, initial_logprobs, rtol=1e-3, atol=1e-3)
            print(
                "Fresh-worker HF-load control: "
                f"model_tensors_exact={initial_by_rank == fresh_by_rank} "
                f"output_mismatches={fresh_mismatches.sum().item()}/{fresh_mismatches.numel()} "
                f"max_abs={fresh_diff.max().item():.6f} mean_abs={fresh_diff.mean().item():.6f}"
            )

        # Load checkpoints
        loaded_global_step, loaded_checkpoint_dir = trainer2.load_checkpoints()
        assert loaded_global_step == saved_global_step, (
            f"Expected global_step={saved_global_step}, got {loaded_global_step}"
        )
        assert loaded_checkpoint_dir == payload_dir, "Checkpoint payload path mismatch"

        # ============= PHASE 3: Continue Training =============
        print("Phase 3: Resumed optimizer step and second checkpoint save")

        if strategy == "megatron":
            assert expected_parameter_fingerprints is not None
            actual_parameter_fingerprints = ray.get(
                trainer2.policy_model.async_run_ray_method("pass_through", "checkpoint_parameter_fingerprints")
            )
            expected_by_rank = {result["rank"]: result["hashes"] for result in expected_parameter_fingerprints}
            actual_by_rank = {result["rank"]: result["hashes"] for result in actual_parameter_fingerprints}
            assert expected_by_rank.keys() == actual_by_rank.keys()
            mismatches = {
                rank: [
                    name for name, digest in expected_by_rank[rank].items() if actual_by_rank[rank].get(name) != digest
                ]
                for rank in expected_by_rank
            }
            assert not any(mismatches.values()), f"Model tensors differ after resume: {mismatches}"
            assert expected_pre_step_logprobs is not None
            actual_pre_step_logprobs = megatron_policy_logprobs(trainer2)
            torch.testing.assert_close(actual_pre_step_logprobs, expected_pre_step_logprobs, rtol=1e-3, atol=1e-3)
            assert expected_next_step_logprobs is not None
            actual_next_step_logprobs = megatron_next_step_logprobs(trainer2)
            torch.testing.assert_close(actual_next_step_logprobs, expected_next_step_logprobs, rtol=1e-3, atol=1e-3)

        # Try to save another checkpoint to test cleanup logic
        trainer2.global_step = 3
        trainer2.save_checkpoints()

        next_checkpoint_dir = os.path.join(cfg.trainer.export_path, f"global_step_{trainer2.global_step}")
        assert os.path.exists(resolve_checkpoint_payload(next_checkpoint_dir, verify_files=True)), (
            "Could not save checkpoint after resume"
        )
        if strategy == "megatron":
            assert saved_optimizer_format(next_checkpoint_dir) == resumed_sharding_type

        # Verify atomic tracking file is updated
        latest_ckpt_file = os.path.join(cfg.trainer.ckpt_path, "latest_ckpt_global_step.txt")
        assert os.path.exists(latest_ckpt_file)
        with open(latest_ckpt_file, "r") as f:
            latest_step = int(f.read())
        assert latest_step == trainer2.global_step, "Atomic tracking file was not updated after second save"

    finally:
        if checkpoint_dir and os.path.exists(os.path.dirname(checkpoint_dir)):
            print(f"Cleaning up checkpoint directory: {os.path.dirname(checkpoint_dir)}")
            shutil.rmtree(os.path.dirname(checkpoint_dir))
