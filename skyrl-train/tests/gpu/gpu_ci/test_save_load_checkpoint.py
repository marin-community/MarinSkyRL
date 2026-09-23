"""
For FSDP and DeepSpeed, run:
uv run --isolated --group dev --extra deepspeed -- pytest tests/gpu/gpu_ci/test_save_load_checkpoint.py -m "not megatron"

For Megatron, run:
uv run --isolated --group dev --extra vllm --extra megatron -- pytest tests/gpu/gpu_ci/test_save_load_checkpoint.py -m "megatron"
"""

import ray
import pytest
import hydra
import torch
import os
import shutil
import json
import pickle
import fsspec
from datetime import timedelta
from pathlib import Path
from omegaconf import DictConfig
from transformers import AutoTokenizer
from torch.distributed import checkpoint
import torch.distributed as dist
import torch.multiprocessing as mp

from skyrl_train.utils.utils import print_mem
from tests.gpu.utils import init_worker_with_type, make_dummy_experience, get_model_logits_from_actor, validate_cfg
from skyrl_train.entrypoints.main_base import config_dir

MODEL_NAME = "Qwen/Qwen3-0.6B"
CKPT_PATH = "$HOME/ckpts/test/"
NUM_GPUS = 4


@pytest.mark.megatron
def test_megatron_plan_cache_reuses_stable_schema_and_refreshes_changed_dtype(tmp_path):
    from skyrl_train.distributed.megatron.direct_checkpoint import (
        _SchemaGuardedMCoreSavePlanner,
        invalidate_checkpoint_plan_cache,
    )
    from skyrl_train.io.torch_distributed_checkpoint import StreamingFsspecWriter

    cache_key = f"test-{tmp_path.name}"

    class ObservedPlanner(_SchemaGuardedMCoreSavePlanner):
        def create_local_plan(self):
            plan = super().create_local_plan()
            self.local_plan_usable = plan.usable
            return plan

    try:
        for step, dtype in ((1, torch.float32), (2, torch.float32), (3, torch.float16)):
            state = {"tensor": torch.arange(6, dtype=dtype).reshape(2, 3) + step}
            path = tmp_path / f"step-{step}"
            planner = ObservedPlanner(
                cache_key=cache_key,
                dedup_replicated_tensors=False,
                flatten_state_dict=False,
                flatten_sharded_tensors=False,
            )
            writer = StreamingFsspecWriter(str(path), filesystem=fsspec.filesystem("file"))
            checkpoint.save(state, storage_writer=writer, planner=planner)

            assert planner.local_plan_usable == (step != 2)
            with (path / ".metadata").open("rb") as source:
                metadata = pickle.load(source)
            assert metadata.state_dict_metadata["tensor"].properties.dtype == dtype

            restored = {"tensor": torch.zeros_like(state["tensor"])}
            checkpoint.load(restored, checkpoint_id=str(path))
            torch.testing.assert_close(restored["tensor"], state["tensor"], atol=0, rtol=0)
    finally:
        invalidate_checkpoint_plan_cache(cache_key)


def _run_megatron_plan_cache_rank(rank: int, checkpoint_root: str, rendezvous_file: str) -> None:
    from skyrl_train.distributed.megatron.direct_checkpoint import (
        _SchemaGuardedMCoreSavePlanner,
        invalidate_checkpoint_plan_cache,
    )
    from skyrl_train.io.torch_distributed_checkpoint import StreamingFsspecWriter

    class ObservedPlanner(_SchemaGuardedMCoreSavePlanner):
        def create_local_plan(self):
            plan = super().create_local_plan()
            self.local_plan_usable = plan.usable
            return plan

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous_file}", rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    cache_key = f"two-rank-{checkpoint_root}"
    try:
        for step in (1, 2, 3, 4):
            dtype = torch.float16 if rank == 1 and step == 3 else torch.float32
            state = {f"rank_{rank}": torch.arange(6, dtype=dtype).reshape(2, 3) + step + rank}
            path = os.path.join(checkpoint_root, f"step-{step}")
            planner = ObservedPlanner(
                cache_key=cache_key,
                dedup_replicated_tensors=False,
                flatten_state_dict=False,
                flatten_sharded_tensors=False,
            )
            writer = StreamingFsspecWriter(path, filesystem=fsspec.filesystem("file"))
            checkpoint.save(state, storage_writer=writer, planner=planner)

            assert planner.local_plan_usable == (step == 1 or (rank == 1 and step >= 3))
            if rank == 0:
                with open(os.path.join(path, ".metadata"), "rb") as source:
                    metadata = pickle.load(source)
                assert metadata.state_dict_metadata["rank_0"].properties.dtype == torch.float32
                assert metadata.state_dict_metadata["rank_1"].properties.dtype == (
                    torch.float16 if step == 3 else torch.float32
                )

            restored = {f"rank_{rank}": torch.zeros_like(state[f"rank_{rank}"])}
            checkpoint.load(restored, checkpoint_id=path)
            torch.testing.assert_close(restored[f"rank_{rank}"], state[f"rank_{rank}"], atol=0, rtol=0)
    finally:
        invalidate_checkpoint_plan_cache(cache_key)
        dist.destroy_process_group()


@pytest.mark.megatron
def test_megatron_plan_cache_refreshes_one_changed_rank_without_stale_metadata(tmp_path):
    mp.spawn(_run_megatron_plan_cache_rank, args=(str(tmp_path), str(tmp_path / "rendezvous")), nprocs=2)


@pytest.mark.megatron
def test_megatron_direct_save_failure_invalidates_every_plan_cache(monkeypatch):
    from torch.distributed.checkpoint.planner import SavePlanner
    from skyrl_train.distributed.megatron import direct_checkpoint

    cache_key = "failed-direct-save"
    caches = (
        SavePlanner._cached_save_plan,
        SavePlanner._cached_all_plans,
        SavePlanner._cached_global_plan,
        SavePlanner._cached_metadata,
        SavePlanner._cached_final_save_plan,
    )
    for cache in caches:
        cache[cache_key] = object()

    monkeypatch.setattr(direct_checkpoint.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(
        direct_checkpoint, "_replace_state_dict_keys_with_sharded_keys", lambda state, _: (state, {}, {})
    )
    monkeypatch.setattr(direct_checkpoint, "mcore_to_pyt_state_dict", lambda state, _: state)
    monkeypatch.setattr(direct_checkpoint, "get_s3_fs", lambda: object())
    monkeypatch.setattr(direct_checkpoint, "s3_refresh_if_expiring", lambda _: None)
    monkeypatch.setattr(direct_checkpoint, "StreamingFsspecWriter", lambda *args, **kwargs: object())

    def fail_after_planning(state, *, storage_writer, planner):
        assert isinstance(planner, direct_checkpoint._SchemaGuardedMCoreSavePlanner)
        raise RuntimeError("injected checkpoint write failure")

    monkeypatch.setattr(direct_checkpoint.checkpoint, "save", fail_after_planning)
    strategy = direct_checkpoint.DirectS3TorchDistSaveShardedStrategy(
        "s3://unit-test/global_step_2/policy", plan_cache_key=cache_key
    )
    with pytest.raises(RuntimeError, match="injected checkpoint write failure"):
        strategy.save({"tensor": torch.ones(1)}, Path("."))
    assert all(cache_key not in cache for cache in caches)


def run_one_training_step(
    actor_group,
    strategy,
    experience=None,
    global_step=None,
    local_step=None,
    accumulation_steps=None,
    megatron_batch=None,
):
    if strategy == "megatron":
        assert megatron_batch is not None, "Megatron requires a TrainingInputBatch for ppo_train"
        return ray.get(actor_group.async_run_ray_method("mesh", "ppo_train", megatron_batch))
    else:
        assert experience is not None, f"{strategy} requires an Experience for training_step"
        return ray.get(
            actor_group.async_run_ray_method(
                "pass_through", "training_step", experience, global_step, local_step, accumulation_steps
            )
        )


def get_test_actor_config(strategy: str, optimizer_checkpoint_sharding_type: str | None = None) -> DictConfig:
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")

    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.trainer.placement.policy_num_gpus_per_node = NUM_GPUS
    cfg.trainer.strategy = strategy
    if strategy == "megatron":
        cfg.trainer.policy.megatron_config.optimizer_checkpoint_sharding_type = optimizer_checkpoint_sharding_type

    cfg.trainer.ckpt_path = CKPT_PATH
    cfg.trainer.export_path = CKPT_PATH
    cfg.trainer.logger = "console"

    validate_cfg(cfg)

    return cfg


@pytest.mark.parametrize(
    ("strategy, optimizer_checkpoint_sharding_type"),
    [
        ("deepspeed", None),
        ("fsdp", None),
        ("fsdp2", None),
        pytest.param("megatron", "fully_reshardable", marks=pytest.mark.megatron),
        pytest.param("megatron", "dp_reshardable", marks=pytest.mark.megatron),
    ],
)
def test_save_load_checkpoint(ray_init_fixture, strategy, optimizer_checkpoint_sharding_type):
    """
    Test checkpointing logic by:
    1. Creating model and doing one training step
    2. Saving checkpoint
    3. Doing second training step and recording model logits
    4. Loading checkpoint
    5. Repeating second training step and comparing logits
    """
    cfg = get_test_actor_config(strategy, optimizer_checkpoint_sharding_type)

    try:
        actor_group = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        checkpoint_dir = None
        # Create dummy experiences for training steps
        dummy_experience_1 = make_dummy_experience()  # First training step
        dummy_experience_2 = make_dummy_experience()  # Second training step

        # Ensure the second experience is different from the first
        for i, seq in enumerate(dummy_experience_2.sequences):
            dummy_experience_2.sequences[i] = torch.randint(100, 200, seq.shape, device=seq.device)

        # For Megatron, build training batches and reuse the second one pre/post checkpoint resume
        if "megatron" in strategy:
            from tests.gpu.test_megatron_worker import get_test_training_batch

            dp_size = actor_group.actor_infos[0].rank.dp_size
            train_batch_1 = get_test_training_batch(dp_size if dp_size % NUM_GPUS == 0 else NUM_GPUS)
            train_batch_2 = get_test_training_batch(dp_size if dp_size % NUM_GPUS == 0 else NUM_GPUS)
        else:
            train_batch_1 = None
            train_batch_2 = None

        global_step, local_step, accumulation_steps = 0, 0, 1

        # Step 1: Do initial training step
        run_one_training_step(
            actor_group,
            strategy,
            experience=dummy_experience_1,
            global_step=global_step,
            local_step=local_step,
            accumulation_steps=accumulation_steps,
            megatron_batch=train_batch_1,
        )

        checkpoint_path = os.path.expandvars(os.path.join(cfg.trainer.ckpt_path, "global_step_1", "policy"))
        checkpoint_dir = os.path.expandvars(os.path.join(cfg.trainer.ckpt_path, "global_step_1"))  # Store for cleanup

        # Step 2: Save checkpoint
        ray.get(
            actor_group.async_run_ray_method(
                "pass_through", "save_checkpoint", ckpt_dir=checkpoint_path, tokenizer=tokenizer
            )
        )
        if strategy == "megatron":
            # This persisted format controls how the next process constructs its load template.
            from megatron.core import dist_checkpointing
            from skyrl_train.distributed.megatron.megatron_strategy import _saved_optimizer_sharding_type

            common_state = dist_checkpointing.load_common_state_dict(checkpoint_path)
            assert _saved_optimizer_sharding_type(common_state) == optimizer_checkpoint_sharding_type

        # Step 2.1: Make sure that offloading still works after saving checkpoint
        memory_after_saving = ray.get(actor_group.async_run_ray_method("pass_through", "get_cuda_memory"))[0]
        print_mem("memory after saving checkpoint", memory_after_saving)

        actor_group.offload_to_cpu()

        memory_after_offloading = ray.get(actor_group.async_run_ray_method("pass_through", "get_cuda_memory"))[0]
        print_mem("memory after offloading", memory_after_offloading)

        assert memory_after_offloading["allocated"] < memory_after_saving["allocated"], (
            f"Memory after offloading should be less than after saving: {memory_after_offloading} bytes < {memory_after_saving} bytes"
        )
        actor_group.backload_to_gpu()

        # check that relevant files are saved
        huggingface_dir = os.path.join(checkpoint_path, "huggingface")
        expected_files = ["config.json", "generation_config.json", "tokenizer.json"]
        for file in expected_files:
            assert os.path.exists(os.path.join(huggingface_dir, file)), (
                f"File {file} not found in huggingface directory"
            )
        if "fsdp" in strategy:
            fsdp_config_path = os.path.join(checkpoint_path, "fsdp_config.json")
            with open(fsdp_config_path, "r") as f:
                fsdp_config = json.load(f)
            assert fsdp_config["fsdp_strategy"] == strategy
            assert fsdp_config["world_size"] == NUM_GPUS

        # Step 3: Do second training step and record results
        run_one_training_step(
            actor_group,
            strategy,
            experience=dummy_experience_2,
            global_step=global_step + 1,
            local_step=local_step,
            accumulation_steps=accumulation_steps,
            megatron_batch=train_batch_2,
        )

        # Create test input for comparing model outputs
        dp_size = actor_group.actor_infos[0].rank.dp_size
        test_input = torch.randint(0, 1000, (dp_size, 20), device="cpu")  # batch_size=dp_size, seq_len=20
        attention_mask = torch.ones_like(test_input)

        # Step 4: Get logits after the second training step (this should be different from after checkpoint load)
        logits_after_second_training = get_model_logits_from_actor(actor_group, test_input, attention_mask)

        # Step 5: Load checkpoint via strategy's load_checkpoint method
        assert os.path.exists(checkpoint_path), f"Checkpoint directory {checkpoint_path} does not exist"
        ray.get(actor_group.async_run_ray_method("pass_through", "load_checkpoint", ckpt_dir=checkpoint_path))

        # Step 6: Now repeat the exact same second training step
        run_one_training_step(
            actor_group,
            strategy,
            experience=dummy_experience_2,
            global_step=global_step + 1,
            local_step=local_step,
            accumulation_steps=accumulation_steps,
            megatron_batch=train_batch_2,
        )

        # Get logits after loading checkpoint and repeating second training
        logits_after_reload_and_training = get_model_logits_from_actor(actor_group, test_input, attention_mask)

        # The logits should be exactly the same (checkpoint loading worked correctly)
        torch.testing.assert_close(logits_after_second_training, logits_after_reload_and_training, atol=0.0, rtol=0.0)

    finally:
        # Clean up checkpoint directory
        if checkpoint_dir and os.path.exists(checkpoint_dir):
            print(f"Removing checkpoint directory: {checkpoint_dir}")
            shutil.rmtree(checkpoint_dir)
