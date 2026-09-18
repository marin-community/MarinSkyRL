"""The expert-block transport is refused at configuration time unless every precondition holds."""

import subprocess
import sys

import pytest

from marinskyrl.inference_placement import validate_expert_block_transport
from skyrl_train.utils.utils import validate_cfg
from tests.cpu.util import example_dummy_config


def expert_block_config():
    cfg = example_dummy_config()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.expert_model_parallel_size = 8
    cfg.generator.update(
        backend="vllm",
        async_engine=True,
        run_engines_locally=True,
        weight_sync_backend="nccl",
        weight_sync_transport="expert_block",
        inference_engine_node_local=True,
        inference_engine_tensor_parallel_size=1,
        inference_engine_pipeline_parallel_size=1,
        inference_engine_data_parallel_size=8,
        inference_engine_expert_parallel_size=8,
    )
    return cfg


def test_a_complete_expert_block_configuration_is_accepted():
    validate_expert_block_transport(expert_block_config())


def test_unequal_expert_parallel_degrees_are_accepted():
    cfg = expert_block_config()
    cfg.trainer.policy.megatron_config.expert_model_parallel_size = 16
    validate_expert_block_transport(cfg)


def test_a_tensor_parallel_trainer_is_accepted():
    cfg = expert_block_config()
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 2
    validate_expert_block_transport(cfg)


def test_the_default_transport_needs_nothing():
    cfg = example_dummy_config()
    assert cfg.generator.weight_sync_transport == "broadcast"
    validate_expert_block_transport(cfg)


@pytest.mark.parametrize(
    "path,value,message",
    [
        ("trainer.strategy", "fsdp2", "megatron strategy"),
        ("trainer.policy.megatron_config.expert_tensor_parallel_size", 2, "expert_tensor_parallel_size 1"),
        ("trainer.policy.megatron_config.expert_model_parallel_size", 0, "must be positive"),
        ("generator.backend", "sglang", "local async vLLM"),
        ("generator.async_engine", False, "local async vLLM"),
        ("trainer.placement.colocate_all", True, "not be colocated"),
        ("generator.weight_sync_backend", "gloo", "must be nccl"),
        ("generator.inference_engine_node_local", False, "inference_engine_node_local must be true"),
        ("generator.inference_engine_tensor_parallel_size", 2, "TP=1"),
        ("generator.expert_block_sync.timeout_seconds", 0, "must be positive"),
        ("generator.weight_sync_transport", "shard", "must be one of"),
    ],
)
def test_each_missing_precondition_is_named(path, value, message):
    cfg = expert_block_config()
    node = cfg
    *parents, leaf = path.split(".")
    for key in parents:
        node = node[key]
    node[leaf] = value
    with pytest.raises(ValueError, match=message):
        validate_expert_block_transport(cfg)


def test_validate_cfg_runs_the_transport_check():
    cfg = example_dummy_config()
    cfg.trainer.train_batch_size = 4
    cfg.trainer.policy_mini_batch_size = 4
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.generator.weight_sync_transport = "expert_block"
    with pytest.raises(ValueError, match="weight_sync_transport=expert_block requires"):
        validate_cfg(cfg)


def test_the_model_package_imports_before_the_trainer_utilities():
    # The frozen-runtime bootstrap imports the Grug model first; that chain reaches
    # skyrl_train.utils, which must not pull the weight_sync package back into the models.
    subprocess.run(
        [sys.executable, "-c", "from skyrl_train.models.grug_moe import GRUG_MOE_ARCHITECTURE"],
        check=True,
        capture_output=True,
        text=True,
    )
