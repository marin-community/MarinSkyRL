# SPDX-FileCopyrightText: 2026 NovaSkyAI
# SPDX-License-Identifier: Apache-2.0

"""Cheap pre-allocation gates for the narrow Snowball workload."""

import copy

import pytest
from omegaconf import OmegaConf
from skyrl_train.learner import UnsupportedLearnerConfiguration
from skyrl_train.learners.levanter_config import LevanterSnowballRuntimeConfig

from tests.cpu.util import example_dummy_config


def _valid_config():
    cfg = copy.deepcopy(example_dummy_config())
    updates = {
        "trainer.max_steps": 4,
        "trainer.train_batch_size": 2,
        "trainer.policy_mini_batch_size": 2,
        "trainer.micro_train_batch_size_per_gpu": 1,
        "trainer.micro_forward_batch_size_per_gpu": 1,
        "trainer.placement.policy_num_nodes": 1,
        "trainer.placement.policy_num_gpus_per_node": 1,
        "trainer.placement.colocate_all": False,
        "trainer.algorithm.use_kl_loss": False,
        "trainer.algorithm.use_kl_in_reward": False,
        "trainer.policy.optimizer_config.lr": 1e-5,
        "trainer.policy.optimizer_config.max_grad_norm": 0.5,
        "trainer.policy.optimizer_config.optimizer_kwargs": {"eps": 1e-8},
        "generator.weight_sync_backend": "gloo",
    }
    for path, value in updates.items():
        OmegaConf.update(cfg, path, value, force_add=True)
    return cfg


def test_supported_config_lowers_without_importing_the_concrete_learner():
    runtime = LevanterSnowballRuntimeConfig.from_msrl(_valid_config())

    assert runtime.training_gpus == 1
    assert runtime.training_nodes == 1
    assert runtime.training_gpus_per_node == 1
    assert runtime.train_batch_size == 2
    assert runtime.publication_backend == "gloo"
    assert runtime.inference_world_size == 4


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        ("trainer.algorithm.use_kl_loss", True, "no KL"),
        ("trainer.use_sample_packing", True, "use_sample_packing=false"),
        ("trainer.policy.fsdp_config.expert_model_parallel_size", 2, "expert parallel size 1"),
        ("generator.weight_sync_backend", "nccl", "weight_sync_backend=gloo"),
        ("trainer.micro_train_batch_size_per_gpu", 2, "micro_train_batch_size_per_gpu=1"),
        ("trainer.micro_forward_batch_size_per_gpu", 2, "micro_forward_batch_size_per_gpu=1"),
        ("trainer.algorithm.eps_clip_low", 0.1, "eps_clip_low=0.2"),
        ("trainer.algorithm.eps_clip_high", 0.3, "eps_clip_high=0.2"),
        ("trainer.algorithm.grpo_norm_by_std", False, "grpo_norm_by_std=true"),
        ("trainer.algorithm.advantage_batch_normalize", True, "advantage_batch_normalize=false"),
        ("trainer.restore_dataloader_state", False, "restore_dataloader_state=true"),
        ("trainer.policy.model.lora.rank", 8, "model.lora.rank=0"),
        ("trainer.policy.optimizer_config.lr", 2e-5, "optimizer_config.lr=1e-5"),
        ("trainer.policy.optimizer_config.adam_betas", [0.8, 0.99], "optimizer_config.adam_betas"),
        ("trainer.policy.optimizer_config.optimizer_kwargs", {"eps": 1e-6}, "optimizer_kwargs.eps=1e-8"),
        ("trainer.policy.optimizer_config.weight_decay", 0.0, "optimizer_config.weight_decay=0.01"),
        ("trainer.policy.optimizer_config.max_grad_norm", 1.0, "optimizer_config.max_grad_norm=0.5"),
        ("generator.inference_engine_pipeline_parallel_size", 2, "pipeline parallel size 1"),
        ("generator.sampling_params.temperature", 0.7, "sampling_params.temperature=1.0"),
    ],
)
def test_unsupported_semantics_fail_in_the_head_process(path, value, match):
    cfg = _valid_config()
    OmegaConf.update(cfg, path, value)

    with pytest.raises(UnsupportedLearnerConfiguration, match=match):
        LevanterSnowballRuntimeConfig.from_msrl(cfg)


def test_stale_resolved_grpo_group_size_fails_in_the_head_process():
    cfg = _valid_config()
    cfg.trainer.algorithm.resolved_group_advantage = {
        "kind": "exact_physical",
        "physical_group_size": 2,
        "minimum_group_size": None,
    }
    cfg.generator.n_samples_per_prompt = 3

    with pytest.raises(UnsupportedLearnerConfiguration, match="resolved GRPO physical group size"):
        LevanterSnowballRuntimeConfig.from_msrl(cfg)


def test_runtime_train_batch_counts_generated_trajectories():
    cfg = _valid_config()
    cfg.generator.n_samples_per_prompt = 2
    cfg.trainer.algorithm.resolved_group_advantage.physical_group_size = 2

    runtime = LevanterSnowballRuntimeConfig.from_msrl(cfg)

    assert runtime.train_batch_size == 4


def test_multi_host_runtime_counts_all_learner_gpus():
    cfg = _valid_config()
    cfg.trainer.placement.policy_num_nodes = 2
    cfg.trainer.placement.policy_num_gpus_per_node = 2
    cfg.trainer.train_batch_size = 2
    cfg.trainer.policy_mini_batch_size = 2
    cfg.generator.n_samples_per_prompt = 2
    cfg.trainer.algorithm.resolved_group_advantage.physical_group_size = 2

    runtime = LevanterSnowballRuntimeConfig.from_msrl(cfg)

    assert runtime.training_nodes == 2
    assert runtime.training_gpus_per_node == 2
    assert runtime.training_gpus == 4
    assert runtime.train_batch_size == 4
