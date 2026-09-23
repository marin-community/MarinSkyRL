"""Config validation rejects ``expert_block`` unless its requirements are met, and ``auto`` picks it only then."""

import json
import subprocess
import sys

import pytest

from marinskyrl.inference_placement import validate_expert_block_trainer, validate_expert_block_transport
from skyrl_train.entrypoints.main_base import BasePPOExp
from skyrl_train.utils.utils import resolve_weight_sync_transport, validate_cfg
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


def test_the_default_transport_validates_without_a_readable_model():
    cfg = example_dummy_config()
    cfg.trainer.train_batch_size = 4
    cfg.trainer.policy_mini_batch_size = 4
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.policy.model.path = "gs://bucket/no-such-model"
    # Must not raise: the default defers the choice to the entrypoint, so the driver's validation
    # needs neither the model config nor the trainer type.
    validate_cfg(cfg)


def write_model_config(directory, **overrides):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps({"model_type": "grug_moe", **overrides}))


def auto_config(tmp_path):
    cfg = expert_block_config()
    cfg.generator.weight_sync_transport = "auto"
    cfg.generator.engine_init_kwargs = {"moe_backend": "triton"}
    write_model_config(tmp_path / "model")
    cfg.trainer.policy.model.path = str(tmp_path / "model")
    return cfg


def test_auto_picks_expert_block_for_a_qualifying_run(tmp_path):
    cfg = auto_config(tmp_path)
    resolve_weight_sync_transport(cfg, uses_fully_async_trainer=True)
    assert cfg.generator.weight_sync_transport == "expert_block"


def test_auto_accepts_the_moe_backend_under_kernel_config(tmp_path):
    cfg = auto_config(tmp_path)
    cfg.generator.engine_init_kwargs = {"kernel_config": {"moe_backend": "triton"}}
    resolve_weight_sync_transport(cfg, uses_fully_async_trainer=True)
    assert cfg.generator.weight_sync_transport == "expert_block"


def fsdp_policy(cfg, tmp_path):
    cfg.trainer.strategy = "fsdp2"


def tensor_parallel_policy(cfg, tmp_path):
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 2


def engine_ep_differs_from_dp(cfg, tmp_path):
    cfg.generator.inference_engine_expert_parallel_size = 4


def colocated_engines(cfg, tmp_path):
    cfg.trainer.placement.colocate_all = True


def cutlass_moe_backend(cfg, tmp_path):
    cfg.generator.engine_init_kwargs = {"moe_backend": "flashinfer_cutlass"}


def unpinned_moe_backend(cfg, tmp_path):
    cfg.generator.engine_init_kwargs = {}


def quantised_engines(cfg, tmp_path):
    cfg.generator.engine_init_kwargs = {"moe_backend": "triton", "quantization": "fp8"}


def dense_model(cfg, tmp_path):
    write_model_config(tmp_path / "model", model_type="qwen3")


def quantised_model(cfg, tmp_path):
    write_model_config(tmp_path / "model", quantization_config={"quant_method": "fp8"})


def missing_model_config(cfg, tmp_path):
    cfg.trainer.policy.model.path = str(tmp_path / "absent")


def remote_model_config(cfg, tmp_path):
    cfg.trainer.policy.model.path = "gs://bucket/model"


@pytest.mark.parametrize(
    "unmet",
    [
        fsdp_policy,
        tensor_parallel_policy,
        engine_ep_differs_from_dp,
        colocated_engines,
        cutlass_moe_backend,
        unpinned_moe_backend,
        quantised_engines,
        dense_model,
        quantised_model,
        missing_model_config,
        remote_model_config,
    ],
)
def test_auto_falls_back_to_broadcast_when_a_requirement_is_unmet(tmp_path, unmet):
    cfg = auto_config(tmp_path)
    unmet(cfg, tmp_path)
    resolve_weight_sync_transport(cfg, uses_fully_async_trainer=True)
    assert cfg.generator.weight_sync_transport == "broadcast"


def test_auto_falls_back_to_broadcast_on_the_synchronous_entrypoint(tmp_path):
    cfg = auto_config(tmp_path)
    resolve_weight_sync_transport(cfg, uses_fully_async_trainer=False)
    assert cfg.generator.weight_sync_transport == "broadcast"


@pytest.mark.parametrize("explicit", ["broadcast", "expert_block"])
def test_explicit_transports_are_left_as_written(tmp_path, explicit):
    qualifying = auto_config(tmp_path)
    qualifying.generator.weight_sync_transport = explicit
    resolve_weight_sync_transport(qualifying, uses_fully_async_trainer=True)
    assert qualifying.generator.weight_sync_transport == explicit
    unmet = auto_config(tmp_path)
    unmet.generator.weight_sync_transport = explicit
    unmet.trainer.strategy = "fsdp2"
    resolve_weight_sync_transport(unmet, uses_fully_async_trainer=False)
    assert unmet.generator.weight_sync_transport == explicit


@pytest.mark.parametrize(
    "path,value,message",
    [
        ("trainer.strategy", "fsdp2", "megatron strategy"),
        ("trainer.policy.megatron_config.tensor_model_parallel_size", 2, "tensor_model_parallel_size 1"),
        ("trainer.policy.megatron_config.expert_tensor_parallel_size", 2, "expert_tensor_parallel_size 1"),
        ("trainer.policy.megatron_config.expert_model_parallel_size", 0, "must be positive"),
        ("generator.backend", "sglang", "non-colocated async vLLM"),
        ("generator.async_engine", False, "non-colocated async vLLM"),
        ("trainer.placement.colocate_all", True, "non-colocated async vLLM"),
        ("generator.weight_sync_backend", "gloo", "must be nccl"),
        ("generator.inference_engine_tensor_parallel_size", 2, "TP=1"),
        ("generator.inference_engine_expert_parallel_size", 4, "EP equal to DP"),
        ("generator.inference_engine_data_parallel_size", 1, "DP=1"),
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


def test_pipeline_parallel_engines_on_the_mp_backend_are_refused():
    cfg = expert_block_config()
    cfg.generator.inference_engine_pipeline_parallel_size = 2
    validate_expert_block_transport(cfg)
    cfg.generator.inference_engine_mp_backend = True
    with pytest.raises(ValueError, match="Ray executor"):
        validate_expert_block_transport(cfg)


def test_validate_cfg_runs_the_transport_check():
    cfg = example_dummy_config()
    cfg.trainer.train_batch_size = 4
    cfg.trainer.policy_mini_batch_size = 4
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.generator.weight_sync_transport = "expert_block"
    with pytest.raises(ValueError, match="weight_sync_transport=expert_block requires"):
        validate_cfg(cfg)


def test_only_an_entrypoint_running_the_fully_async_trainer_may_select_expert_block():
    cfg = expert_block_config()
    # Must not raise: the fully async trainer runs the transport.
    validate_expert_block_trainer(cfg, uses_fully_async_trainer=True)
    with pytest.raises(ValueError, match="FullyAsyncRayPPOTrainer"):
        validate_expert_block_trainer(cfg, uses_fully_async_trainer=False)
    # Must not raise: every trainer supports broadcast.
    validate_expert_block_trainer(example_dummy_config(), uses_fully_async_trainer=False)


def test_the_standard_entrypoint_refuses_expert_block_instead_of_syncing_by_broadcast():
    # BasePPOExp runs RayPPOTrainer, which ignores the option. The check is the first line of
    # trainer setup, so the test skips tokenizer and dataset loading.
    exp = object.__new__(BasePPOExp)
    exp.cfg = expert_block_config()
    with pytest.raises(ValueError, match="FullyAsyncRayPPOTrainer"):
        exp._setup_trainer()


def test_the_model_package_imports_before_the_trainer_utilities():
    # The frozen-runtime bootstrap imports the Grug model first. That import reaches
    # skyrl_train.utils and this validator, and must not be circular.
    subprocess.run(
        [sys.executable, "-c", "from skyrl_train.models.grug_moe import GRUG_MOE_ARCHITECTURE"],
        check=True,
        capture_output=True,
        text=True,
    )
