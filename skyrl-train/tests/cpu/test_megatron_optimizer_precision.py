"""Configuration gates; GPU optimizer state and numerical checks remain separate."""

from copy import deepcopy

from omegaconf import OmegaConf
import pytest
import torch

from skyrl_train.distributed.megatron.optimizer import megatron_optimizer_kwargs
from skyrl_train.entrypoints.probe_megatron_optimizer_precision import (
    MATRIX_WIDTH,
    expected_reduced_gradient,
    gradient_for_step,
    tensor_inventory,
    validate_moment_inventory,
)


def test_native_optimizer_default_arguments_preserve_existing_contract():
    config = {"optimizer": "AdamW", "lr": 1e-6}
    native = {
        "overlap_cpu_optimizer_d2h_h2d": False,
        "use_precision_aware_optimizer": False,
        "optimizer_cpu_offload": False,
        "optimizer_offload_fraction": 0.0,
    }
    assert megatron_optimizer_kwargs(config, native) == {
        "optimizer": "adam",
        "lr": 1e-6,
        "min_lr": 0.0,
        "clip_grad": 1.0,
        "weight_decay": 0.01,
        "bf16": True,
        "params_dtype": torch.bfloat16,
        "use_distributed_optimizer": True,
        **native,
    }


@pytest.mark.parametrize(
    "name", ["main_params_dtype", "main_grads_dtype", "exp_avg_dtype", "exp_avg_sq_dtype", "params_dtype"]
)
@pytest.mark.parametrize("value", ["float32", "torch.float32", torch.float32])
def test_explicit_dtype_strings_and_native_dtypes_produce_native_dtype_arguments(name, value):
    kwargs = {name: value}
    before = deepcopy(kwargs)
    result = megatron_optimizer_kwargs({"lr": 1e-6}, kwargs)
    assert result[name] is torch.float32
    assert kwargs == before


@pytest.mark.parametrize("second", ["float32", "bfloat16"])
def test_hydra_bf16_moment_controls_preserve_master_gradient_and_remainder_contract(second):
    kwargs = OmegaConf.create(
        {
            "use_precision_aware_optimizer": True,
            "main_params_dtype": "float32",
            "main_grads_dtype": "float32",
            "exp_avg_dtype": "bfloat16",
            "exp_avg_sq_dtype": second,
            "store_param_remainders": False,
            "optimizer_cuda_graph": False,
        }
    )
    result = megatron_optimizer_kwargs({"lr": 1e-6}, kwargs)
    assert result["exp_avg_dtype"] is torch.bfloat16
    assert result["exp_avg_sq_dtype"] is (torch.float32 if second == "float32" else torch.bfloat16)
    assert result["main_params_dtype"] is torch.float32 and result["main_grads_dtype"] is torch.float32
    assert result["store_param_remainders"] is False and result["optimizer_cuda_graph"] is False
    assert kwargs.exp_avg_dtype == "bfloat16"


@pytest.mark.parametrize(
    "name,value",
    [
        ("exp_avg_dtype", "torch.not_a_dtype"),
        ("exp_avg_dtype", "float64"),
        ("exp_avg_dtype", None),
        ("exp_avg_sq_dtype", 16),
        ("main_params_dtype", "bfloat16"),
        ("main_grads_dtype", "bfloat16"),
        ("params_dtype", "int16"),
    ],
)
def test_unsupported_or_ineffective_dtype_settings_fail_before_native_construction(name, value):
    with pytest.raises(ValueError):
        megatron_optimizer_kwargs({"lr": 1e-6}, {"use_precision_aware_optimizer": True, name: value})


@pytest.mark.parametrize("name", ["main_params_dtype", "exp_avg_dtype", "exp_avg_sq_dtype"])
def test_non_fp32_state_cannot_silently_enter_the_native_path(name):
    dtype = "float16" if name == "main_params_dtype" else "bfloat16"
    with pytest.raises(ValueError, match="precision_aware"):
        megatron_optimizer_kwargs({"lr": 1e-6}, {name: dtype})


@pytest.mark.parametrize(
    "overrides",
    [
        {"optimizer": "sgd"},
        {"use_distributed_optimizer": False},
        {"optimizer_cuda_graph": True},
        {"optimizer_cuda_graph": True, "store_param_remainders": False, "exp_avg_dtype": "bfloat16"},
    ],
)
def test_precision_aware_path_rejects_incompatible_optimizer_and_graph_packages(overrides):
    with pytest.raises(ValueError):
        megatron_optimizer_kwargs({"lr": 1e-6}, {"use_precision_aware_optimizer": True, **overrides})


def test_fp32_precision_aware_control_does_not_inherit_bf16_moments_or_force_remainders():
    declared = {
        "use_precision_aware_optimizer": True,
        "main_params_dtype": "float32",
        "main_grads_dtype": "float32",
        "exp_avg_dtype": "float32",
        "exp_avg_sq_dtype": "float32",
        "store_param_remainders": False,
    }
    result = megatron_optimizer_kwargs({"lr": 1e-6}, declared)
    assert all(
        result[name] is torch.float32
        for name in ("main_params_dtype", "main_grads_dtype", "exp_avg_dtype", "exp_avg_sq_dtype")
    )
    assert result["store_param_remainders"] is False


def test_fp16_master_requires_explicitly_disabling_fp32_remainder_storage():
    kwargs = {"use_precision_aware_optimizer": True, "main_params_dtype": "float16"}
    with pytest.raises(ValueError, match="remainders"):
        megatron_optimizer_kwargs({"lr": 1e-6}, kwargs)
    assert (
        megatron_optimizer_kwargs({"lr": 1e-6}, {**kwargs, "store_param_remainders": False})["main_params_dtype"]
        is torch.float16
    )


def test_state_inventory_distinguishes_logical_shards_from_retained_storage():
    buffer = torch.zeros(128, dtype=torch.float32)
    first_moment = torch.zeros(32, dtype=torch.bfloat16)
    second_moment = torch.zeros(32, dtype=torch.float32)
    inventory = tensor_inventory(
        [
            ("model", buffer),
            ("master_shard", buffer[32:64]),
            ("first_moment", first_moment),
            ("second_moment", second_moment),
        ]
    )
    assert inventory["logical_bytes"] == (128 + 32) * 4 + 32 * (2 + 4)
    assert inventory["unique_retained_storage_bytes"] == 128 * 4 + 32 * (2 + 4)
    rows = {row["name"]: row for row in inventory["tensors"]}
    assert rows["master_shard"]["storage_id"] == rows["model"]["storage_id"]
    assert rows["master_shard"]["storage_offset"] == 32
    assert rows["first_moment"]["dtype"] == "torch.bfloat16"


def moment_inventory():
    return tensor_inventory(
        [
            (f"optimizer.{part}.0.{parameter}.{suffix}", torch.zeros(4, dtype=dtype))
            for part in range(2)
            for parameter in range(2)
            for suffix, dtype in (
                ("parameter", torch.bfloat16),
                ("exp_avg", torch.bfloat16),
                ("exp_avg_sq", torch.float32),
            )
        ]
    )


def test_moment_inventory_requires_each_parameter_and_component_to_have_both_moments():
    validate_moment_inventory(moment_inventory(), 2, "bfloat16", "float32")


@pytest.mark.parametrize("suffix", ["exp_avg", "exp_avg_sq"])
@pytest.mark.parametrize("part", [0, 1])
def test_missing_moment_cannot_pass_when_other_parameters_have_valid_moments(part, suffix):
    inventory = moment_inventory()
    inventory["tensors"] = [row for row in inventory["tensors"] if row["name"] != f"optimizer.{part}.0.1.{suffix}"]
    with pytest.raises(AssertionError, match="Missing persistent moment"):
        validate_moment_inventory(inventory, 2, "bfloat16", "float32")


@pytest.mark.parametrize(
    "replacement",
    [
        {"name": "optimizer.1.0.1.scale.exp_avg"},
        {"dtype": "torch.float32"},
        {"shape": [2, 2]},
        {"logical_bytes": 0},
    ],
)
def test_scales_wrong_dtype_wrong_shape_and_empty_moments_do_not_qualify(replacement):
    inventory = moment_inventory()
    row = next(row for row in inventory["tensors"] if row["name"] == "optimizer.1.0.1.exp_avg")
    row.update(replacement)
    with pytest.raises(AssertionError, match="moment"):
        validate_moment_inventory(inventory, 2, "bfloat16", "float32")


def test_empty_component_and_duplicate_inventory_names_fail_closed():
    inventory = moment_inventory()
    with pytest.raises(AssertionError, match="Missing optimizer parameters"):
        validate_moment_inventory(inventory, 3, "bfloat16", "float32")
    inventory["tensors"].append(inventory["tensors"][0])
    with pytest.raises(AssertionError, match="Duplicate"):
        validate_moment_inventory(inventory, 2, "bfloat16", "float32")


@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize("step", [0, 1, 2])
def test_analytic_reduced_gradient_matches_actual_bf16_leaf_autograd(step, world_size):
    device = torch.device("cpu")
    actual_gradients = []
    for rank in range(world_size):
        weight = torch.nn.Parameter(torch.ones(MATRIX_WIDTH, MATRIX_WIDTH, dtype=torch.bfloat16))
        (weight.float() * gradient_for_step(step, device, rank)).sum().backward()
        actual_gradients.append(weight.grad.float())
    expected = expected_reduced_gradient(step, device, world_size)
    torch.testing.assert_close(expected, torch.stack(actual_gradients).mean(0), rtol=0, atol=0)
    if world_size == 2 and step:
        assert not torch.equal(expected, actual_gradients[0])
        assert not torch.equal(expected, actual_gradients[1])
