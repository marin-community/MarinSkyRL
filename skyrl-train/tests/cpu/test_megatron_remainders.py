"""Exact remainder decoding and independently scoped qualification gates."""

import torch

from skyrl_train.entrypoints.probe_megatron_optimizer_precision import reconstruct_remainder_master
from skyrl_train.entrypoints.probe_megatron_remainders import compare_arm_states, tensor_bytes_equal


def test_decode_native_remainders_recovers_float_bits_including_rounding_and_signed_zero():
    master = torch.tensor([1.00001, -1.00001, 3.14159, -0.0, 1.00390625], dtype=torch.float32)
    parameters = master.to(torch.bfloat16)
    # TE's kernel rounds exact half ties up, unlike the ordinary BF16 even cast.
    parameters[-1] = 1.0078125
    lower_words = master.view(torch.int16).reshape(-1, 2)[:, 0].clone()
    actual = reconstruct_remainder_master(parameters, lower_words)
    assert torch.equal(actual.view(torch.int32), master.view(torch.int32))


def observed_states():
    results = {
        name: {"optimizer_inventory": {"bytes_per_owned_parameter": size}, "checkpoint_continuation_exact": True}
        for name, size in {"native_fp32": 12, "aware_fp32": 12, "bf16_both": 8,
                           "fp32_remainders": 10, "bf16_remainders": 6}.items()
    }
    master = torch.tensor([1.00001, -0.0], dtype=torch.float32)
    model = master.to(torch.bfloat16).view(torch.int16)
    snapshots = {name: (master.clone(), model.clone()) for name in ("aware_fp32", "fp32_remainders")}
    return results, snapshots


def test_memory_gate_applies_combined_saving_while_exact_arm_saves_only_master_bytes():
    results, snapshots = observed_states()
    result = compare_arm_states(results, snapshots)
    assert result["savings_bytes_per_owned_parameter"]["fp32_remainders"] == 2
    assert result["combined_remainders_memory_pass"] is True
    assert result["bf16_moments_memory_pass"] is True
    results["bf16_remainders"]["optimizer_inventory"]["bytes_per_owned_parameter"] = 7
    assert compare_arm_states(results, snapshots)["combined_remainders_memory_pass"] is False


def test_exact_gate_rejects_single_master_bit_change_and_model_rounding_change():
    results, snapshots = observed_states()
    snapshots["fp32_remainders"][0].view(torch.int32)[0] ^= 1
    snapshots["fp32_remainders"][1][0] += 1
    result = compare_arm_states(results, snapshots)
    assert result["fp32_remainders_master_bits_equal_at_update10"] is False
    assert result["fp32_remainders_model_bits_equal_at_update10"] is False
    assert result["fp32_remainders_master_mismatched_elements"] == 1
    assert result["fp32_remainders_model_mismatched_elements"] == 1


def test_checkpoint_exactness_rejects_signed_zero_difference():
    original = torch.tensor([0.0, 1.0], dtype=torch.float32)
    restored = torch.tensor([-0.0, 1.0], dtype=torch.float32)
    assert not tensor_bytes_equal(original, restored)
    assert tensor_bytes_equal(original, original.clone())
