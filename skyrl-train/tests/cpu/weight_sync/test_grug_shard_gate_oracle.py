"""Mutations of every model category must fail the independent Grug byte oracle."""

import pytest
import torch

from tests.gpu.grug_shard_gate import assert_source_preserved, compare_receiver_parameters


PREFIX = "model.layers.0.mlp.experts"
NATIVE = PREFIX + ".routed_experts"


def oracle_fixture():
    hf = {
        PREFIX + ".gate_proj.weight": torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]]),
        PREFIX + ".up_proj.weight": torch.tensor([[[5.0, 6.0]], [[7.0, 8.0]]]),
        PREFIX + ".down_proj.weight": torch.tensor([[[9.0], [10.0]], [[11.0], [12.0]]]),
        "model.layers.0.shared_expert.up_proj.weight": torch.tensor([[13.0, 14.0]]),
        "model.layers.0.mlp.router.weight": torch.tensor([[0.3333333, 15.0]]),
        "model.layers.0.mlp.router.bias": torch.tensor([0.1234567, 0.25]),
        "model.norm.weight": torch.tensor([1.0, 2.0]),
    }
    native = {
        NATIVE + ".w13_weight": torch.tensor([[[3.0, 4.0], [7.0, 8.0]]], dtype=torch.bfloat16),
        NATIVE + ".w2_weight": torch.tensor([[[11.0], [12.0]]], dtype=torch.bfloat16),
        "model.layers.0.shared_expert.up_proj.weight": torch.tensor([[13.0, 14.0]], dtype=torch.bfloat16),
        "model.layers.0.mlp.router.weight": torch.tensor([[0.333984375, 15.0]], dtype=torch.float32),
        "model.layers.0.mlp.router.bias": torch.tensor([0.1234567, 0.25]),
        "model.norm.weight": torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
    }
    return hf, {"ep_rank": 1, "ep_size": 2, "expert_maps": {NATIVE: [-1, 0]}, "parameters": native}


def test_complete_native_oracle_preserves_router_precision_and_all_categories():
    hf, snapshot = oracle_fixture()
    assert compare_receiver_parameters(hf, snapshot, expected_ep=2, experts=2) == 36


@pytest.mark.parametrize(
    "parameter",
    [
        NATIVE + ".w13_weight",
        NATIVE + ".w2_weight",
        "model.layers.0.shared_expert.up_proj.weight",
        "model.layers.0.mlp.router.weight",
        "model.layers.0.mlp.router.bias",
        "model.norm.weight",
    ],
)
def test_one_raw_byte_corruption_is_rejected_for_every_parameter_category(parameter):
    hf, snapshot = oracle_fixture()
    snapshot["parameters"][parameter].view(torch.uint8).reshape(-1)[-1].bitwise_xor_(1)
    with pytest.raises(AssertionError):
        compare_receiver_parameters(hf, snapshot, expected_ep=2, experts=2)


@pytest.mark.parametrize("mutation", ["wrong_owner", "missing_parameter", "extra_parameter", "unequal_ep"])
def test_oracle_rejects_ownership_or_inventory_coverage_errors(mutation):
    hf, snapshot = oracle_fixture()
    if mutation == "wrong_owner":
        snapshot["expert_maps"][NATIVE] = [0, -1]
    elif mutation == "missing_parameter":
        del snapshot["parameters"]["model.norm.weight"]
    elif mutation == "extra_parameter":
        snapshot["parameters"]["unaccounted.weight"] = torch.ones(1)
    else:
        snapshot["ep_size"] = 1
    with pytest.raises(AssertionError):
        compare_receiver_parameters(hf, snapshot, expected_ep=2, experts=2)


def test_source_preservation_checks_nonroot_bytes_and_signed_zero():
    before = [{"expert": torch.tensor([1.0])}, {"expert": torch.tensor([0.0])}]
    after = [{"expert": torch.tensor([1.0])}, {"expert": torch.tensor([-0.0])}]
    with pytest.raises(AssertionError):
        assert_source_preserved(before, after)
    after[1]["expert"] = torch.tensor([0.0])
    assert assert_source_preserved(before, after) == 8
