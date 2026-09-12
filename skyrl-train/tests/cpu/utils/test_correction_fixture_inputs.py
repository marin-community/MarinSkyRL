"""Reproduce zero-gradient padded actions at the real Megatron scalar scatter boundary."""

import ast
from pathlib import Path

import pytest
import torch

from skyrl_train.training_batch import TrainingInputBatch
from tests.correction_fixture_inputs import left_pad_correction_batch, m2_fixture_diagnostics
from tests.cpu.utils.test_offpolicy_masks import mask_config, objective
from tests.offpolicy_mask_reference import regular_correction_reference_policy_loss


def scatter_token_values():
    # Execute the actual tensor function without importing its GPU-only Megatron dependencies.
    source = Path(__file__).parents[3] / "skyrl_train/distributed/megatron/megatron_utils.py"
    tree = ast.parse(source.read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "scatter_token_values"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["scatter_token_values"]


@pytest.mark.parametrize("logprob", [-2.0, -16.0, -64.0])
@pytest.mark.parametrize("repair", [False, True])
def test_actual_scatter_preserves_nonzero_m2_gradient_only_with_supported_actions(logprob, repair):
    # Lengths and padding from the native helper with the byte-pinned Qwen tokenizer.
    lengths, left_padding = [19, 24, 23, 15], [4, 0, 1, 6]
    sequences = torch.zeros((4, 31), dtype=torch.long)
    attention = torch.zeros_like(sequences)
    for row, (length, left) in enumerate(zip(lengths, left_padding, strict=True)):
        sequences[row, left : left + length] = torch.arange(1, length + 1)
        attention[row, left : left + length] = 1
    batch = TrainingInputBatch({"sequences": sequences, "attention_mask": attention})
    batch.metadata = {"response_length": 10}
    if repair:
        batch = left_pad_correction_batch(batch)
        for row, length in enumerate(lengths):
            valid = batch["attention_mask"][row].bool()
            assert batch["sequences"][row, valid].tolist() == list(range(1, length + 1))
            # MegatronPolicyWorker creates position IDs from this exact expression.
            positions = batch["attention_mask"][row].long().cumsum(-1) - 1
            assert positions[valid].tolist() == list(range(length))
            assert valid[-11:].all()

    compact = torch.full((4, 31), logprob, dtype=torch.float32, requires_grad=True)
    current = scatter_token_values()(compact, batch["attention_mask"], drop_last=True)[:, -10:]
    delta = current.new_tensor([-0.221, -0.220, -0.219, -0.218, -0.217, -0.216, -0.215, -0.214, -0.213, 0.05])
    old = current.detach() - delta
    diagnostics = m2_fixture_diagnostics(current.detach(), old, batch["attention_mask"])
    assert [item["supported_action_positions"] for item in diagnostics] == ([10] * 4 if repair else [2, 3, 3, 0])
    assert [item["retained_gradient_positions"] for item in diagnostics] == ([6] * 4 if repair else [0] * 4)
    config = mask_config(**{"m2_mask.enabled": True})
    for row in range(4):
        actual_current = current[row : row + 1]
        previous = old[row : row + 1]
        advantages, mask = -torch.ones_like(actual_current), torch.ones_like(actual_current)
        actual = objective(config, actual_current, previous, previous, advantages, mask)
        expected, _ = regular_correction_reference_policy_loss(
            actual_current, previous, advantages, config, mask, previous, mode="m2"
        )
        torch.testing.assert_close(actual.policy_loss, expected, atol=1e-6, rtol=0)
        gradient = torch.autograd.grad(actual.policy_loss, compact, retain_graph=True)[0]
        reference_gradient = torch.autograd.grad(expected, compact, retain_graph=True)[0]
        torch.testing.assert_close(gradient, reference_gradient, atol=1e-6, rtol=0)
        assert int(torch.count_nonzero(gradient)) == (6 if repair else 0)
        assert actual.metrics["m2_mask/masked_fraction"] == pytest.approx(0.4)
