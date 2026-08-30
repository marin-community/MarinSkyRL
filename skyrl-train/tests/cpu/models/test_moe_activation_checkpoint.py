import torch
from torch import nn
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from skyrl_train.model_wrapper import HFModelWrapper
from skyrl_train.models.layers.moe_checkpoint import moe_recompute_context_fn
from tests.cpu.models.moe_test_imports import import_grouped_moe_module

MoE = import_grouped_moe_module().MoE


class _FlippingGate(nn.Module):
    """Model a near-tied router whose winner changes during recomputation."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.weight = nn.Parameter(torch.zeros(4, 2))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        first_wins = inputs.new_tensor([1.0, 0.999, 0.998, 0.997])
        second_wins = inputs.new_tensor([0.997, 0.998, 0.999, 1.0])
        logits = first_wins if self.calls == 1 else second_wins
        return inputs @ self.weight.t() + logits


class _CheckpointableModel(nn.Module):
    def gradient_checkpointing_enable(self, *, gradient_checkpointing_kwargs) -> None:
        self.checkpoint_kwargs = gradient_checkpointing_kwargs


def _grouped_moe_checkpoint_kwargs() -> dict:
    wrapper = HFModelWrapper.__new__(HFModelWrapper)
    nn.Module.__init__(wrapper)
    wrapper.moe_grouped_gemm = True
    wrapper.model = _CheckpointableModel()
    wrapper.gradient_checkpointing_enable({"use_reentrant": False})
    return wrapper.model.checkpoint_kwargs


def test_activation_checkpoint_recompute_reuses_forward_expert_routes() -> None:
    moe = MoE(
        dim=2,
        hidden_dim=2,
        num_experts=4,
        top_k=2,
        route_norm=True,
    )
    moe.router.gate = _FlippingGate()
    with torch.no_grad():
        moe.experts.w1.fill_(1.0)
        expert_scales = torch.arange(1, 5, dtype=moe.experts.w2.dtype).view(4, 1, 1)
        moe.experts.w2.copy_(expert_scales.expand_as(moe.experts.w2))
        moe.experts.w3.fill_(1.0)

    inputs = torch.ones(1, 2, 2, requires_grad=True)
    checkpoint_kwargs = _grouped_moe_checkpoint_kwargs()
    assert checkpoint_kwargs["context_fn"] is moe_recompute_context_fn
    # Grouped GEMM keeps the checkpointed graph live past the router. Disable
    # early stop to exercise the same full-tape comparison on CPU.
    with set_checkpoint_early_stop(False):
        output = checkpoint(moe, inputs, **checkpoint_kwargs)
    output.sum().backward()

    assert moe.router.gate.calls == 2
    assert torch.count_nonzero(moe.router.gate.weight.grad) > 0
    assert torch.count_nonzero(moe.experts.w2.grad[0]) > 0
    assert torch.count_nonzero(moe.experts.w2.grad[1]) > 0
    assert torch.count_nonzero(moe.experts.w2.grad[2:]) == 0
