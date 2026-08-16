import sys
import types

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from skyrl_train.models.layers.moe_checkpoint import moe_recompute_context_fn

try:
    from skyrl_train.models.layers.moe import MoE
except ModuleNotFoundError as error:
    if error.name != "torchtitan":
        raise
    torchtitan = types.ModuleType("torchtitan")
    torchtitan_distributed = types.ModuleType("torchtitan.distributed")
    torchtitan_ep = types.ModuleType("torchtitan.distributed.expert_parallel")
    torchtitan_ep.expert_parallel = lambda function: function
    sys.modules["torchtitan"] = torchtitan
    sys.modules["torchtitan.distributed"] = torchtitan_distributed
    sys.modules["torchtitan.distributed.expert_parallel"] = torchtitan_ep
    from skyrl_train.models.layers.moe import MoE

    del sys.modules["torchtitan.distributed.expert_parallel"]
    del sys.modules["torchtitan.distributed"]
    del sys.modules["torchtitan"]

from skyrl_train.model_wrapper import HFModelWrapper


class _FlippingGate(nn.Module):
    """Model a near-tied router whose winner changes during recomputation."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        first_wins = inputs.new_tensor([1.0, 0.999])
        second_wins = inputs.new_tensor([0.999, 1.0])
        logits = first_wins if self.calls == 1 else second_wins
        return logits.expand(inputs.shape[0], -1)


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
        num_experts=2,
        top_k=1,
        route_norm=True,
    )
    moe.router.gate = _FlippingGate()
    with torch.no_grad():
        moe.experts.w1.fill_(1.0)
        moe.experts.w2.fill_(1.0)
        moe.experts.w3.fill_(1.0)

    inputs = torch.ones(1, 2, 2, requires_grad=True)
    checkpoint_kwargs = _grouped_moe_checkpoint_kwargs()
    assert checkpoint_kwargs["context_fn"] is moe_recompute_context_fn
    output = checkpoint(moe, inputs, **checkpoint_kwargs)
    output.sum().backward()

    assert moe.router.gate.calls == 2
    assert torch.count_nonzero(moe.experts.w2.grad[0]) > 0
    assert torch.count_nonzero(moe.experts.w2.grad[1]) == 0
