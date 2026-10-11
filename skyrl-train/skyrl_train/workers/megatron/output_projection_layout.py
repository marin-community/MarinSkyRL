"""Project unsharded GPT output in its consumer layout without duplicating vocabulary logits."""

import torch
from torch import nn


def _batch_major_input(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return (inputs[0].transpose(0, 1).contiguous(), *inputs[1:])


def _sequence_major_output(
    _module: nn.Module,
    _inputs: tuple[torch.Tensor, ...],
    output: tuple[torch.Tensor, torch.Tensor | None],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    logits, bias = output
    return logits.transpose(0, 1), bias


def use_batch_major_output_projection(projection: nn.Module) -> None:
    """Make GPT's final batch-major conversion a view for TP1, non-sequence-parallel output layers.

    Moving the smaller hidden-state tensor before projection avoids a second full-vocabulary
    allocation in GPTModel._postprocess. Hooks retain the module and its checkpoint parameter names.
    """
    projection.register_forward_pre_hook(_batch_major_input)
    projection.register_forward_hook(_sequence_major_output)
