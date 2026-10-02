import gc

import torch

from skyrl_train.models.grug_handoffs import clear_hand_offs, hand_off, take_hand_off


def _megatron_viewless(view: torch.Tensor) -> torch.Tensor:
    """What Megatron's ``make_viewless_tensor`` gives the transformer block for a view: a new tensor on its storage."""
    out = torch.empty((1,), dtype=view.dtype, device=view.device)
    out.data = view.data
    return out


def test_hand_off_reaches_the_norm_through_the_viewless_copy_of_a_view():
    clear_hand_offs()
    # The embedding gated norm's output under the vLLM numerics: a kernel's [rows, hidden] output viewed as [S, B, hidden].
    output = torch.arange(24.0).reshape(6, 4).view(3, 2, 4)
    hand_off(output, "embedding product")
    block_input = _megatron_viewless(output)

    assert block_input is not output
    assert take_hand_off(block_input) == "embedding product"
    # The norm took it; nothing is left for another reader.
    assert take_hand_off(output) is None


def test_hand_off_reaches_only_its_own_tensor():
    clear_hand_offs()
    storage = torch.arange(24.0)
    first, second = storage[:12].view(3, 4), storage[12:].view(3, 4)
    hand_off(first, "first")

    # Same storage, another offset: not what formed this tensor.
    assert take_hand_off(second) is None
    # A freed tensor's entry never reaches a new tensor that reuses its id.
    del first, storage
    gc.collect()
    assert take_hand_off(torch.zeros(3, 4)) is None
