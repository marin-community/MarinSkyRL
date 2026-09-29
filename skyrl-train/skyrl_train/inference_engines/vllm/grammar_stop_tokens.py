"""Keep XGrammar's token mask consistent with its per-request stop tokens."""

from typing import Protocol

import torch

MASK_WORD_BITS = 32


class GrammarMatcher(Protocol):
    @property
    def stop_token_ids(self) -> list[int]: ...

    def is_completed(self) -> bool: ...

    def fill_next_token_bitmask(self, bitmask: torch.Tensor, index: int) -> bool: ...


class GrammarMask(Protocol):
    matcher: GrammarMatcher


def fill_xgrammar_bitmask(grammar: GrammarMask, bitmask: torch.Tensor, idx: int) -> None:
    """Exclude stop tokens that the matcher rejects before grammar completion."""
    matcher = grammar.matcher
    matcher.fill_next_token_bitmask(bitmask, idx)
    if matcher.is_completed():
        return
    # The compiler's cached string masks can include a token that a request
    # overrides as EOS. The matcher rejects it, so sampling must exclude it too.
    for token_id in matcher.stop_token_ids:
        bit = token_id % MASK_WORD_BITS
        # Bit 31 needs a positive signed-int32 mask; its Python complement
        # would otherwise overflow when PyTorch converts it to int32.
        keep_mask = (1 << bit) - 1 if bit == MASK_WORD_BITS - 1 else ~(1 << bit)
        bitmask[idx, token_id // MASK_WORD_BITS] &= keep_mask


def register_xgrammar_stop_token_mask() -> None:
    """Install the stop-token mask in every vLLM engine process via its plugin loader."""
    # vLLM is an optional GPU dependency; this entry point is called by vLLM only.
    from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar

    XgrammarGrammar.fill_bitmask = fill_xgrammar_bitmask
