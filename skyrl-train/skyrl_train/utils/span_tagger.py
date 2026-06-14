"""Response-token span tagger (Stage B / F4) for loop-behavior reward shaping.

Labels each *response* token with one of ``{OTHER, THINK, ACTION, EDIT}`` on the
**exact token-id layout** the TIS / training path uses — i.e. the same per-turn
segmentation produced by
``skyrl_train.generators.utils.get_response_ids_and_loss_mask_from_messages``.

Why this matters: Stages C (PBS test-delta) and D (think-token budget) write the
per-token shaping channel (``token_level_shaping``) onto specific token spans —
the edit-action tokens (C) and the ``<think>`` tokens (D). Those spans must align
1:1 with the *training* tokens, NOT a fresh re-tokenization, or the shaping lands
on the wrong positions. This tagger therefore re-walks the SAME loop that builds
``response_ids`` (encode each message with ``encode_messages_subset``, take the
assistant generated-token slice ``[prefix_len : last_eos+1]``) and tags within
that slice using delimiter tokens located in the EXACT id stream. No char→token
guessing: the segmentation is by token id, so the tags are exact by construction.

Stage B emits the tags as a no-op (the channel they index carries zeros); C/D
consume them. The tagger is pure / CPU-only and has no torch dependency.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from loguru import logger

from skyrl_train.generators.utils import (
    encode_messages_subset,
    get_generation_prompt_ids,
)

# Tag constants. OTHER=0 so an all-zeros tag vector == "everything is OTHER",
# which is the safe default (Stage C/D only act on non-OTHER spans).
SPAN_OTHER: int = 0
SPAN_THINK: int = 1
SPAN_ACTION: int = 2
SPAN_EDIT: int = 3

SPAN_TAG_NAMES: Dict[int, str] = {
    SPAN_OTHER: "other",
    SPAN_THINK: "think",
    SPAN_ACTION: "action",
    SPAN_EDIT: "edit",
}

# Markers used to decide ACTION vs EDIT within an assistant turn's non-think text.
# These are intentionally broad and content-based (mirrors reward_shaping's action
# payload extraction). The token-level boundary is taken from the <think> delimiter
# tokens; ACTION-vs-EDIT is a turn-level classification applied to the post-think
# generated span (sub-think granularity for EDIT is a Stage-C/D refinement, not B).
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
# An "edit" turn is one whose generated text writes file content: heredocs, file
# write tool calls, apply_patch / str_replace style payloads. Heuristic, refined
# in Stage C against the real test-result parser.
_EDIT_PATTERN = re.compile(
    r"(<<\s*['\"]?EOF|>\s*\S+\.\w+|apply_patch|str_replace|create_file|write_file|\bcat\s*>)",
    re.IGNORECASE,
)


def _encode_one(message: Dict, tokenizer, custom_chat_template: Optional[str]) -> List[int]:
    return encode_messages_subset([message], tokenizer, custom_chat_template)


def _generated_slice_bounds(
    cur_token_ids: List[int],
    generation_prompt_ids: List[int],
    eos_token_id: int,
):
    """Replicate get_response_ids_and_loss_mask_from_messages' assistant split.

    Returns (prefix_len, gen_start, gen_end) where the generated (loss_mask=1)
    span is cur_token_ids[gen_start:gen_end]; everything outside is OTHER.
    """
    prefix_len = len(generation_prompt_ids)
    if cur_token_ids[:prefix_len] != generation_prompt_ids:
        # Same fallback as the training path: treat the whole message as generated.
        prefix_len = 0
    if eos_token_id in cur_token_ids:
        last_eos = len(cur_token_ids) - 1 - cur_token_ids[::-1].index(eos_token_id)
        gen_end = last_eos + 1
    else:
        gen_end = len(cur_token_ids)
    return prefix_len, prefix_len, gen_end


def _tag_assistant_generated_span(
    generated_token_ids: List[int],
    tokenizer,
) -> List[int]:
    """Tag a single assistant turn's generated-token span.

    Strategy (exact, token-id based for the THINK boundary):
      - Decode the generated tokens once to locate <think>…</think> by string,
        then map the think character span back to a token span by decoding a
        growing prefix (monotonic, no re-tokenization of sub-strings).
      - Tokens inside <think>…</think> -> THINK.
      - Remaining generated tokens -> ACTION, upgraded to EDIT for the whole
        turn's non-think tokens if the non-think text matches the edit heuristic.

    Returns a tag list of length len(generated_token_ids).
    """
    n = len(generated_token_ids)
    tags = [SPAN_ACTION] * n
    if n == 0:
        return tags

    # Decode the full generated span and each token incrementally so we can map a
    # character offset to the token that covers it. tokenizer.decode on a growing
    # prefix is monotonic in the produced string length for normal BPE tokenizers.
    try:
        full_text = tokenizer.decode(generated_token_ids, skip_special_tokens=False)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("span_tagger: decode failed ({}); tagging turn as ACTION", e)
        return tags

    # Build per-token cumulative char lengths via incremental decode.
    char_ends: List[int] = []
    for k in range(1, n + 1):
        try:
            prefix_text = tokenizer.decode(generated_token_ids[:k], skip_special_tokens=False)
        except Exception:  # pragma: no cover
            prefix_text = full_text[: char_ends[-1] if char_ends else 0]
        char_ends.append(len(prefix_text))

    def _char_to_token(char_idx: int) -> int:
        # First token whose decoded prefix reaches char_idx.
        for k, end in enumerate(char_ends):
            if end > char_idx:
                return k
        return n

    # Determine non-think text (for ACTION vs EDIT) and tag THINK token spans.
    non_think_text = full_text
    for m in re.finditer(re.escape(_THINK_OPEN) + r"(.*?)" + re.escape(_THINK_CLOSE), full_text, re.DOTALL):
        start_tok = _char_to_token(m.start())
        end_tok = _char_to_token(max(m.end() - 1, m.start()))
        for k in range(start_tok, min(end_tok + 1, n)):
            tags[k] = SPAN_THINK
        non_think_text = non_think_text.replace(m.group(0), " ")

    is_edit_turn = bool(_EDIT_PATTERN.search(non_think_text))
    for k in range(n):
        if tags[k] != SPAN_THINK:
            tags[k] = SPAN_EDIT if is_edit_turn else SPAN_ACTION
    return tags


def tag_response_spans(
    messages: List[Dict],
    tokenizer,
    custom_chat_template: Optional[str] = None,
    assistant_token_ids: Optional[List[List[int]]] = None,
) -> List[int]:
    """Tag every response token with {OTHER, THINK, ACTION, EDIT}.

    Walks the SAME per-turn token accumulation as
    ``get_response_ids_and_loss_mask_from_messages`` so the returned tag list is
    1:1 (same length, same positions) with that function's ``response_ids``.

    Non-assistant (user/observation) tokens and assistant generation-prompt /
    post-EOS tokens are tagged OTHER (these are exactly the loss_mask==0 tokens).
    Assistant *generated* tokens are tagged THINK / ACTION / EDIT.

    Args:
        messages: the response messages (conversation[1:]) — same input the
            training-path id builder receives.
        tokenizer: HF tokenizer (must define eos_token_id).
        custom_chat_template: optional custom chat template string.
        assistant_token_ids: unused for tagging (the segmentation is recomputed
            via encode_messages_subset to match the training ids exactly); accepted
            for signature parity with the TIS seam / future exact-id assertions.

    Returns:
        List[int] tags, len == len(response_ids) from the training path.
    """
    assert len(messages), "messages list cannot be empty"
    generation_prompt_ids = get_generation_prompt_ids(tokenizer, custom_chat_template=custom_chat_template)
    eos_token_id = tokenizer.eos_token_id

    tags: List[int] = []
    for cur_message in messages:
        cur_token_ids = _encode_one(cur_message, tokenizer, custom_chat_template)
        if cur_message["role"] != "assistant":
            tags.extend([SPAN_OTHER] * len(cur_token_ids))
            continue
        prefix_len, gen_start, gen_end = _generated_slice_bounds(
            cur_token_ids, generation_prompt_ids, eos_token_id
        )
        # generation-prompt prefix -> OTHER
        tags.extend([SPAN_OTHER] * prefix_len)
        generated_token_ids = cur_token_ids[gen_start:gen_end]
        tags.extend(_tag_assistant_generated_span(generated_token_ids, tokenizer))
        # post-EOS tokens (e.g. trailing "\n") -> OTHER
        tags.extend([SPAN_OTHER] * (len(cur_token_ids) - gen_end))

    return tags
