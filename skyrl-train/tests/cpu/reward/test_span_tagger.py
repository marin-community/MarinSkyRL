"""Stage B (F4) — response-token span tagger aligns 1:1 with the TIS layout.

The tagger must produce a tag list the SAME length as response_ids from
get_response_ids_and_loss_mask_from_messages (the exact-token-id / TIS layout),
with THINK tokens inside <think>...</think>, ACTION/EDIT on the generated
non-think tokens, and OTHER on every loss_mask==0 token (user/observation,
generation-prompt prefix, post-EOS).

Run:
    pytest tests/cpu/reward/test_span_tagger.py
"""

import pytest
from transformers import AutoTokenizer

from skyrl_train.generators.utils import get_response_ids_and_loss_mask_from_messages
from skyrl_train.utils.span_tagger import (
    SPAN_ACTION,
    SPAN_EDIT,
    SPAN_OTHER,
    SPAN_THINK,
    tag_response_spans,
)


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


def _messages():
    return [
        {"role": "user", "content": "fix the failing test"},
        {
            "role": "assistant",
            "content": "<think>The test imports foo; I should inspect it first</think>"
            "Let me look at the test output and run pytest",
        },
        {"role": "user", "content": "<observation>1 failed, 3 passed</observation>"},
        {
            "role": "assistant",
            "content": "<think>I need to patch the bug</think>"
            "cat > foo.py <<EOF\ndef foo():\n    return 42\nEOF",
        },
    ]


def test_tags_align_one_to_one_with_tis_layout(tokenizer):
    msgs = _messages()
    response_ids, loss_mask, _ = get_response_ids_and_loss_mask_from_messages(msgs, tokenizer)
    tags = tag_response_spans(msgs, tokenizer)
    # 1:1 with the exact-token-id layout TIS uses.
    assert len(tags) == len(response_ids)
    assert len(tags) == len(loss_mask)


def test_other_tokens_match_loss_mask_zero(tokenizer):
    """Every loss_mask==0 token (user/observation/prefix/post-EOS) is OTHER, and
    every generated (loss_mask==1) token is NOT OTHER."""
    msgs = _messages()
    response_ids, loss_mask, _ = get_response_ids_and_loss_mask_from_messages(msgs, tokenizer)
    tags = tag_response_spans(msgs, tokenizer)
    for m, t in zip(loss_mask, tags):
        if m == 0:
            assert t == SPAN_OTHER
        else:
            assert t in (SPAN_THINK, SPAN_ACTION, SPAN_EDIT)


def test_think_tokens_tagged(tokenizer):
    msgs = _messages()
    tags = tag_response_spans(msgs, tokenizer)
    assert SPAN_THINK in tags, "no THINK tokens tagged despite <think> spans"
    # Both assistant turns have a <think> block -> several THINK tokens.
    assert tags.count(SPAN_THINK) >= 4


def test_edit_turn_tagged_edit(tokenizer):
    """The second assistant turn writes a file (heredoc) -> its non-think
    generated tokens are EDIT, not ACTION."""
    msgs = _messages()
    tags = tag_response_spans(msgs, tokenizer)
    assert SPAN_EDIT in tags, "edit (heredoc) turn not tagged EDIT"


def test_action_turn_tagged_action(tokenizer):
    """A turn with a non-edit action (run pytest) -> ACTION tokens."""
    msgs = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "<think>run it</think>pytest -q tests/"},
    ]
    tags = tag_response_spans(msgs, tokenizer)
    assert SPAN_ACTION in tags
    assert SPAN_EDIT not in tags


def test_empty_messages_raises(tokenizer):
    with pytest.raises(AssertionError):
        tag_response_spans([], tokenizer)
