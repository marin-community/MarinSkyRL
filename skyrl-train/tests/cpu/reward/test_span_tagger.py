"""Stage B (F4) — response-token span tagger aligns 1:1 with the TIS layout.

The tagger must produce a tag list the SAME length as response_ids from
get_response_ids_and_loss_mask_from_messages (the exact-token-id / TIS layout),
with THINK tokens inside <think>...</think>, ACTION/EDIT on the generated
non-think tokens, and OTHER on every loss_mask==0 token (user/observation,
generation-prompt prefix, post-EOS).

Run:
    pytest tests/cpu/reward/test_span_tagger.py
"""

import itertools

import pytest
from transformers import AutoTokenizer

from skyrl_train.trajectory_runners.trajectory_processing import get_response_ids_and_loss_mask_from_messages
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
            "content": "<think>I need to patch the bug</think>cat > foo.py <<EOF\ndef foo():\n    return 42\nEOF",
        },
    ]


def test_tags_follow_tis_layout_and_turn_structure(tokenizer):
    msgs = _messages()
    response_ids, loss_mask, _ = get_response_ids_and_loss_mask_from_messages(msgs, tokenizer)
    tags = tag_response_spans(msgs, tokenizer)

    assert len(tags) == len(response_ids) == len(loss_mask)
    # OTHER exactly on loss_mask==0 tokens (generation prefix, observations, post-EOS).
    assert [t == SPAN_OTHER for t in tags] == [m == 0 for m in loss_mask]
    # Each assistant turn is its <think> span followed by ACTION (run pytest) or EDIT (heredoc write).
    runs = [tag for tag, _ in itertools.groupby(tags)]
    assert runs == [SPAN_OTHER, SPAN_THINK, SPAN_ACTION, SPAN_OTHER, SPAN_THINK, SPAN_EDIT, SPAN_OTHER]
