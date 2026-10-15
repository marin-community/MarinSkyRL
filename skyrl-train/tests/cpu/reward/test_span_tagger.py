"""Span tags stay aligned with the supplied generated tokens."""

import itertools

import pytest
from transformers import AutoTokenizer

from skyrl_train.utils.span_tagger import (
    SPAN_ACTION,
    SPAN_EDIT,
    SPAN_THINK,
    tag_generated_tokens,
)


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


@pytest.mark.parametrize(
    "response,action_tag",
    [
        ("<think>Read the test output</think>Run pytest", SPAN_ACTION),
        ("<think>Repair the function</think>cat > foo.py <<EOF\ndef foo():\n    return 42\nEOF", SPAN_EDIT),
    ],
)
def test_generated_tokens_keep_think_and_action_spans(tokenizer, response, action_tag):
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    tags = tag_generated_tokens(response_ids, tokenizer)

    assert len(tags) == len(response_ids)
    runs = [tag for tag, _ in itertools.groupby(tags)]
    assert runs == [SPAN_THINK, action_tag]
