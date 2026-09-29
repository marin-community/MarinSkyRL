"""Regression coverage for overridden EOS tokens inside structured output."""

from types import SimpleNamespace

import pytest
import torch
import xgrammar as xgr

from skyrl_train.inference_engines.vllm.grammar_stop_tokens import fill_xgrammar_bitmask


VOCAB = ["{", "}", '"', "value", ":", "foo", "<EOS>", "<EOT>", " ", "\n", "bar", '{"value":"']
# Most tokens must be invalid in a JSON string to exercise the compiler's
# cached accepted-token mask, rather than its complementary rejected mask.
VOCAB += [f"\ninvalid_{index}" for index in range(len(VOCAB), 96)]
VOCAB[31] = "<END>"
EOS = 6
EOT = 7
SCHEMA = '{"type":"object","properties":{"value":{"type":"string"}},"required":["value"],"additionalProperties":false}'


@pytest.fixture(params=[EOT, 31])
def matcher(request):
    info = xgr.TokenizerInfo(VOCAB, xgr.VocabType.RAW, stop_token_ids=[EOS])
    compiled = xgr.GrammarCompiler(info).compile_json_schema(SCHEMA)
    return xgr.GrammarMatcher(compiled, override_stop_tokens=[EOS, request.param])


def sampled_token(matcher, bitmask, index):
    fill_xgrammar_bitmask(SimpleNamespace(matcher=matcher), bitmask, index)
    logits = torch.zeros((1, len(VOCAB)))
    logits[0, matcher.stop_token_ids[-1]] = 100
    logits[0, 5] = 10
    xgr.apply_token_bitmask_inplace(logits, bitmask[index : index + 1])
    return int(logits.argmax())


def test_sampling_cannot_stop_inside_json_string(matcher):
    assert matcher.accept_token(11)
    bitmask = xgr.allocate_token_bitmask(1, len(VOCAB))
    token = sampled_token(matcher, bitmask, 0)
    assert token == 5
    assert matcher.accept_token(token)
    assert matcher.accept_token(2)
    assert matcher.accept_token(1)
    assert matcher.is_completed()
    token = sampled_token(matcher, bitmask, 0)
    assert token == matcher.stop_token_ids[-1]
    assert matcher.accept_token(token)
    assert matcher.is_terminated()


def test_speculative_rollback_restores_the_stop_mask(matcher):
    assert matcher.accept_token(11)
    bitmask = xgr.allocate_token_bitmask(4, len(VOCAB))
    for index, token in enumerate([5, 2, 1]):
        assert sampled_token(matcher, bitmask, index) != matcher.stop_token_ids[-1]
        assert matcher.accept_token(token)
    assert sampled_token(matcher, bitmask, 3) == matcher.stop_token_ids[-1]
    assert matcher.accept_token(matcher.stop_token_ids[-1])
    matcher.rollback(4)
    assert not matcher.is_completed()
    assert not matcher.is_terminated()
    assert sampled_token(matcher, bitmask, 0) == 5
    assert matcher.accept_token(5)
