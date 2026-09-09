import json

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from skyrl_gym.envs.thinking_contract import post_thinking_segment, score_thinking_contract


@pytest.fixture
def decoder():
    vocabulary = [
        "[UNK]",
        "<|start_think|>",
        "<|end_think|>",
        "<|eot_id|>",
        "<|start_header_id|>",
        "####",
        "42",
        "41",
        "Answer:",
        "reasoning",
        "user",
    ]
    tokenizer = Tokenizer(WordLevel(dict(zip(vocabulary, range(len(vocabulary)))), unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.add_special_tokens(vocabulary[1:5])
    return tokenizer


def score(decoder, text, *, env_class="gsm8k", stop_reason="stop"):
    ids = decoder.encode(text, add_special_tokens=False).ids
    return score_thinking_contract(
        env_class=env_class,
        ground_truth="42",
        native_response=decoder.decode(ids, skip_special_tokens=True),
        prompt_tokens=[decoder.token_to_id("<|start_think|>")],
        response_tokens=ids,
        stop_reason=stop_reason,
        decoder=decoder,
    )


def test_answer_inside_unfinished_thinking_is_not_a_verifier_success(decoder):
    result = score(decoder, "reasoning #### 42", stop_reason="length")
    assert result.legacy_full_text_reward == 1
    assert result.verifier_reward == result.contract_correct == result.score_contract_completed == 0
    assert result.boundary_status == "missing_thinking_end"


def test_correct_final_answer_overrides_an_incorrect_thinking_prefix(decoder):
    result = score(decoder, "#### 41 <|end_think|> #### 42 <|eot_id|>")
    assert result.legacy_full_text_reward == 0
    assert result.verifier_reward == result.contract_correct == result.score_contract_completed == 1


@pytest.mark.parametrize(
    "stop_reason,completed", [("stop", 1), ("end_turn", 1), ("length", 0), ("repetition", 0), ("error", 0)]
)
def test_stopping_does_not_implicitly_penalize_correct_native_reward(decoder, stop_reason, completed):
    result = score(decoder, "reasoning <|end_think|> #### 42", stop_reason=stop_reason)
    assert result.verifier_reward == result.contract_correct == 1
    assert result.score_contract_completed == completed


@pytest.mark.parametrize(
    "text,status",
    [
        ("<|end_think|> #### 42 <|end_think|>", "multiple_thinking_ends"),
        ("<|start_think|> <|end_think|> #### 42", "unexpected_thinking_start"),
        ("<|end_think|> #### 42 <|start_header_id|> user", "role_or_thinking_continuation"),
    ],
)
def test_ambiguous_or_continued_turn_has_no_accepted_answer(decoder, text, status):
    result = score(decoder, text)
    assert result.boundary_status == status
    assert result.contract_correct == result.score_contract_completed == 0


def test_missing_or_nonempty_inherited_thinking_prefix_is_unresolved(decoder):
    response = decoder.encode("<|end_think|> #### 42").ids
    assert post_thinking_segment(decoder, [], response) == (None, "missing_thinking_prompt")
    prompt = decoder.encode("<|start_think|> reasoning").ids
    assert post_thinking_segment(decoder, prompt, response) == (None, "invalid_thinking_prompt")


def test_aime_unresolved_answer_keeps_native_negative_scale(decoder):
    result = score(decoder, "Answer: 42", env_class="aime", stop_reason="length")
    assert result.legacy_full_text_reward == 1
    assert result.verifier_reward == -1 and result.contract_correct == 0


def test_reasoning_gym_fractional_reward_remains_incorrect(decoder):
    text = "<|end_think|> Answer: 42 reasoning"
    ids = decoder.encode(text).ids
    gold = json.dumps(
        {
            "task": "chain_sum",
            "entry": {"question": "41 + 1", "answer": "42", "metadata": {"source_dataset": "chain_sum"}},
        }
    )
    result = score_thinking_contract(
        env_class="reasoning_gym",
        ground_truth=gold,
        native_response=decoder.decode(ids, skip_special_tokens=True),
        prompt_tokens=[decoder.token_to_id("<|start_think|>")],
        response_tokens=ids,
        stop_reason="stop",
        decoder=decoder,
    )
    assert 0 < result.verifier_reward < 1
    assert result.contract_correct == result.score_contract_completed == 0
