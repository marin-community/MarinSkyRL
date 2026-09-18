import pytest
from transformers import AutoTokenizer

from skyrl_train.trajectory_runners.trajectory_processing import get_custom_chat_template

MARIN_TOKENIZER = "marin-community/marin-tokenizer"
MARIN_TOKENIZER_REVISION = "a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2"
CONVERSATION = [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]


@pytest.fixture(scope="module")
def marin_tokenizer():
    return AutoTokenizer.from_pretrained(MARIN_TOKENIZER, revision=MARIN_TOKENIZER_REVISION)


@pytest.fixture
def template():
    return get_custom_chat_template({"source": "name", "name_or_path": "marin_tokenizer"})


def test_marin_tokenizer_template_is_the_tokenizers_own(marin_tokenizer, template):
    assert template == marin_tokenizer.chat_template


def test_marin_tokenizer_template_masks_exactly_the_assistant_turn(marin_tokenizer, template):
    encoded = marin_tokenizer.apply_chat_template(
        CONVERSATION, chat_template=template, tokenize=True, return_assistant_tokens_mask=True, return_dict=True
    )
    masked = [t for t, m in zip(encoded["input_ids"], encoded["assistant_masks"], strict=True) if m]
    unmasked = [t for t, m in zip(encoded["input_ids"], encoded["assistant_masks"], strict=True) if not m]

    assert marin_tokenizer.decode(masked).strip() == "answer<|eot_id|>"
    assert "question" in marin_tokenizer.decode(unmasked) and "answer" not in marin_tokenizer.decode(unmasked)
