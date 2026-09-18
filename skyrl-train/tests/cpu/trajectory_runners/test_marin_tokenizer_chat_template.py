from transformers import AutoTokenizer

from skyrl_train.trajectory_runners.trajectory_processing import get_custom_chat_template

CONVERSATION = [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]


def test_marin_tokenizer_template_renders_headers_and_a_generation_prompt():
    template = get_custom_chat_template({"source": "name", "name_or_path": "marin_tokenizer"})
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    text = tokenizer.apply_chat_template(CONVERSATION[:1], chat_template=template, tokenize=False, add_generation_prompt=True)

    assert "<|start_header_id|>user<|end_header_id|>\nquestion<|eot_id|>" in text
    assert text.endswith("<|start_header_id|>assistant<|end_header_id|>\n")


def test_marin_tokenizer_template_masks_exactly_the_assistant_turn():
    template = get_custom_chat_template({"source": "name", "name_or_path": "marin_tokenizer"})
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    encoded = tokenizer.apply_chat_template(
        CONVERSATION, chat_template=template, tokenize=True, return_assistant_tokens_mask=True, return_dict=True
    )
    masked = [t for t, m in zip(encoded["input_ids"], encoded["assistant_masks"], strict=True) if m]
    unmasked = [t for t, m in zip(encoded["input_ids"], encoded["assistant_masks"], strict=True) if not m]

    assert "answer" in tokenizer.decode(masked)
    assert "question" in tokenizer.decode(unmasked) and "answer" not in tokenizer.decode(unmasked)
