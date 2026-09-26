"""A tiny Qwen2 policy and GSM8K-format dataset for end-to-end CPU training runs.

The model is a hand-shaped bigram policy. Zeroed attention and MLP output projections make each
position's hidden state a function of its current token only, and the embedding and LM head encode
a transition table over five answer tokens. The policy emits ``#### 1`` (the GSM8K reward format)
with moderate probability, so GRPO groups have reward variance and RL can raise the success rate.
Training updates every parameter, so the model is free to leave the bigram form.
"""

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

TOKENIZER_SOURCE = "Qwen/Qwen2.5-0.5B-Instruct"
GROUND_TRUTH = "1"
HIDDEN_SIZE = 16
DISALLOWED_LOGIT = -8.0
ALLOWED_LOGIT_OFFSET = 8.0

# Answer tokens in transition-table order.
ANSWER_TOKENS = ("####", " ", "1", "2")
# Row: the current token's state (0 = any other token, then ANSWER_TOKENS). Column: the next token,
# ANSWER_TOKENS followed by end of turn.
TRANSITIONS = (
    (0.6, 0.1, 0.1, 0.1, 0.1),
    (0.1, 0.6, 0.1, 0.1, 0.1),
    (0.1, 0.1, 0.4, 0.3, 0.1),
    (0.1, 0.1, 0.1, 0.1, 0.6),
    (0.2, 0.1, 0.1, 0.1, 0.5),
)


def build_tiny_policy(output_dir: Path) -> Path:
    """Write the tiny policy and its tokenizer as a Hugging Face model directory."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_SOURCE)
    answer_ids = [_single_token_id(tokenizer, text) for text in ANSWER_TOKENS]
    next_ids = [*answer_ids, tokenizer.convert_tokens_to_ids("<|im_end|>")]

    config = Qwen2Config(
        vocab_size=151936,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=2 * HIDDEN_SIZE,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=1024,
        tie_word_embeddings=False,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        torch_dtype="float32",
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(config)
    with torch.no_grad():
        for layer in model.model.layers:
            layer.self_attn.o_proj.weight.zero_()
            layer.mlp.down_proj.weight.zero_()
        embedding = model.model.embed_tokens.weight
        embedding.zero_()
        embedding[:, 0] = 1.0
        for state, token_id in enumerate(answer_ids, start=1):
            embedding[token_id] = 0.0
            embedding[token_id, state] = 1.0
        # RMSNorm scales a one-hot state vector to sqrt(hidden_size).
        state_scale = HIDDEN_SIZE**0.5
        lm_head = model.lm_head.weight
        lm_head.normal_(std=0.02)
        lm_head[:, : len(TRANSITIONS)] = DISALLOWED_LOGIT / state_scale
        transitions = torch.tensor(TRANSITIONS)
        for column, token_id in enumerate(next_ids):
            logits = transitions[:, column].log() + ALLOWED_LOGIT_OFFSET
            lm_head[token_id, : len(TRANSITIONS)] = logits / state_scale

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def write_gsm8k_dataset(path: Path, num_prompts: int) -> Path:
    """Write ``num_prompts`` GSM8K-environment rows whose ground truth is always ``1``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for index in range(num_prompts):
            row = {
                "prompt": [{"role": "user", "content": f"Question {index}: what is one? End with '#### 1'."}],
                "env_class": "gsm8k",
                "reward_spec": {"method": "rule", "ground_truth": GROUND_TRUTH},
            }
            handle.write(json.dumps(row) + "\n")
    return path


def _single_token_id(tokenizer, text: str) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"{text!r} is not a single token: {token_ids}")
    return token_ids[0]
