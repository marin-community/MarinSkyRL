"""Stage B (F5/F4) flag-off byte-identical guarantee — the gating invariant.

Mirrors tests/cpu/test_ep_config_noop.py and the routed_experts no-op test: with
the per-token reward channel DISABLED, (1) the config key defaults to false and is
purely additive, (2) the collator returns tensors byte-identical to the pre-Stage-B
call and emits NO new tensors, (3) the TrainingInputBatch keyset is unchanged.

Run:
    pytest tests/cpu/reward/test_token_channel_noop.py
"""

import pytest
import torch
from transformers import AutoTokenizer

from skyrl_train.config.utils import get_default_config
from skyrl_train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl_train.generators.utils import concatenate_generator_outputs


def test_config_key_defaults_false():
    cfg = get_default_config()
    assert "enable_token_reward_channel" in cfg.trainer.algorithm
    assert cfg.trainer.algorithm.enable_token_reward_channel is False


def test_stage_d_config_keys_default_noop():
    """Stage D (F7 + M4) config keys default to the byte-identical no-op:
    think_token_weight=1.0 (no loss down-weight), think_token_cost=0.0 (no cost)."""
    cfg = get_default_config()
    assert "think_token_weight" in cfg.trainer.algorithm
    assert cfg.trainer.algorithm.think_token_weight == 1.0
    assert "think_token_cost" in cfg.trainer.algorithm
    assert cfg.trainer.algorithm.think_token_cost == 0.0
    assert "think_max_cost_per_turn" in cfg.trainer.algorithm


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


def test_collator_flag_off_byte_identical(tokenizer):
    """Channel-off collator: last two returns are None and the first 7 tensors are
    byte-identical to a call that never passed the channel args."""
    prompts = tokenizer(["abc", "12345"])["input_ids"]
    responses = tokenizer(["def", "67890"])["input_ids"]
    rewards = [torch.tensor([0.0, 1.0, 0.0]), torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])]
    loss_masks = [[1, 1, 0], [1, 1, 1, 0, 0]]
    logprobs = [[0.0] * len(r) for r in responses]

    # Pre-Stage-B-equivalent call (no channel args at all).
    res_off = convert_prompts_responses_to_batch_tensors(
        tokenizer, prompts, responses, rewards, loss_masks, logprobs
    )
    assert len(res_off) == 9
    assert res_off[7] is None  # token_level_shaping_tensor
    assert res_off[8] is None  # response_span_tags_tensor

    # Explicit None channel args must produce identical first-7 tensors.
    res_off2 = convert_prompts_responses_to_batch_tensors(
        tokenizer, prompts, responses, rewards, loss_masks, logprobs, None, None, None
    )
    for a, b in zip(res_off[:7], res_off2[:7]):
        if a is None and b is None:
            continue
        assert torch.equal(a, b)
    assert res_off2[7] is None
    assert res_off2[8] is None


def test_concatenate_flag_off_keys_absent():
    """When no batch carries the channel keys, the concatenated GeneratorOutput
    must NOT contain them (key absent, not None) -> byte-identical keyset."""
    out = {
        "prompt_token_ids": [[1, 2]],
        "response_ids": [[3, 4, 5]],
        "rewards": [[0.0, 0.0, 1.0]],
        "loss_masks": [[1, 1, 1]],
        "stop_reasons": ["stop"],
        "rollout_logprobs": None,
        "rollout_metrics": {},
    }
    merged = concatenate_generator_outputs([out, out])
    assert "token_level_shaping" not in merged
    assert "response_span_tags" not in merged
    assert "rollout_routed_experts" not in merged


def test_concatenate_flag_on_sentinel_fill():
    """When SOME batches carry the channel and others don't, the missing batches
    are sentinel-filled (zeros) so the concatenated lists stay 1:1 with
    response_ids."""
    out_with = {
        "prompt_token_ids": [[1, 2]],
        "response_ids": [[3, 4, 5]],
        "rewards": [[0.0, 0.0, 1.0]],
        "loss_masks": [[1, 1, 1]],
        "stop_reasons": ["stop"],
        "rollout_logprobs": None,
        "rollout_metrics": {},
        "token_level_shaping": [[0.0, 0.0, 0.0]],
        "response_span_tags": [[1, 2, 2]],
    }
    out_without = {
        "prompt_token_ids": [[9]],
        "response_ids": [[6, 7]],
        "rewards": [[0.0, 1.0]],
        "loss_masks": [[1, 1]],
        "stop_reasons": ["stop"],
        "rollout_logprobs": None,
        "rollout_metrics": {},
    }
    merged = concatenate_generator_outputs([out_with, out_without])
    assert merged["token_level_shaping"] == [[0.0, 0.0, 0.0], [0.0, 0.0]]
    assert merged["response_span_tags"] == [[1, 2, 2], [0, 0]]
