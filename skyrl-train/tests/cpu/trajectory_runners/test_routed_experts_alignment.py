"""Route capture, alignment, and collation across the rollout boundary."""

import base64
import io
import pickle

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from skyrl_train.dataset.preprocess import _collate_routed_experts_from_arrays
from skyrl_train.trajectory_runners.routed_experts import normalize_routed_experts
from skyrl_train.trajectory_runners.trajectory_processing import (
    align_routed_experts_with_lcs,
    concatenate_trajectory_batches,
    encode_messages_subset,
    extract_routed_experts_from_rollout_details,
    get_generation_prompt_ids,
    get_response_ids_and_loss_mask_from_messages,
)


def _encoded_routes(rows):
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(rows, dtype=np.uint16), allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _route_row(value):
    return [[value, value + 1], [value + 2, value + 3]]


@pytest.mark.parametrize(
    ("expert_id", "dtype"),
    [(7, np.uint8), (300, np.uint16)],
)
def test_wire_routes_slice_prompt_and_append_final_sentinel(expert_id, dtype):
    payload = _encoded_routes([_route_row(1), _route_row(2), _route_row(expert_id)])

    routes = normalize_routed_experts(payload, [10, 11], [20, 21])

    assert routes.shape == (2, 2, 2)
    assert routes.dtype == dtype
    np.testing.assert_array_equal(routes[0], _route_row(expert_id))
    np.testing.assert_array_equal(routes[1], np.zeros((2, 2), dtype=dtype))
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(routes)), routes)


def test_harbor_extract_uses_each_turns_exact_prompt_and_completion_ids():
    details = [
        {
            "prompt_token_ids": [[10, 11], [10, 11, 20, 30]],
            "completion_token_ids": [[20, 21], [40, 41]],
            "extra": {
                "routed_experts": [
                    _encoded_routes([_route_row(1), _route_row(2), _route_row(3)]),
                    _encoded_routes([_route_row(1), _route_row(2), _route_row(3), _route_row(4), _route_row(5)]),
                ]
            },
        }
    ]

    routes = extract_routed_experts_from_rollout_details(details)

    assert len(routes) == 2
    np.testing.assert_array_equal(routes[0][0], _route_row(3))
    np.testing.assert_array_equal(routes[1][0], _route_row(5))
    assert np.count_nonzero(routes[0][-1]) == np.count_nonzero(routes[1][-1]) == 0


def test_lcs_routes_keep_token_positions_and_dtype():
    class Tokenizer:
        def convert_ids_to_tokens(self, ids):
            return [str(value) for value in ids]

    routes = np.asarray([_route_row(value) for value in (1, 5, 9)], dtype=np.uint8)
    aligned = align_routed_experts_with_lcs(
        [10, 20, 30, 40], routes, Tokenizer(), vllm_token_strings=["10", "30", "40"]
    )

    assert aligned.dtype == np.uint8
    np.testing.assert_array_equal(aligned[0], routes[0])
    np.testing.assert_array_equal(aligned[1], np.zeros((2, 2), dtype=np.uint8))
    np.testing.assert_array_equal(aligned[2:], routes[1:])


@pytest.mark.parametrize("model_name", ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen3-0.6B"])
def test_multiturn_assembly_preserves_generated_routes_and_masks_context(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generation_prompt_ids = get_generation_prompt_ids(tokenizer)
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello there"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "Good"},
    ]

    def generated_count(content):
        ids = encode_messages_subset([{"role": "assistant", "content": content}], tokenizer)
        eos = len(ids) - 1 - ids[::-1].index(tokenizer.eos_token_id)
        return eos + 1 - len(generation_prompt_ids)

    turns = [
        np.asarray([_route_row(10 + index) for index in range(generated_count("Hello there"))], dtype=np.uint8),
        np.asarray([_route_row(50 + index) for index in range(generated_count("Good"))], dtype=np.uint8),
    ]

    response_ids, loss_mask, _, routes = get_response_ids_and_loss_mask_from_messages(
        messages, tokenizer, assistant_routed_experts=turns
    )

    assert routes.shape == (len(response_ids), 2, 2)
    assert routes.dtype == np.uint8
    np.testing.assert_array_equal(routes[np.asarray(loss_mask, dtype=bool)], np.concatenate(turns))
    assert not np.any(routes[np.logical_not(loss_mask)])


@pytest.mark.parametrize(("num_experts", "expected_dtype"), [(256, torch.uint8), (512, torch.int16)])
def test_collator_pads_routes_in_final_dtype(num_experts, expected_dtype):
    first = np.asarray([_route_row(1), _route_row(5)], dtype=np.uint8)
    second = np.asarray([_route_row(9)], dtype=np.uint8)

    routes = _collate_routed_experts_from_arrays([first, second], max_output_len=3, num_experts=num_experts)

    assert routes.shape == (2, 3, 2, 2)
    assert routes.dtype == expected_dtype
    np.testing.assert_array_equal(routes[0, :2].numpy(), first)
    np.testing.assert_array_equal(routes[1, 0].numpy(), second[0])
    assert not torch.any(routes[:, 2])
    assert not torch.any(routes[1, 1:])


def test_concatenation_fills_missing_sample_with_matching_route_geometry():
    routes = np.asarray([_route_row(1), _route_row(5)], dtype=np.uint8)
    captured = {
        "prompt_token_ids": [[10]],
        "response_ids": [[20, 21]],
        "rewards": [[0.0, 1.0]],
        "loss_masks": [[1, 1]],
        "rollout_routed_experts": [routes],
    }
    missing = {
        "prompt_token_ids": [[30]],
        "response_ids": [[40]],
        "rewards": [[1.0]],
        "loss_masks": [[1]],
    }

    merged = concatenate_trajectory_batches([captured, missing], tis_lcs_alert_threshold=0.005)

    assert len(merged["rollout_routed_experts"]) == 2
    assert merged["rollout_routed_experts"][0] is routes
    sentinel = merged["rollout_routed_experts"][1]
    assert sentinel.shape == (1, 2, 2)
    assert sentinel.dtype == np.uint8
    assert not np.any(sentinel)
