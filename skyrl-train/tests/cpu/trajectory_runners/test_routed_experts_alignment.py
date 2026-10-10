"""Route capture, alignment, and collation across the rollout boundary."""

import base64
import io
import pickle

import numpy as np
import pytest
import torch

from skyrl_train.dataset.routed_expert_batch import RoutedExpertRows, _collate_routed_experts_from_arrays
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.trajectory_runners.routed_experts import normalize_routed_experts
from skyrl_train.trajectory_runners.trajectory_processing import (
    concatenate_trajectory_batches,
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
def test_wire_routes_select_prediction_positions(expert_id, dtype):
    payload = _encoded_routes([_route_row(1), _route_row(2), _route_row(expert_id)])

    routes = normalize_routed_experts(payload, [10, 11], [20, 21])

    assert routes.shape == (2, 2, 2)
    assert routes.dtype == dtype
    np.testing.assert_array_equal(routes[0], _route_row(2))
    np.testing.assert_array_equal(routes[1], _route_row(expert_id))
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(routes)), routes)


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


def test_compact_routes_survive_batch_slicing_and_materialize_to_local_response_lengths():
    rows = (
        np.asarray([_route_row(value) for value in (1, 5, 9, 13)], dtype=np.uint8),
        np.asarray([_route_row(21)], dtype=np.uint8),
        np.asarray([_route_row(value) for value in (29, 33)], dtype=np.uint8),
    )
    batch = TrainingInputBatch(
        {"sequences": torch.tensor([[1, 0, 0, 0, 0], [2, 0, 0, 0, 0], [3, 0, 0, 0, 0]])},
        routed_expert_rows=RoutedExpertRows(rows, response_len=4, num_experts=256),
    )
    batch.metadata = {"response_length": 4}

    restored = pickle.loads(pickle.dumps(batch))
    forward = restored.select(keys=["sequences", "rollout_routed_experts"], metadata_keys=["response_length"])
    assert forward.metadata == {"response_length": 4}
    np.testing.assert_array_equal(forward.chunk(1)[1].routed_experts_tensor()[0].numpy(), rows[1])

    microbatches = restored.chunk(1)
    assert [micro.routed_experts_tensor().shape[1] for micro in microbatches] == [4, 1, 2]
    for index, (row, micro) in enumerate(zip(rows, microbatches, strict=True), start=1):
        assert micro["sequences"][0, 0].item() == index
        np.testing.assert_array_equal(micro.routed_experts_tensor()[0].numpy(), row)


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
