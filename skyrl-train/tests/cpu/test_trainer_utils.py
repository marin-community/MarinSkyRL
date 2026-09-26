"""
uv run --isolated --group dev --extra cpu pytest tests/cpu/test_trainer_utils.py
"""

from skyrl_train.batch_sampling import filter_trajectory_batch
from skyrl_train.utils.trainer_utils import (
    run_on_each_node,
    cleanup_old_checkpoints,
    validate_consistency_for_latest_checkpoint,
    sanitize_data_source,
    calculate_per_dataset_metrics,
    evaluation_response_metrics,
    dump_per_dataset_eval_results,
    build_eval_dataloader,
)
from skyrl_train.trajectory_runners.base import TrajectoryRequestBatch, TrajectoryBatch
from skyrl_train.trajectory_runners.trajectory_processing import validate_trajectory_batch
from typing import Union
import ray
import os
import tempfile
import pytest
import re

from unittest.mock import Mock, patch
import json
import fsspec
from skyrl_train.evaluate import evaluation_dump_dir
from skyrl_train.io import io
from tests.cpu.util import example_dummy_config

BasicType = Union[int, float, str, bool, type(None)]


@pytest.fixture
def dummy_config():
    return example_dummy_config()


@pytest.mark.usefixtures("ray_init")
def test_run_on_node_local_rank_0():
    def fn(x):
        return x + 1

    all_nodes = [node for node in ray.nodes() if node.get("CPU", 0) > 0]
    # repeat the node ids 4 times to test that the function is called only once per node
    node_ids = [all_nodes[i]["NodeID"] for i in range(len(all_nodes))] * 4
    ret = run_on_each_node(node_ids, fn, 1)
    assert ret == [2] * len(all_nodes)


def setup_mock_ckpts(tmpdir, checkpoint_steps):
    """
    Sets up dummy checkpoint directories.
    """
    # Create dummy checkpoint directories
    for step in checkpoint_steps:
        os.makedirs(os.path.join(tmpdir, f"global_step_{step}"))
    return


def test_cleanup_old_checkpoints():
    """
    Verify that _cleanup_old_checkpoints correctly removes old checkpoints
    while keeping the most recent ones.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Setup
        checkpoint_steps = [1, 2, 10, 11]
        setup_mock_ckpts(tmpdir, checkpoint_steps=checkpoint_steps)

        # 2. Execute
        cleanup_old_checkpoints(tmpdir, max_checkpoints=2)

        # 3. Verify
        remaining_dirs = sorted(os.listdir(tmpdir))
        expected_remaining = ["global_step_10", "global_step_11"]

        assert len(remaining_dirs) == 2, "Incorrect number of checkpoints remaining"
        assert remaining_dirs == expected_remaining, "Did not keep the correct (most recent) checkpoints"

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Setup
        checkpoint_steps = [1, 2, 10, 11]
        setup_mock_ckpts(tmpdir, checkpoint_steps=checkpoint_steps)

        # 2. Execute
        cleanup_old_checkpoints(tmpdir, max_checkpoints=0)

        # 3. Verify
        remaining_dirs = sorted(os.listdir(tmpdir))

        assert len(remaining_dirs) == 0, "Cleanup should have removed all checkpoints"

    # Test cleanup with `current_global_step` less than the highest global step in the folder
    # This means that the folder contains checkpoints from a previous run.
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Setup
        checkpoint_steps = [1, 2, 10, 11]
        setup_mock_ckpts(tmpdir, checkpoint_steps=checkpoint_steps)

        # 2. Execute
        cleanup_old_checkpoints(tmpdir, max_checkpoints=4)

        remaining_dirs = sorted(os.listdir(tmpdir))
        assert len(remaining_dirs) == 4, "Cleanup should not have removed any checkpoints"


def test_cleanup_does_not_run_when_not_needed():
    """
    Verify that cleanup does not remove any checkpoints if the total number
    is less than or equal to max_checkpoints.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Setup
        checkpoint_steps = [1, 2, 3, 4]
        setup_mock_ckpts(tmpdir, checkpoint_steps=checkpoint_steps)

        # 2. Execute
        cleanup_old_checkpoints(tmpdir, max_checkpoints=5)

        # 3. Verify
        remaining_dirs = sorted(os.listdir(tmpdir))
        assert len(remaining_dirs) == 4, "Cleanup should not have removed any checkpoints"


def test_cleanup_with_negative_max_checkpoints():
    """
    Verify that cleanup is disabled when max_checkpoints is -1
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Setup
        checkpoint_steps = [1, 2, 3, 4, 5]
        setup_mock_ckpts(tmpdir, checkpoint_steps=checkpoint_steps)

        # 2. Execute
        cleanup_old_checkpoints(tmpdir, max_checkpoints=-1)

        # 3. Verify
        remaining_dirs = sorted(os.listdir(tmpdir))
        assert len(remaining_dirs) == 5, "Cleanup should be disabled when max_checkpoints is -1"


def test_validate_consistency_for_latest_checkpoint():
    """
    Verify that `validate_consistency_for_latest_checkpoint` correctly validates the checkpoint folder.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Setup
        checkpoint_steps = [1, 2, 3, 4, 5]
        setup_mock_ckpts(tmpdir, checkpoint_steps=checkpoint_steps)

        latest_ckpt_file = os.path.join(tmpdir, "latest_ckpt_global_step.txt")
        with open(latest_ckpt_file, "w") as f:
            f.write("5")

        latest_ckpt_path = os.path.join(tmpdir, "global_step_5")
        ckpt_iteration = 5

        # 2. Execute
        validate_consistency_for_latest_checkpoint(
            tmpdir, ckpt_iteration, latest_ckpt_path, latest_ckpt_file, save_interval=1
        )


def test_validate_consistency_for_latest_checkpoint_with_inconsistent_folder():
    """
    Verify that `validate_consistency_for_latest_checkpoint` correctly validates the checkpoint folder.
    """
    # Example 1: `latest_ckpt_global_step.txt` points to a lower global step than the highest global step in the folder
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Setup
        checkpoint_steps = [1, 2, 3, 4, 5]
        setup_mock_ckpts(tmpdir, checkpoint_steps=checkpoint_steps)

        # change the latest checkpoint file to point to a lower global step
        latest_ckpt_file = os.path.join(tmpdir, "latest_ckpt_global_step.txt")
        with open(latest_ckpt_file, "w") as f:
            f.write("3")

        latest_ckpt_path = os.path.join(tmpdir, "global_step_3")
        ckpt_iteration = 3
        save_interval = 1

        # 2. Execute
        with pytest.raises(ValueError, match="Inconsistent checkpoint folder"):
            validate_consistency_for_latest_checkpoint(
                tmpdir, ckpt_iteration, latest_ckpt_path, latest_ckpt_file, save_interval=save_interval
            )

    # Example 2: `latest_ckpt_global_step.txt` points to a lower global step but it's within the save interval
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Setup
        checkpoint_steps = [1, 3, 5]
        setup_mock_ckpts(tmpdir, checkpoint_steps=checkpoint_steps)

        # change the latest checkpoint file to point to a lower global step
        latest_ckpt_file = os.path.join(tmpdir, "latest_ckpt_global_step.txt")
        with open(latest_ckpt_file, "w") as f:
            f.write("3")

        save_interval = 2
        latest_ckpt_path = os.path.join(tmpdir, "global_step_3")
        ckpt_iteration = 3

        # 2. Execute
        validate_consistency_for_latest_checkpoint(
            tmpdir, ckpt_iteration, latest_ckpt_path, latest_ckpt_file, save_interval=save_interval
        )


def test_sanitize_data_source_none():
    """Test sanitize_data_source with None input."""
    result = sanitize_data_source(None)
    assert result == "unknown"


def test_sanitize_data_source_slash_replacement():
    """Test sanitize_data_source replaces slashes with underscores."""
    result = sanitize_data_source("dataset/with/slashes")
    assert result == "dataset_with_slashes"


def test_sanitize_data_source_normal_string():
    """Test sanitize_data_source with normal string."""
    result = sanitize_data_source("normal_dataset")
    assert result == "normal_dataset"


def test_evaluation_response_metrics_report_work_and_stop_contributions():
    batch = {
        "response_ids": [[1, 2, 3], [4], [5, 6]],
        "rewards": [1.0, 0.0, 1.0],
        "stop_reasons": ["stop", "length", "stop"],
    }
    metrics = evaluation_response_metrics(batch)
    assert metrics["response_tokens"] == 6
    assert metrics["response_tokens_mean"] == pytest.approx(2.0)
    assert metrics["response_tokens_max"] == 3
    assert metrics["stop_reason_coverage"] == 1
    assert metrics["completed_stop_fraction"] == pytest.approx(2 / 3)
    # Contributions divide by every evaluated response.
    assert metrics["completed_stop_score_contribution"] == pytest.approx(2 / 3)
    assert metrics["length_stop_score_contribution"] == 0


def test_evaluation_response_metrics_suppress_fractions_without_full_stop_coverage():
    batch = {"response_ids": [[1, 2], [3]], "rewards": [1.0, 1.0], "stop_reasons": ["stop", None]}
    metrics = evaluation_response_metrics(batch)
    assert metrics["stop_reason_coverage"] < 1
    assert "completed_stop_fraction" not in metrics
    assert "completed_stop_score_contribution" not in metrics
    assert metrics["response_tokens"] == 3
    with pytest.raises(ValueError):
        evaluation_response_metrics({"response_ids": [], "rewards": []})


def test_calculate_per_dataset_metrics_single_source():
    """Test calculate_per_dataset_metrics with single data source."""
    # Create test data
    trajectory_batches = {
        "rewards": [0.5, 0.7, 0.9],
        "prompt_token_ids": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        "response_ids": [[10, 11], [12, 13], [14, 15]],
    }
    uids = ["uid1", "uid2", "uid3"]
    data_sources = ["dataset1", "dataset1", "dataset1"]

    result = calculate_per_dataset_metrics(trajectory_batches, uids, data_sources, 2)

    # Verify results - actual computed values
    # Mean reward: (0.5 + 0.7 + 0.9) / 3 = 0.7
    # Pass@N: all rewards > 0, all unique uids, so 3/3 = 1.0
    assert "eval/dataset1/avg_score" in result
    assert "eval/dataset1/pass_at_2" in result
    assert result["eval/dataset1/avg_score"] == pytest.approx(0.7)
    assert result["eval/dataset1/pass_at_2"] == 1.0


def test_calculate_per_dataset_metrics_multiple_sources():
    """Test calculate_per_dataset_metrics with multiple data sources including None."""
    # Create test data with mixed sources
    trajectory_batches = {
        "rewards": [0.5, 0.7, 0.9, 0.4],
        "prompt_token_ids": [[1, 2], [3, 4], [5, 6], [7, 8]],
        "response_ids": [[10, 11], [12, 13], [14, 15], [16, 17]],
    }
    uids = ["uid1", "uid2", "uid3", "uid4"]
    data_sources = ["dataset1", None, "dataset1", None]

    result = calculate_per_dataset_metrics(trajectory_batches, uids, data_sources, 2)

    # Verify results for both datasets - actual computed values
    # dataset1: indices 0, 2 -> rewards [0.5, 0.9] -> mean = 0.7, pass@n = 2/2 = 1.0
    # unknown (None): indices 1, 3 -> rewards [0.7, 0.4] -> mean = 0.55, pass@n = 2/2 = 1.0
    assert "eval/dataset1/avg_score" in result
    assert "eval/dataset1/pass_at_2" in result
    assert "eval/unknown/avg_score" in result
    assert "eval/unknown/pass_at_2" in result

    assert result["eval/dataset1/avg_score"] == pytest.approx(0.7)
    assert result["eval/dataset1/pass_at_2"] == 1.0
    assert result["eval/unknown/avg_score"] == pytest.approx(0.55)
    assert result["eval/unknown/pass_at_2"] == 1.0


def test_dump_per_dataset_eval_results_preserves_dataset_and_metrics(tmp_path):
    mock_tokenizer = Mock()
    mock_tokenizer.decode.side_effect = lambda x: f"decoded_{x}"
    trajectory_batches = {
        "prompt_token_ids": [[1, 2], [3, 4], [5, 6]],
        "response_ids": [[10, 11], [12, 13], [14, 15]],
        "rewards": [0.5, 0.7, 0.9],
        "stop_reasons": ["stop1", "stop2", "stop3"],
    }
    data_sources = ["dataset1", None, "dataset1"]
    all_envs = ["env1", "env2", "env3"]
    env_extras = [{"extra1": "val1"}, {"extra2": "val2"}, {"extra3": "val3"}]
    eval_metrics = {"eval/dataset1/avg_score": 0.8, "eval/unknown/avg_score": 0.6}

    dump_per_dataset_eval_results(
        str(tmp_path), mock_tokenizer, trajectory_batches, data_sources, all_envs, env_extras, eval_metrics
    )
    dataset_rows = [json.loads(line) for line in (tmp_path / "dataset1.jsonl").read_text().splitlines()]
    unknown_rows = [json.loads(line) for line in (tmp_path / "unknown.jsonl").read_text().splitlines()]
    assert [row["output_response"] for row in dataset_rows] == ["decoded_[10, 11]", "decoded_[14, 15]"]
    assert unknown_rows[0]["data_source"] == "unknown"
    assert json.loads((tmp_path / "aggregated_results.jsonl").read_text()) == eval_metrics


def test_eval_dump_writes_to_cloud_uri_without_corrupting_scheme(monkeypatch):
    directory = evaluation_dump_dir("s3://bucket/users/exports", 2)
    filesystem = fsspec.filesystem("memory")
    monkeypatch.setattr(io, "open_file", lambda path, mode: filesystem.open(path.removeprefix("s3://"), mode))
    tokenizer = Mock()
    tokenizer.decode.side_effect = lambda tokens: str(tokens)
    batch = {"prompt_token_ids": [[1]], "response_ids": [[2]], "rewards": [1.0]}

    dump_per_dataset_eval_results(directory, tokenizer, batch, ["aime_2024"], ["aime"], [{}], {"accuracy": 1.0})

    expected = "s3://bucket/users/exports/dumped_evals/global_step_2_evals"
    assert directory == expected
    saved = json.loads(filesystem.cat(f"{expected.removeprefix('s3://')}/aime_2024.jsonl"))
    assert saved["output_response"] == "[2]"
    assert json.loads(filesystem.cat(f"{expected.removeprefix('s3://')}/aggregated_results.jsonl")) == {"accuracy": 1.0}


def test_dump_per_dataset_eval_results_preserves_error_disposition(tmp_path):
    tokenizer = Mock()
    tokenizer.decode.side_effect = lambda tokens: str(tokens)
    batch = {
        "prompt_token_ids": [[1], [2]],
        "response_ids": [[3], [4]],
        "rewards": [1.0, -1.0],
        "stop_reasons": ["stop", "error"],
        "exception_types": [None, "TimeoutError"],
        "error_treatments": [None, "mask"],
    }

    dump_per_dataset_eval_results(str(tmp_path), tokenizer, batch, ["aime_2024"] * 2, ["aime"] * 2, [{}, {}], {})

    rows = [json.loads(line) for line in (tmp_path / "aime_2024.jsonl").read_text().splitlines()]
    assert [(row["exception_type"], row["error_treatment"]) for row in rows] == [
        (None, None),
        ("TimeoutError", "mask"),
    ]


def test_filter_trajectory_batch():
    """Test the filter_trajectory_batch utility function."""
    trajectory_batch = {
        "prompt_token_ids": [[1, 2], [3, 4], [5, 6]],
        "response_ids": [[7, 8], [9, 10], [11, 12]],
        "rewards": [1.0, 2.0, 3.0],
        "unshaped_rewards": [0.0, 1.0, 0.0],
        "loss_masks": [[1, 1]] * 3,
        "stop_reasons": ["length", "length", "stop"],
        "rollout_metrics": {"metric": "value"},
        "rollout_logprobs": [[0.16, 0.4], [0.1, 0.2], [0.3, 0.4]],
        "reward_shaping_components": [
            {"non_termination": 0.0, "overlong": 0.0, "successful_length": 0.0},
            {"non_termination": 0.0, "overlong": 0.0, "successful_length": 0.0},
            {"non_termination": 0.0, "overlong": 0.0, "successful_length": 0.0},
        ],
        "reward_shaping_loop_spans": [[{"start": 0, "end": 2}], [], [{"start": 1, "end": 2}]],
        "loop_advantages": [[-0.05, -0.05], [0.0, 0.0], [0.0, -0.2]],
        "reward_shaping_versions": [2, 2, 2],
        "verifier_tests": [
            {"parser": "pytest", "complete": True, "tests": [{"record_id": "a"}]},
            None,
            {"parser": "pytest", "complete": True, "tests": [{"record_id": "c"}]},
        ],
    }
    kept_indices = [0, 2]  # Keep first and third samples

    filtered = filter_trajectory_batch(trajectory_batch, kept_indices)

    assert filtered["prompt_token_ids"] == [[1, 2], [5, 6]]
    assert filtered["response_ids"] == [[7, 8], [11, 12]]
    assert filtered["response_ids"][0] is not trajectory_batch["response_ids"][0]
    assert filtered["rewards"] == [1.0, 3.0]
    assert filtered["unshaped_rewards"] == [0.0, 0.0]
    assert filtered["loss_masks"] == [[1, 1]] * 2
    assert filtered["stop_reasons"] == ["length", "stop"]
    assert filtered["rollout_metrics"]["metric"] == "value"
    assert filtered["rollout_metrics"]["generate/reward_shaping/loop_advantage_mean"] == pytest.approx(-0.15)
    assert filtered["rollout_logprobs"] == [[0.16, 0.4], [0.3, 0.4]]
    assert filtered["reward_shaping_components"] == [
        {"non_termination": 0.0, "overlong": 0.0, "successful_length": 0.0},
        {"non_termination": 0.0, "overlong": 0.0, "successful_length": 0.0},
    ]
    assert filtered["loop_advantages"] == [[-0.05, -0.05], [0.0, -0.2]]
    assert filtered["reward_shaping_loop_spans"] == [
        [{"start": 0, "end": 2}],
        [{"start": 1, "end": 2}],
    ]
    assert filtered["reward_shaping_versions"] == [2, 2]
    assert [collection["tests"][0]["record_id"] for collection in filtered["verifier_tests"]] == ["a", "c"]


def test_validate_trajectory_batch_valid_case():
    """Test validate_trajectory_batch with valid case."""
    input_batch = TrajectoryRequestBatch(
        prompts=["prompt1", "prompt2", "prompt3"],
        env_classes=["env1", "env2", "env3"],
        env_extras=[{"extra": 1}, {"extra": 2}, {"extra": 3}],
        sampling_params={"temperature": 0.7},
    )

    trajectory_batch = TrajectoryBatch(
        prompt_token_ids=[[1, 2, 3, 4], [5, 6], [7, 8, 9]],
        response_ids=[[10, 11, 12], [13, 14], [15]],
        rewards=[[0.5, 0.6, 0.7], [0.8, 0.9], [1.0]],  # Nested list structure
        loss_masks=[[1, 1, 0], [1, 1], [0]],
        stop_reasons=["stop", "length", "stop"],
        rollout_metrics={"metric1": 0.5, "metric2": 0.6},
        rollout_logprobs=None,
    )

    # Should not raise any exceptions
    validate_trajectory_batch(len(input_batch["prompts"]), trajectory_batch)

    # per trajectory rewards should work too
    trajectory_batch["rewards"] = [0.5, 0.6, 0.7]
    validate_trajectory_batch(len(input_batch["prompts"]), trajectory_batch)

    # valid rollout logprobs
    trajectory_batch["rollout_logprobs"] = [[0.11, 0.12, 0.13], [0.2, 0.3], [0.4]]
    validate_trajectory_batch(len(input_batch["prompts"]), trajectory_batch)


def test_validate_trajectory_batch_empty_response_ids():
    """Test validate_trajectory_batch raises RuntimeError when response_ids is empty."""
    input_batch = TrajectoryRequestBatch(
        prompts=["prompt1"], env_classes=["env1"], env_extras=None, sampling_params=None
    )

    trajectory_batch = TrajectoryBatch(
        prompt_token_ids=[[1, 2, 3]],
        response_ids=[],
        rewards=[],
        loss_masks=[],
        stop_reasons=[],
        rollout_logprobs=[],  # Empty response_ids
    )

    with pytest.raises(RuntimeError, match="No outputs generated"):
        validate_trajectory_batch(len(input_batch["prompts"]), trajectory_batch)


def test_validate_trajectory_batch_mismatched_prompts_responses():
    """Test validate_trajectory_batch raises AssertionError when prompts and response_ids lengths don't match."""
    input_batch = TrajectoryRequestBatch(
        prompts=["prompt1", "prompt2", "prompt3"],  # 3 prompts
        env_classes=["env1", "env2", "env3"],
        env_extras=None,
        sampling_params=None,
    )

    trajectory_batch = TrajectoryBatch(
        prompt_token_ids=[[1, 2], [3, 4]],
        response_ids=[[7, 8], [9, 10]],  # Only 2 responses
        rewards=[0.5, 0.7],
        loss_masks=[[1, 1], [1, 0]],
        stop_reasons=["eos", "eos"],
        rollout_logprobs=None,
    )

    with pytest.raises(AssertionError, match=re.escape("Mismatch between prompts (3) and responses (2)")):
        validate_trajectory_batch(len(input_batch["prompts"]), trajectory_batch)


def test_validate_trajectory_batch_all_loss_masked():
    """Test validate_trajectory_batch logs warning when all outputs are loss masked."""
    input_batch = TrajectoryRequestBatch(
        prompts=["prompt1", "prompt2"], env_classes=["env1", "env2"], env_extras=None, sampling_params=None
    )

    trajectory_batch = TrajectoryBatch(
        prompt_token_ids=[[1, 2, 3], [4, 5, 6]],
        response_ids=[[7, 8], [9, 10]],
        rewards=[0.5, 0.7],
        loss_masks=[[0, 0], [0, 0]],  # All zeros - completely loss masked
        stop_reasons=["eos", "eos"],
        rollout_logprobs=None,
    )

    # Capture log output to verify warning is issued
    with patch("skyrl_train.trajectory_runners.trajectory_processing.logger") as mock_logger:
        validate_trajectory_batch(len(input_batch["prompts"]), trajectory_batch)
        mock_logger.warning.assert_called_once_with(
            "All outputs are loss masked, which may lead to NaN loss, please check your generation logic!!"
        )


def test_validate_trajectory_batch_mismatched_list_lengths():
    """Test validate_trajectory_batch rejects mismatched trajectory batch lists."""
    input_batch = TrajectoryRequestBatch(
        prompts=["prompt1", "prompt2"], env_classes=["env1", "env2"], env_extras=None, sampling_params=None
    )

    trajectory_batch = TrajectoryBatch(
        prompt_token_ids=[[1, 2, 3], [4, 5, 6]],
        response_ids=[[7, 8], [9, 10]],  # Length 2
        rewards=[0.5, 0.7, 0.9],  # Length 3 - mismatch!
        loss_masks=[[1, 1], [1, 0]],
        stop_reasons=["eos", "eos"],
        rollout_logprobs=None,
    )

    with pytest.raises(AssertionError, match="Trajectory batch rewards length must equal response_ids length"):
        validate_trajectory_batch(len(input_batch["prompts"]), trajectory_batch)


def test_validate_trajectory_batch_element_length_mismatch():
    """Test validate_trajectory_batch with element length mismatch."""
    input_batch = TrajectoryRequestBatch(
        prompts=["prompt1", "prompt2", "prompt3"],
        env_classes=["env1", "env2", "env3"],
        env_extras=[{"extra": 1}, {"extra": 2}, {"extra": 3}],
        sampling_params={"temperature": 0.7},
    )

    trajectory_batch = TrajectoryBatch(
        prompt_token_ids=[[1, 2, 3, 4], [5, 6], [7, 8, 9]],
        response_ids=[[10, 11, 12], [13, 14], [15]],
        rewards=[[0.5, 0.6, 0.7], [0.8, 0.9], [1.0]],
        loss_masks=[[1, 1], [1], [1, 1]],  # loss masks are not the same length as response ids
        stop_reasons=["stop", "length", "stop"],
        rollout_metrics={"metric1": 0.5, "metric2": 0.6},
        rollout_logprobs=None,
    )

    with pytest.raises(AssertionError, match="Response ids and loss masks must have the same length"):
        validate_trajectory_batch(len(input_batch["prompts"]), trajectory_batch)

    trajectory_batch["loss_masks"] = [[1, 1, 1], [1, 1], [1]]  # add correct loss masks
    trajectory_batch["rewards"] = [[0.5, 0.6], [0.8], [1.0, 2.0]]  # add incorrect rewards
    with pytest.raises(AssertionError, match="Token rewards and response ids must have the same length"):
        validate_trajectory_batch(len(input_batch["prompts"]), trajectory_batch)

    trajectory_batch = TrajectoryBatch(
        prompt_token_ids=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        response_ids=[[7, 8], [9, 10], [11, 12]],
        rewards=[0.5, 0.7, -0.1],
        loss_masks=[[1, 1], [1, 0], [1, 1]],
        stop_reasons=["eos", "eos", "length"],
        rollout_logprobs=[[0.17, 0.2], [0.9], [0.1, 0.2]],  # Second entry has length 1 - mismatch !
    )

    with pytest.raises(AssertionError, match="Response ids and rollout logprobs must have the same length"):
        validate_trajectory_batch(len(input_batch["prompts"]), trajectory_batch)


class MultiItemDataset:
    """Distinct items, so which prompts a loader selects is observable."""

    def __init__(self, size=10):
        self.data = [f"item_{i}" for i in range(size)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        return batch


def test_eval_dataloader_num_prompts_is_bounded_and_reproducible(dummy_config):
    dataset = MultiItemDataset(size=20)

    def selected_prompts(seed):
        config = dummy_config.copy()
        config.trainer.seed = seed
        config.trainer.eval_batch_size = 4
        config.trainer.eval_num_prompts = 6
        config.generator.enable_http_endpoint = True
        return [item for batch in build_eval_dataloader(config, dataset) for item in batch]

    first = selected_prompts(42)

    assert len(first) == 6
    assert len(set(first)) == 6
    assert first == selected_prompts(42)
    assert first != selected_prompts(123)


def test_validate_trajectory_batch_invalid_rewards():
    """Test validate_trajectory_batch raises AssertionError when rewards is neither List[float-like] nor List[List[float-like]]."""
    input_batch = TrajectoryRequestBatch(
        prompts=["prompt1", "prompt2"], env_classes=["env1", "env2"], env_extras=None, sampling_params=None
    )

    trajectory_batch = TrajectoryBatch(
        prompt_token_ids=[[1, 2, 3], [4, 5, 6]],
        response_ids=[[7, 8], [9, 10]],
        rewards=[[0.5, 0.6], 0.7],
        loss_masks=[[1, 1], [1, 0]],
        stop_reasons=["eos", "eos"],
        rollout_logprobs=None,
    )

    with pytest.raises(
        AssertionError,
        match=re.escape("rewards must be `List[float]` or `List[List[float]]`"),
    ):
        validate_trajectory_batch(len(input_batch["prompts"]), trajectory_batch)

    trajectory_batch["rewards"] = [0.5, 0.7]
    validate_trajectory_batch(len(input_batch["prompts"]), trajectory_batch)

    trajectory_batch["rewards"] = [[0.5, 0.6], [0.7, 0.8]]
    validate_trajectory_batch(len(input_batch["prompts"]), trajectory_batch)
