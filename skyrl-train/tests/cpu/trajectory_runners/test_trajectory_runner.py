import pytest

from skyrl_train.trajectory_runners.base import TrajectoryRequestBatch, TrajectoryRunner, TrajectoryBatch
from skyrl_train.trajectory_runners.trajectory_processing import concatenate_trajectory_batches


class _AlignedRunner(TrajectoryRunner):
    async def _run(self, input_batch: TrajectoryRequestBatch, disable_tqdm: bool = False) -> TrajectoryBatch:
        return {
            "prompt_token_ids": [[1]],
            "response_ids": [[2, 3, 4]],
            "rewards": [1.0],
            "loss_masks": [[1, 0, 1]],
            "stop_reasons": ["stop"],
            "rollout_metrics": {"environment/score": 1.0},
            "rollout_logprobs": [[-0.1, 0.0, -0.2]],
        }


class _ReconstructedRunner(TrajectoryRunner):
    async def _run(self, input_batch: TrajectoryRequestBatch, disable_tqdm: bool = False) -> TrajectoryBatch:
        return {
            "prompt_token_ids": [[1]],
            "response_ids": [[2, 3]],
            "rewards": [1.0],
            "loss_masks": [[1, 1]],
            "stop_reasons": ["stop"],
            "rollout_metrics": {
                "generate/tis/aligned_tokens": 2.0,
                "generate/tis/exact_match_fraction": 0.5,
                "generate/tis/lcs_fallback_fraction": 0.5,
                "generate/tis/unaligned_fraction": 0.0,
                "generate/tis/alignment_fail_count": 0.0,
                "generate/tis/lcs_fallback_messages": 1.0,
                "generate/tis/lcs_fallback_alert": 1.0,
            },
            "rollout_logprobs": [[-0.1, -0.2]],
        }


@pytest.mark.asyncio
async def test_run_adds_alignment_health_for_position_aligned_logprobs():
    output = await _AlignedRunner().run({})

    assert output["rollout_metrics"] == {
        "environment/score": 1.0,
        "generate/tis/aligned_tokens": 2.0,
        "generate/tis/exact_match_fraction": 1.0,
        "generate/tis/lcs_fallback_fraction": 0.0,
        "generate/tis/unaligned_fraction": 0.0,
        "generate/tis/alignment_fail_count": 0.0,
        "generate/tis/lcs_fallback_messages": 0.0,
        "generate/tis/lcs_fallback_alert": 0.0,
    }

    concatenated = concatenate_trajectory_batches([output], tis_lcs_alert_threshold=0.005)
    assert concatenated["rollout_metrics"]["generate/tis/exact_match_fraction"] == 1.0
    assert concatenated["rollout_metrics"]["generate/tis/aligned_tokens"] == 2.0


@pytest.mark.asyncio
async def test_run_preserves_measured_reconstruction_alignment_metrics():
    measured_metrics = {
        "generate/tis/aligned_tokens": 2.0,
        "generate/tis/exact_match_fraction": 0.5,
        "generate/tis/lcs_fallback_fraction": 0.5,
        "generate/tis/unaligned_fraction": 0.0,
        "generate/tis/alignment_fail_count": 0.0,
        "generate/tis/lcs_fallback_messages": 1.0,
        "generate/tis/lcs_fallback_alert": 1.0,
    }
    output = await _ReconstructedRunner().run({})

    for name, value in measured_metrics.items():
        assert output["rollout_metrics"][name] == value
