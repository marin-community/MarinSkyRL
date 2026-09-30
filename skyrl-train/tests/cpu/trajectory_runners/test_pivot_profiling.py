"""Frozen profiling preserves repetition identities and rejects missing verdicts."""

import pytest
from omegaconf import OmegaConf
from skyrl_gym.verification import VerificationResult

from skyrl_train.pivot_profiling import profile_candidates
from skyrl_train.trajectory_runners.base import TrajectoryRunner


class ProfilingRunner(TrajectoryRunner):
    def __init__(self, verdicts):
        self.verdicts = verdicts
        self.requests = []
        self.stopped = False

    async def _run(self, request, disable_tqdm=False):
        self.requests.append(request)
        return {
            "prompt_token_ids": [[1]] * 2,
            "response_ids": [[2]] * 2,
            "loss_masks": [[1]] * 2,
            "rewards": [0.0, 1.0],
            "verification_results": self.verdicts,
        }

    async def stop_eval_session(self):
        self.stopped = True


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [False, True])
async def test_profile_streams_all_batches_and_stops_session_on_error(invalid):
    cfg = OmegaConf.create(
        {
            "trainer": {"run_name": "profile"},
            "generator": {
                "backend": "vllm",
                "eval_n_samples_per_prompt": 2,
                "eval_sampling_params": {
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "max_generate_length": 65535,
                    "min_p": 0.0,
                    "logprobs": None,
                },
            },
            "environment": {"env_class": "nemotron_ultra"},
        }
    )
    batches = [
        [
            {
                "prompt": [{"role": "user", "content": "prefix"}],
                "env_class": "nemotron_ultra",
                "env_extras": {},
                "uid": str(i),
            }
        ]
        for i in range(2)
    ]
    runner = ProfilingRunner([None if invalid else VerificationResult.verified(0), VerificationResult.verified(1)])
    if invalid:
        with pytest.raises(ValueError, match="binary verifier"):
            await profile_candidates(batches, runner, cfg)
    else:
        result = await profile_candidates(batches, runner, cfg)
        assert result["profile/actions"] == 4
        assert result["profile/mean_success"] == 0.5
        assert len(runner.requests) == 2
        for request in runner.requests:
            assert [identity.repetition_id for identity in request["trajectory_ids"]] == [0, 1]
            assert request["batch_metadata"].global_step == 0
    assert runner.stopped
