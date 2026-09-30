"""Frozen profiling preserves repetitions and continues past retained request errors."""

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
@pytest.mark.parametrize("verdict", [VerificationResult.verified(0), None, VerificationResult.error("HTTP 500")])
async def test_profile_streams_all_batches_and_counts_failed_verdicts(verdict):
    cfg = OmegaConf.create(
        {
            "trainer": {"run_name": "profile"},
            "generator": {
                "backend": "vllm",
                "pivot_profiling_max_retries": 0,
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
    runner = ProfilingRunner([verdict, VerificationResult.verified(1)])
    invalid = verdict is None or verdict.status != "verified"
    result = await profile_candidates(batches, runner, cfg)
    assert result["profile/actions"] == (2 if invalid else 4)
    assert result["profile/attempts"] == 4
    assert result["profile/errors"] == (2 if invalid else 0)
    assert result["profile/mean_success"] == (1.0 if invalid else 0.5)
    for request in runner.requests:
        assert [identity.repetition_id for identity in request["trajectory_ids"]] == [0, 1]
        assert request["batch_metadata"].global_step == 0
    assert runner.stopped


@pytest.mark.asyncio
@pytest.mark.parametrize("all_failed,persistent", [(False, False), (True, False), (False, True)])
async def test_profile_retries_only_failed_samples_with_original_identity(all_failed, persistent):
    cfg = OmegaConf.create(
        {
            "trainer": {"run_name": "profile"},
            "generator": {
                "backend": "vllm",
                "eval_n_samples_per_prompt": 2,
                "pivot_profiling_max_retries": 2,
                "eval_sampling_params": {
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "max_generate_length": 100,
                    "min_p": 0.0,
                    "logprobs": None,
                },
            },
            "environment": {"env_class": "nemotron_ultra"},
        }
    )

    class RecoveringRunner(TrajectoryRunner):
        def __init__(self):
            self.visits = []

        async def _run(self, request, disable_tqdm=False):
            verdicts = []
            for identity, extras in zip(request["trajectory_ids"], request["env_extras"]):
                attempt = extras["extra_info"]["profiling_attempt"]
                self.visits.append((identity.instance_id, identity.repetition_id, attempt))
                if (identity.repetition_id == 0 or all_failed) and (attempt == 0 or persistent):
                    verdicts.append(VerificationResult.error("HTTP 500"))
                else:
                    verdicts.append(VerificationResult.verified(identity.repetition_id))
            size = len(verdicts)
            return {
                "prompt_token_ids": [[1]] * size,
                "response_ids": [[2]] * size,
                "loss_masks": [[1]] * size,
                "rewards": [0.0] * size,
                "verification_results": verdicts,
            }

    runner = RecoveringRunner()
    prompts = [
        {
            "prompt": [{"role": "user", "content": "prefix"}],
            "env_class": "nemotron_ultra",
            "env_extras": {"extra_info": {"source_id": "source"}},
            "uid": "row",
        }
    ]
    result = await profile_candidates([prompts], runner, cfg)
    failures = [0, 1] if all_failed else [0]
    expected_visits = [("row", 0, 0), ("row", 1, 0)] + [("row", rep, 1) for rep in failures]
    if persistent:
        expected_visits += [("row", rep, 2) for rep in failures]
    assert runner.visits == expected_visits
    assert result["profile/actions"] == (1 if persistent else 2)
    assert result["profile/attempts"] == len(expected_visits)
    assert result["profile/errors"] == (3 if persistent else len(failures))
    assert result["profile/recovered_actions"] == (0 if persistent else len(failures))
    assert result["profile/unresolved_actions"] == int(persistent)
    assert result["profile/mean_success"] == (1.0 if persistent else 0.5)
    assert result["profile/prompt_tokens"] == result["profile/generated_tokens"] == len(expected_visits)
    assert "profiling_attempt" not in prompts[0]["env_extras"]["extra_info"]
