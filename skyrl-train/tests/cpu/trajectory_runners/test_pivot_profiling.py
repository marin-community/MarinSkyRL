"""Frozen profiling preserves repetitions and continues past retained request errors."""

import asyncio
import pytest
import os
import gzip
import json
import threading
from zipfile import ZipFile
from omegaconf import OmegaConf
from skyrl_gym.verification import VerificationResult

from skyrl_train.pivot_profiling import profile_candidates
from skyrl_train.trajectory_runners.base import TrajectoryRunner
from skyrl_train.trajectory_runners.trajectory_processing import prepare_trajectory_request, select_request_rows
from skyrl_train.trajectory_runners.trajectory_retention import TrajectorySink, parse_trajectory_retention_config
from marinskyrl.pivot_history import load_profile_history
import marinskyrl.pivot_history as pivot_history


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


@pytest.mark.asyncio
async def test_profile_resume_skips_saved_samples_and_continues_retry_history(tmp_path):
    retention = {
        "enabled": True,
        "required": True,
        "output_path": str(tmp_path),
        "run_id": "profile",
        "sample_fraction": 1.0,
        "phases": ["eval"],
        "max_bytes_per_run": None,
        "max_bytes_per_step": None,
        "model_path": "org/policy",
        "model_source_identity": "revision",
    }
    cfg = OmegaConf.create(
        {
            "trainer": {"run_name": "profile", "policy": {"model": {"path": "org/policy", "revision": "revision"}}},
            "generator": {
                "backend": "vllm",
                "pivot_profiling_resume": True,
                "pivot_profiling_max_retries": 2,
                "eval_n_samples_per_prompt": 2,
                "trajectory_retention": retention,
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

    class Tokenizer:
        def decode(self, tokens, **kwargs):
            return " ".join(map(str, tokens))

    class ResumeRunner(TrajectoryRunner):
        def __init__(self, verdicts=None):
            self.verdicts = verdicts or {}
            self.visits = []

        async def _run(self, request, disable_tqdm=False):
            results = []
            for identity, extras in zip(request["trajectory_ids"], request["env_extras"]):
                self.visits.append((identity.repetition_id, extras["extra_info"]["profiling_attempt"]))
                results.append(self.verdicts.get(identity.repetition_id, VerificationResult.verified(1)))
            size = len(results)
            return {
                "prompt_token_ids": [[1]] * size,
                "response_ids": [[2]] * size,
                "loss_masks": [[1]] * size,
                "rewards": [result.score or 0 for result in results],
                "verification_results": results,
            }

    prompts = [
        {
            "prompt": [{"role": "user", "content": "prefix"}],
            "uid": "row",
            "env_class": "nemotron_ultra",
            "env_extras": {"extra_info": {"source_id": "source", "profiling_attempt": 0}},
        }
    ]
    request, _ = prepare_trajectory_request(prompts, 2, {}, "nemotron_ultra", "eval", 0)
    sink = TrajectorySink(parse_trajectory_retention_config(retention), Tokenizer())
    original = ResumeRunner({1: VerificationResult.error("HTTP 500")})
    original.set_trajectory_sink(sink)
    await original.run(request)
    first = next(tmp_path.rglob("*.zip"))
    os.utime(first, (1, 1))
    # An old preemption replay must not replace the first committed success with failure.
    replay = ResumeRunner({0: VerificationResult.verified(0)})
    replay.set_trajectory_sink(sink)
    await replay.run(select_request_rows(request, [0]))
    for archive in tmp_path.rglob("*.zip"):
        if archive != first:
            os.utime(archive, (2, 2))

    resumed = ResumeRunner()
    resumed.set_trajectory_sink(sink)
    try:
        metrics = await profile_candidates([prompts], resumed, cfg)
        assert resumed.visits == [(1, 1)]
        assert metrics["profile/actions"] == 2
        assert metrics["profile/mean_success"] == 1
        assert metrics["profile/errors"] == 1
        assert metrics["profile/recovered_actions"] == 1
        assert metrics["profile/replayed_actions_discarded"] == 1
        assert metrics["profile/attempts"] == 4
        assert metrics["profile/prompt_tokens"] == metrics["profile/generated_tokens"] == 4
        again = ResumeRunner()
        again.set_trajectory_sink(sink)
        restored = await profile_candidates([prompts], again, cfg)
        assert again.visits == []
        assert restored["profile/actions"] == 2
        assert restored["profile/generated_tokens"] == 4
    finally:
        sink.close()
    history = load_profile_history(str(tmp_path), "org/policy", "revision")
    assert len(history.samples) == 3
    with pytest.raises(ValueError, match="same frozen initial policy"):
        load_profile_history(str(tmp_path), "org/policy", "another-revision")


@pytest.mark.parametrize("first_score", [0, 1])
def test_parallel_history_keeps_first_commit_when_later_archive_finishes_first(tmp_path, monkeypatch, first_score):
    for order, score in enumerate((first_score, 1 - first_score), start=1):
        record = {
            "record_id": str(order),
            "phase": "eval",
            "global_step": 0,
            "provenance": {
                "model_path": "org/policy",
                "model_source_identity": "revision",
                "model_version_step": 0,
                "resume_path": None,
            },
            "trajectory": {
                "repetition_id": 0,
                "environment_extras": {"extra_info": {"source_id": "source", "profiling_attempt": 0}},
            },
            "verification_result": {"status": "verified", "score": score},
            "disposition": {"exception_type": None},
            "reward": {"outcome": score, "shaped": score},
            "prompt_tokens": 2,
            "response_tokens": 1,
        }
        path = tmp_path / f"{order}.zip"
        with ZipFile(path, "w") as archive:
            archive.writestr("manifest.json", json.dumps({"records": [{"record_id": str(order), "profiling": record}]}))
        os.utime(path, (order, order))

    later_finished = threading.Event()
    read_archive = pivot_history._read_archive

    def read_later_first(filesystem, path):
        if path.endswith("1.zip"):
            assert later_finished.wait(timeout=5), "History reads did not overlap"
        records = read_archive(filesystem, path)
        if path.endswith("2.zip"):
            later_finished.set()
        return records

    monkeypatch.setattr(pivot_history, "_read_archive", read_later_first)
    history = load_profile_history(str(tmp_path), "org/policy", "revision", read_concurrency=2)
    assert history.samples[("source", 0, 0)]["verification_result"]["score"] == first_score
    assert history.discarded_record_ids == ["2"]
    assert history.attempts == 2
    assert history.prompt_tokens == 4
    assert history.response_tokens == 2


def test_profile_history_reads_existing_archives_without_compact_index(tmp_path):
    record = {
        "record_id": "original",
        "phase": "eval",
        "global_step": 0,
        "provenance": {
            "model_path": "org/policy",
            "model_source_identity": "revision",
            "model_version_step": 0,
            "resume_path": None,
        },
        "trajectory": {
            "repetition_id": 0,
            "environment_extras": {"extra_info": {"source_id": "source", "profiling_attempt": 0}},
        },
        "verification_result": {"status": "verified", "score": 1},
        "disposition": {"exception_type": None},
        "reward": {"outcome": 1, "shaped": 1},
        "prompt": {"token_ids": [1, 2]},
        "response": {"token_ids": [3]},
    }
    with ZipFile(tmp_path / "original.zip", "w") as archive:
        archive.writestr("record.json.gz", gzip.compress(json.dumps(record).encode()))
        archive.writestr(
            "manifest.json", json.dumps({"records": [{"record_id": "original", "entry": "record.json.gz"}]})
        )
    history = load_profile_history(str(tmp_path), "org/policy", "revision")
    assert set(history.samples) == {("source", 0, 0)}
    assert history.samples[("source", 0, 0)]["reward"]["outcome"] == 1
    assert history.prompt_tokens == 2
    assert history.response_tokens == 1
    assert history.attempts == 1


@pytest.mark.asyncio
async def test_rolling_profile_refills_while_an_earlier_group_is_slow():
    """A slow group must not prevent other rows from entering the bounded queue."""
    cfg = OmegaConf.create(
        {
            "trainer": {"run_name": "profile", "rollout_buffer": {"batch_policy": "rolling", "max_in_flight": 2}},
            "generator": {
                "backend": "vllm",
                "pivot_profiling_max_retries": 0,
                "eval_n_samples_per_prompt": 2,
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
    release = asyncio.Event()
    started = []

    class SlowRunner(ProfilingRunner):
        async def _run(self, request, disable_tqdm=False):
            uid = request["trajectory_ids"][0].instance_id
            started.append(uid)
            if uid == "0":
                await release.wait()
            if uid == "2":
                assert not release.is_set()
                release.set()
            assert [item.repetition_id for item in request["trajectory_ids"]] == [0, 1]
            return await super()._run(request, disable_tqdm)

    rows = [
        {
            "prompt": [{"role": "user", "content": "prefix"}],
            "env_class": "nemotron_ultra",
            "env_extras": {},
            "uid": str(i),
        }
        for i in range(3)
    ]
    runner = SlowRunner([VerificationResult.verified(0), VerificationResult.verified(1)])
    async with asyncio.timeout(5):
        result = await profile_candidates([rows], runner, cfg)
    assert started == ["0", "1", "2"]
    assert result["profile/actions"] == result["profile/attempts"] == 6
    assert result["profile/mean_success"] == 0.5
    assert runner.stopped
