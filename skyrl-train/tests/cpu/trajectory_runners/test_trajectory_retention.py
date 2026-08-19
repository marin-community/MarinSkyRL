import gzip
import json
from pathlib import Path
import asyncio
import threading
import zipfile

import pytest

from skyrl_train.trajectory_runners.base import (
    BatchMetadata,
    TrajectoryRequestBatch,
    TrajectoryRunner,
    TrajectoryBatch,
    TrajectoryID,
)
from skyrl_train.trajectory_runners.trajectory_retention import (
    TrajectorySink,
    TrajectoryRetentionPublicationError,
    TrajectoryRetentionPublicationTimeout,
    build_trajectory_records,
    parse_trajectory_retention_config,
)
from skyrl_train.trajectory_runners.trajectory_retention_publisher import (
    ProcessTrajectoryPublisher,
    PublicationRequest,
    PublicationResult,
)


class _Tokenizer:
    def decode(self, token_ids, **kwargs):
        return " ".join(str(token_id) for token_id in token_ids)


class _NormalizedRunner(TrajectoryRunner):
    async def _run(self, input_batch, disable_tqdm=False):
        return _output()


class _BlockingPublisher:
    def __init__(self):
        self.pending = False

    def execute(self, request):
        return PublicationResult(request.request_id, request.record_count, ledger=_empty_ledger())

    def submit(self, request):
        self.pending = True
        return True

    def poll(self):
        return None

    def wait_pending(self):
        return None

    def close(self):
        return None


class _FailingPublisher:
    def __init__(self):
        self.result = None

    def execute(self, request):
        if request.operation == "initialize":
            return PublicationResult(request.request_id, request.record_count, ledger=_empty_ledger())
        return PublicationResult(request.request_id, request.record_count, error="OSError: storage unavailable")

    def submit(self, request):
        self.result = PublicationResult(request.request_id, request.record_count, error="OSError: storage unavailable")
        return True

    def poll(self):
        result = self.result
        self.result = None
        return result

    def wait_pending(self):
        return self.poll()

    def close(self):
        return self.poll()


class _TimeoutPublisher(_FailingPublisher):
    def execute(self, request):
        if request.operation == "initialize":
            return PublicationResult(request.request_id, request.record_count, ledger=_empty_ledger())
        return PublicationResult(
            request.request_id,
            request.record_count,
            error="storage operation exceeded 0.05 seconds",
            timed_out=True,
        )


def _empty_ledger():
    return {"schema_version": 1, "total_bytes": 0, "step_bytes": {}, "records": {}, "archives": {}}


def _never_finishes(_request, _sender):
    threading.Event().wait()


def _input(step: int = 7, phase: str = "train") -> TrajectoryRequestBatch:
    return {
        "prompts": [
            [{"role": "user", "content": "first"}],
            [{"role": "user", "content": "second"}],
            [{"role": "user", "content": "third"}],
        ],
        "env_classes": ["math", "math", "math"],
        "env_extras": [{"difficulty": 1}, {"difficulty": 2}, {"difficulty": 3}],
        "sampling_params": {"temperature": 0.7, "max_tokens": 16},
        "trajectory_ids": [
            TrajectoryID(instance_id="a", repetition_id=0),
            TrajectoryID(instance_id="b", repetition_id=0),
            TrajectoryID(instance_id="c", repetition_id=0),
        ],
        "batch_metadata": BatchMetadata(global_step=step, training_phase=phase),
    }


def _output() -> TrajectoryBatch:
    return {
        "prompt_token_ids": [[1], [2], [3]],
        "response_ids": [[10, 11], [20, 21, 22], [30, 31]],
        "rewards": [1.0, -0.25, 0.0],
        "unshaped_rewards": [1.0, 0.0, 0.0],
        "reward_shaping_components": [
            {"non_termination": 0.0, "successful_length": 0.0},
            {"non_termination": -0.25, "successful_length": 0.0},
            {"non_termination": 0.0, "successful_length": 0.0},
        ],
        "reward_shaping_loop_spans": [[], [], [{"start": 0, "end": 2}]],
        "loop_advantages": [[0.0, 0.0], [0.0, 0.0, 0.0], [-0.1, -0.1]],
        "reward_shaping_versions": [2, 2, 2],
        "loss_masks": [[1, 1], [1, 1, 1], [1, 1]],
        "stop_reasons": ["stop", "length", "stop"],
        "rollout_metrics": {"environment/score": 1 / 3},
        "rollout_logprobs": None,
        "trajectory_ids": [
            TrajectoryID(instance_id="a", repetition_id=0),
            TrajectoryID(instance_id="b", repetition_id=0),
            TrajectoryID(instance_id="c", repetition_id=0),
        ],
    }


def _config(output_path: Path, **overrides):
    raw = {
        "enabled": True,
        "output_path": str(output_path),
        "run_id": "run-1",
        "phases": ["train"],
        "sample_count_per_step": 1,
        "sample_fraction": 0.0,
        "always_retain_failures": True,
        "always_retain_non_terminating": True,
        "always_retain_loops": True,
        "max_bytes_per_step": 1_000_000,
        "max_bytes_per_run": 2_000_000,
        "required": True,
        "model_path": "org/model",
        "model_source_identity": "sha256:model",
        "resume_path": "/checkpoints/global_step_6",
        "inference_backend": "vllm",
    }
    raw.update(overrides)
    return parse_trajectory_retention_config(raw)


def _records(output_path: Path) -> list[dict]:
    ledger_path = output_path / "_retention_ledger.json"
    active_ids = None
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
        active_ids = {record_id for record_id, entry in ledger["records"].items() if entry["reasons"]}
    records = []
    for path in sorted(output_path.rglob("*.zip")):
        with zipfile.ZipFile(path) as archive:
            for name in sorted(name for name in archive.namelist() if name.endswith(".json.gz")):
                record = json.loads(gzip.decompress(archive.read(name)))
                if active_ids is None or record["record_id"] in active_ids:
                    records.append(record)
    return records


def _sink(config, publisher=None) -> TrajectorySink:
    sink = TrajectorySink(config, _Tokenizer(), publisher=publisher)
    sink.bind_runner("SkyRLGymTrajectoryRunner")
    return sink


def test_normalized_output_produces_complete_core_trace_schema():
    records = build_trajectory_records(
        _input(),
        _output(),
        _config(Path("/unused")),
        _Tokenizer(),
        runner_name="SkyRLGymTrajectoryRunner",
    )
    record = records[0].to_json()

    assert len(records) == 3
    assert set(record) == {
        "schema_version",
        "record_id",
        "run_id",
        "global_step",
        "phase",
        "trajectory",
        "prompt",
        "response",
        "reward",
        "metrics",
        "provenance",
    }
    assert record["prompt"]["messages"] == [{"role": "user", "content": "first"}]
    assert record["response"]["text"] == "10 11"
    assert record["provenance"]["runner"] == "SkyRLGymTrajectoryRunner"


@pytest.mark.asyncio
async def test_trajectory_runner_finalization_invokes_the_shared_sink(tmp_path):
    trajectory_runner = _NormalizedRunner()
    trajectory_runner.set_trajectory_sink(TrajectorySink(_config(tmp_path), _Tokenizer()))

    output = await trajectory_runner.run(_input())

    assert output["rollout_metrics"]["generate/trajectory_retention/written"] == 3.0
    assert {record["trajectory"]["instance_id"] for record in _records(tmp_path)} == {"a", "b", "c"}
    assert len(list(tmp_path.rglob("*.zip"))) == 1


@pytest.mark.asyncio
async def test_best_effort_retention_does_not_wait_for_blocked_storage(tmp_path):
    publisher = _BlockingPublisher()
    trajectory_runner = _NormalizedRunner()
    trajectory_runner.set_trajectory_sink(
        TrajectorySink(_config(tmp_path, required=False), _Tokenizer(), publisher=publisher)
    )

    output = await asyncio.wait_for(trajectory_runner.run(_input()), 0.1)
    backpressured = await asyncio.wait_for(trajectory_runner.run(_input(step=8)), 0.1)

    assert publisher.pending
    assert output["rollout_metrics"]["generate/trajectory_retention/enqueued"] == 3.0
    assert backpressured["rollout_metrics"]["generate/trajectory_retention/dropped_by_backpressure"] == 3.0


def test_step_wise_rows_form_one_replayable_trajectory_with_explicit_boundaries():
    input_batch = _input()
    for key in ("prompts", "env_classes", "env_extras", "trajectory_ids"):
        input_batch[key] = [input_batch[key][0]]
    trajectory_id = input_batch["trajectory_ids"][0]
    output = _output()
    output.update(
        {
            "prompt_token_ids": [[1], [1]],
            "response_ids": [[10], [20, 21]],
            "rewards": [0.0, 1.0],
            "unshaped_rewards": [0.0, 1.0],
            "reward_shaping_components": None,
            "reward_shaping_loop_spans": None,
            "reward_shaping_versions": None,
            "loss_masks": [[1], [1, 1]],
            "stop_reasons": ["tool", "stop"],
            "trajectory_ids": [trajectory_id, trajectory_id],
            "is_last_step": [False, True],
        }
    )

    record = build_trajectory_records(
        input_batch,
        output,
        _config(Path("/unused")),
        _Tokenizer(),
        runner_name="SkyRLGymTrajectoryRunner",
    )[0].to_json()

    assert record["response"]["token_ids"] == [10, 20, 21]
    assert record["response"]["step_boundaries"] == [
        {"row_index": 0, "token_start": 0, "token_end": 1, "stop_reason": "tool", "is_last_step": False},
        {"row_index": 1, "token_start": 1, "token_end": 3, "stop_reason": "stop", "is_last_step": True},
    ]
    assert record["reward"] == {"outcome": 1.0, "shaped": 1.0, "components": None}


def test_train_phase_retains_sample_and_anomalies(tmp_path):
    sink = _sink(_config(tmp_path))

    metrics = sink.retain(_input(phase="train"), _output())

    records = _records(tmp_path)
    assert {record["trajectory"]["instance_id"] for record in records} == {"a", "b", "c"}
    assert metrics["generate/trajectory_retention/written"] == 3.0
    failed = next(record for record in records if record["trajectory"]["instance_id"] == "b")
    assert failed["reward"]["outcome"] == 0.0
    assert failed["reward"]["shaped"] == -0.25
    assert failed["reward"]["components"]["non_termination"] == -0.25


def test_resume_is_idempotent_and_new_content_appends(tmp_path):
    first_sink = _sink(_config(tmp_path))
    first_sink.retain(_input(), _output())
    original_paths = {path.relative_to(tmp_path) for path in tmp_path.rglob("*.zip")}

    resumed_sink = _sink(_config(tmp_path))
    duplicate_metrics = resumed_sink.retain(_input(), _output())
    changed_output = _output()
    changed_output["response_ids"][1] = [20, 99]
    resumed_sink.retain(_input(), changed_output)

    final_paths = {path.relative_to(tmp_path) for path in tmp_path.rglob("*.zip")}
    assert duplicate_metrics["generate/trajectory_retention/duplicates"] == 3.0
    assert len(final_paths - original_paths) == 1


def test_retention_is_deterministic_and_enforces_compressed_byte_bound_before_write(tmp_path):
    output = _output()
    config = _config(
        tmp_path,
        always_retain_failures=False,
        always_retain_non_terminating=False,
        always_retain_loops=False,
        sample_count_per_step=3,
        max_bytes_per_step=1,
    )

    first = _sink(config).retain(_input(), output)
    second = _sink(config).retain(_input(), output)

    assert first["generate/trajectory_retention/written"] == 0.0
    assert first["generate/trajectory_retention/dropped_by_bounds"] == 3.0
    assert second == first
    assert _records(tmp_path) == []


def test_custom_stop_contract_controls_non_termination_selection(tmp_path):
    output = _output()
    for index in range(3):
        output["rewards"][index] = 1.0
        output["unshaped_rewards"][index] = 1.0
        output["stop_reasons"][index] = "length"
    config = _config(
        tmp_path,
        sample_count_per_step=0,
        always_retain_failures=False,
        always_retain_loops=False,
        accepted_stop_reasons=["length"],
    )

    _sink(config).retain(_input(), output)

    assert _records(tmp_path) == []


def test_reward_extreme_is_retained_after_redaction(tmp_path):
    config = _config(
        tmp_path,
        sample_count_per_step=0,
        always_retain_failures=False,
        always_retain_non_terminating=False,
        always_retain_loops=False,
        reward_above=0.9,
        redact_fields=["prompt.messages"],
    )

    _sink(config).retain(_input(), _output())

    records = _records(tmp_path)
    assert len(records) == 1
    assert all(record["prompt"]["messages"] == "[REDACTED]" for record in records)


def test_sample_count_is_global_and_order_independent_across_async_completions(tmp_path):
    config_overrides = {
        "always_retain_failures": False,
        "always_retain_non_terminating": False,
        "always_retain_loops": False,
        "sample_count_per_step": 1,
    }

    retained_ids = []
    for directory, order in ((tmp_path / "forward", range(3)), (tmp_path / "reverse", reversed(range(3)))):
        sink = _sink(_config(directory, **config_overrides))
        for index in order:
            input_batch = _input()
            output = _output()
            for key in ("prompts", "env_classes", "env_extras", "trajectory_ids"):
                input_batch[key] = [input_batch[key][index]]
            for key in (
                "prompt_token_ids",
                "response_ids",
                "rewards",
                "unshaped_rewards",
                "reward_shaping_components",
                "reward_shaping_loop_spans",
                "reward_shaping_versions",
                "loss_masks",
                "stop_reasons",
                "trajectory_ids",
            ):
                output[key] = [output[key][index]]
            sink.retain(input_batch, output)
        records = _records(directory)
        assert len(records) == 1
        retained_ids.append(records[0]["record_id"])

    assert retained_ids[0] == retained_ids[1]


def test_record_contains_replay_provenance_and_trainable_boundaries():
    record = build_trajectory_records(
        _input(),
        _output(),
        _config(Path("/unused")),
        _Tokenizer(),
        runner_name="SkyRLGymTrajectoryRunner",
    )[0].to_json()

    assert record["provenance"] == {
        "runner": "SkyRLGymTrajectoryRunner",
        "inference_backend": "vllm",
        "model_path": "org/model",
        "model_source_identity": "sha256:model",
        "resume_path": "/checkpoints/global_step_6",
        "model_version_step": 6,
        "sampling": {"temperature": 0.7, "max_tokens": 16},
        "reward_shaping_schema_version": 2,
    }
    assert record["response"]["trainable_spans"] == [{"start": 0, "end": 2}]


def test_best_effort_failure_is_reported_and_required_failure_raises(tmp_path):
    best_effort = _sink(_config(tmp_path, required=False), publisher=_FailingPublisher())
    submitted = best_effort.retain(_input(), _output())
    metrics = best_effort.retain(_input(step=8), _output())

    assert submitted["generate/trajectory_retention/enqueued"] == 3.0
    assert metrics["generate/trajectory_retention/write_errors"] == 6.0
    assert metrics["generate/trajectory_retention/written"] == 0.0

    required = _sink(_config(tmp_path, required=True), publisher=_FailingPublisher())
    with pytest.raises(TrajectoryRetentionPublicationError, match="storage unavailable"):
        required.retain(_input(), _output())

    timed_out = _sink(_config(tmp_path, required=True), publisher=_TimeoutPublisher())
    with pytest.raises(TrajectoryRetentionPublicationTimeout, match=r"\.zip"):
        timed_out.retain(_input(), _output())


def test_disabled_sink_is_a_noop(tmp_path):
    sink = _sink(_config(tmp_path, enabled=False))

    assert sink.retain(_input(), _output()) == {}
    assert list(tmp_path.iterdir()) == []


def test_storage_worker_is_terminated_at_the_publication_deadline():
    publisher = ProcessTrajectoryPublisher(
        _never_finishes,
        publish_timeout_seconds=0.05,
        shutdown_timeout_seconds=0.05,
    )

    result = publisher.execute(PublicationRequest("blocked", "publish", "/unused"))

    assert result.timed_out
    assert result.error is not None


def test_initialization_reconciles_archive_written_before_ledger_commit(tmp_path):
    first_sink = _sink(_config(tmp_path))
    first_sink.retain(_input(), _output())
    (tmp_path / "_retention_ledger.json").unlink()

    resumed_sink = _sink(_config(tmp_path))
    metrics = resumed_sink.retain(_input(), _output())

    assert metrics["generate/trajectory_retention/duplicates"] == 3.0
    assert metrics["generate/trajectory_retention/written"] == 0.0
    assert len(list(tmp_path.rglob("*.zip"))) == 1
