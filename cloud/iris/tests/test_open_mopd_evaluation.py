import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import cloud.iris.open_mopd_evaluation as evaluation
from cloud.iris.open_mopd_evaluation_task import (
    EvaluationCommandError,
    EvaluationInputs,
    StagedBenchmark,
    run_logged_command,
    rollout_commands,
    verify_benchmark_file,
)
from cloud.iris.open_mopd_fidelity import load_config
from cloud.iris.open_mopd_fidelity_task import ArtifactVerification

TASK_IMAGE = "registry.example/open-mopd-eval@sha256:" + "1" * 64
OUTPUT_URI = "s3://bucket/open-mopd/final-eval"


def _inputs(config: evaluation.EvaluationConfig) -> EvaluationInputs:
    verification = ArtifactVerification(repository="fixture", revision="1" * 40, files=())
    return EvaluationInputs(
        source=Path("/work/Open-MOPD"),
        model=Path("/work/model"),
        benchmarks=tuple(
            StagedBenchmark(benchmark=benchmark, path=Path("/work/data") / benchmark.path, verification=verification)
            for benchmark in config.benchmarks
        ),
        model_verification=verification,
    )


def _options(command: tuple[str, ...]) -> dict[str, str | bool]:
    options: dict[str, str | bool] = {}
    index = command.index("--output-dir") + 2
    while index < len(command):
        option = command[index]
        if option == "--trust-remote-code":
            options[option] = True
            index += 1
        else:
            options[option] = command[index + 1]
            index += 2
    return options


def test_full_gate_reports_scored_and_rollout_only_coverage() -> None:
    config = evaluation.load_evaluation_config(evaluation.DEFAULT_CONFIG)

    coverage = {item.name: item for item in evaluation.benchmark_coverage(config, "full")}

    assert coverage["aime24"].rollout_rows == 30 * 64
    assert coverage["livecodebench_v6"].rollout_rows == 175 * 10
    assert coverage["ifeval"].score_status == "scored"
    assert coverage["ifbench_test"].score_status == "rollout_only"
    assert coverage["ifbench_test"].note
    assert evaluation.evaluation_scale(config, "full") == (8_101, 230_050_000)


def test_rollout_commands_preserve_released_domain_protocols() -> None:
    config = evaluation.load_evaluation_config(evaluation.DEFAULT_CONFIG)

    commands = rollout_commands(config, _inputs(config), "full", Path("/work/output"), world_size=8)
    math, code, instruction = (_options(command) for command in commands)

    assert math["--temperature"] == "0.6"
    assert math["--n"] == "64"
    assert math["--max-tokens"] == "31000"
    assert math["--gpu-memory-utilization"] == "0.9"
    assert math["--enable-thinking"] == "true"
    assert code["--temperature"] == "1.0"
    assert code["--n"] == "10"
    assert code["--max-tokens"] == "30000"
    assert code["--gpu-memory-utilization"] == "0.85"
    assert code["--enable-thinking"] == "true"
    assert instruction["--n"] == "1"
    assert instruction["--max-tokens"] == "10000"
    assert instruction["--gpu-memory-utilization"] == "0.85"
    assert instruction["--enable-thinking"] == "true"
    for options in (math, code, instruction):
        assert options["--data-parallel-size"] == "8"
        assert options["--max-model-len"] == "32768"
        assert options["--dtype"] == "bfloat16"
        assert options["--top-p"] == "0.95"
        assert options["--top-k"] == "-1"
        assert options["--stop-token-ids"] == "128012"
        assert options["--trust-remote-code"] is True


def test_smoke_gate_bounds_every_domain_without_claiming_comparability() -> None:
    config = evaluation.load_evaluation_config(evaluation.DEFAULT_CONFIG)

    commands = rollout_commands(config, _inputs(config), "smoke", Path("/work/output"), world_size=8)
    coverage = evaluation.benchmark_coverage(config, "smoke")

    assert all(item.rollout_rows == 1 and not item.comparable_to_paper for item in coverage)
    for command in commands:
        options = _options(command)
        assert options["--n"] == "1"
        assert options["--max-tokens"] == "512"
        assert options["--offset"] == "1"


def test_config_cannot_enable_unreleased_code_scorer(tmp_path: Path) -> None:
    raw = json.loads(evaluation.DEFAULT_CONFIG.read_text())
    code = next(item for item in raw["benchmarks"] if item["name"] == "livecodebench_v5")
    code["score_mode"] = "released"
    code["unavailable_reason"] = None
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="released scoring"):
        evaluation.load_evaluation_config(path)


def test_benchmark_verification_rejects_corrupt_download(tmp_path: Path) -> None:
    config = evaluation.load_evaluation_config(evaluation.DEFAULT_CONFIG)
    benchmark = config.benchmarks[0]
    corrupt = tmp_path / "aime24.parquet"
    corrupt.write_bytes(b"not the released parquet")

    with pytest.raises(ValueError, match="integrity mismatch"):
        verify_benchmark_file(benchmark, corrupt)


def test_benchmark_verification_returns_auditable_observation(tmp_path: Path) -> None:
    config = evaluation.load_evaluation_config(evaluation.DEFAULT_CONFIG)
    original = config.benchmarks[0]
    content = b"fixture parquet"
    benchmark = replace(
        original,
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )
    path = tmp_path / "aime24.parquet"
    path.write_bytes(content)

    observed = verify_benchmark_file(benchmark, path)

    assert observed.expected_size == observed.observed_size == len(content)
    assert observed.expected_sha256 == observed.observed_sha256 == benchmark.sha256


def test_failed_evaluation_command_persists_combined_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    log_path = tmp_path / "logs" / "rollout-code.log"
    command = [
        sys.executable,
        "-c",
        "import sys; print('rank stdout'); print('rank stderr', file=sys.stderr); raise SystemExit(7)",
    ]

    with pytest.raises(EvaluationCommandError) as caught:
        run_logged_command(command, cwd=tmp_path, log_path=log_path)

    assert caught.value.returncode == 7
    assert caught.value.log_path == log_path
    assert set(log_path.read_text().splitlines()) == {"rank stdout", "rank stderr"}
    assert set(capsys.readouterr().out.splitlines()) == {"rank stdout", "rank stderr"}


def test_dry_run_exposes_immutable_inputs_without_submitting(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = evaluation.DEFAULT_CONFIG.parents[3]
    monkeypatch.setattr(evaluation, "resolve_launcher_source", lambda: SimpleNamespace(root=root, commit="abc123"))
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("dry run submitted a process"))

    assert (
        evaluation.main(
            [
                "--cluster-config",
                "/tmp/iris.yaml",
                "--output-uri",
                OUTPUT_URI,
                "--task-image",
                TASK_IMAGE,
                "--gpu-slice",
                "H100x8",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    plan = json.loads(output.split("\nuv run", 1)[0])
    fidelity = load_config(root / "cloud/iris/configs/open_mopd_fidelity.json")
    assert plan["source_commit"] == fidelity.source.commit
    assert plan["protocol_source_commit"] == "4460e57ad87fef996a0c21f96bde9a7d1ba029b6"
    assert plan["model_revision"] == fidelity.evaluation_reference.revision
    assert plan["data_revision"] == "9e897efe3257599d4300e2d5ee865a1cc714af87"
    assert plan["planned_completions"] == 6
    assert plan["maximum_output_tokens"] == 3_072
    assert "--no-sync" in plan["iris_command"]
    assert "--no-preemptible" in plan["iris_command"]
    assert "--max-retries" in plan["iris_command"]


def test_submission_requires_omission_acknowledgement(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = evaluation.DEFAULT_CONFIG.parents[3]
    monkeypatch.setattr(evaluation, "resolve_launcher_source", lambda: SimpleNamespace(root=root, commit="abc123"))
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("guard did not stop submission"))

    with pytest.raises(SystemExit, match="allow-known-omissions"):
        evaluation.main(
            [
                "--cluster-config",
                "/tmp/iris.yaml",
                "--output-uri",
                OUTPUT_URI,
                "--task-image",
                TASK_IMAGE,
                "--submit",
            ]
        )
    capsys.readouterr()


@pytest.mark.parametrize("output_uri", ["/tmp/results", "file:///tmp/results", "s3://bucket"])
def test_plan_rejects_non_durable_output(output_uri: str) -> None:
    with pytest.raises(ValueError, match="durable prefix"):
        evaluation.build_plan(
            evaluation.load_evaluation_config(evaluation.DEFAULT_CONFIG),
            config_path=evaluation.DEFAULT_CONFIG,
            gate="smoke",
            cluster_config=Path("/tmp/iris.yaml"),
            output_uri=output_uri,
            task_image=TASK_IMAGE,
        )
