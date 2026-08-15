from __future__ import annotations

import subprocess
import sys
import json
from pathlib import Path

import pytest

from scripts.iris.jupiter_rl_artifacts import (
    JupiterArtifactSyncResult,
    JupiterJobStatus,
    JupiterRunSpec,
    _remote_exists,
    parse_jupiter_run_spec,
    query_jupiter_job_status,
    sync_jupiter_artifacts,
)
from scripts.iris import watch_coreweave_rl as watcher
from scripts.iris.watch_coreweave_rl import ArtifactResult, Cluster, RlJob, report_row


def test_status_row_surfaces_tis_alignment_and_token_probability_shift(tmp_path: Path) -> None:
    finelog = tmp_path / "finelog.log"
    finelog.write_text(
        "Training Step Progress: 7 / 20\n"
        "WANDB_MIRROR kind=train step=7 metrics="
        '{"policy/policy_entropy": 1.178, "generate/tis/exact_match_fraction": 0.975, '
        '"policy/tis/log_ratio_abs_mean": 0.012, "policy/log_ratio_abs_mean": 0.021, '
        '"policy/log_ratio_abs_p99": 0.44, "policy/log_ratio_abs_max": 3.25}\n'
    )
    job = RlJob(
        cluster=Cluster("jsc-jupiter", Path(), None),
        job_id="1170543",
        state="running",
        submitted_at_ms=0,
        entrypoint="",
    )
    artifacts = ArtifactResult("1 line", "not requested", "not requested", "2 traces", 2, 1, ())

    trend = report_row(job, artifacts, tmp_path)[-1].value

    assert "entropy=1.178" in trend
    assert "TIS exact=0.975" in trend
    assert "TIS |log r|=0.012" in trend
    assert "token |Δlog p| μ/p99/max=0.021/0.44/3.25" in trend


@pytest.mark.parametrize(
    "value",
    [
        "not-a-job=/e/data1/runs/test",
        "1170543=relative/run",
        "1170543=/e/data1/runs/../another-run",
        "1170543=/e/data1/runs/test\nnext-command",
    ],
)
def test_jupiter_run_spec_rejects_ambiguous_or_unsafe_paths(value: str) -> None:
    with pytest.raises(ValueError):
        parse_jupiter_run_spec(value)


def test_jupiter_remote_path_check_uses_bash_compatible_test_syntax(tmp_path: Path) -> None:
    remote_directory = tmp_path / "artifact directory"
    remote_directory.mkdir()
    remote_file = remote_directory / "finelog.out"
    remote_file.touch()

    def run_remote_command(arguments: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["bash", "-c", arguments[-1]], **options)

    assert _remote_exists(run_remote_command, "Jupiter", str(remote_directory), directory=True)
    assert _remote_exists(run_remote_command, "Jupiter", str(remote_file), directory=False)
    assert not _remote_exists(run_remote_command, "Jupiter", str(remote_directory / "missing"), directory=True)


def test_jupiter_artifact_sync_uses_only_explicit_gpfs_subtrees(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    experiment_dir = "/e/data1/experiments/tasktrove-x6"
    trace_root = f"{experiment_dir}/tasktrove-x6/trace_jobs"

    def fake_run(arguments: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        if arguments[0] == "ssh" and "ls -1t" in arguments[-1]:
            return subprocess.CompletedProcess(
                arguments,
                0,
                stdout=f"{experiment_dir}/logs/tasktrove-x6_1170543.out\n",
                stderr="",
            )
        if arguments[0] == "ssh" and "os.scandir" in arguments[-1]:
            return subprocess.CompletedProcess(
                arguments,
                0,
                stdout=f"{trace_root}/trial-new/\n{trace_root}/trial-older/\n",
                stderr="",
            )
        return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")

    result = sync_jupiter_artifacts(
        JupiterRunSpec("1170543", experiment_dir),
        tmp_path,
        host="Jupiter",
        trace_sync_limit=2,
        max_non_log_bytes=100 * 1024 * 1024,
        scope="full",
        finelog_tail_lines=600_000,
        runner=fake_run,
    )

    transferred_sources = [argument for call in calls if call[0] == "rsync" for argument in call]
    assert result.finelog == "synced"
    assert f"Jupiter:{experiment_dir}/logs/tasktrove-x6_1170543.out" in transferred_sources
    assert f"Jupiter:{experiment_dir}/ray_logs/ray_1170543/" in transferred_sources
    assert f"Jupiter:{experiment_dir}/ray_logs/ray_1170543_workers/" in transferred_sources
    assert f"Jupiter:{experiment_dir}/tasktrove-x6/ray_logs/ray_1170543/" in transferred_sources
    assert f"Jupiter:{experiment_dir}/tasktrove-x6/ray_logs/ray_1170543_workers/" in transferred_sources
    assert f"Jupiter:{experiment_dir}/ray_logs/" not in transferred_sources
    assert f"Jupiter:{experiment_dir}/tasktrove-x6/ray_logs/" not in transferred_sources
    assert f"Jupiter:{trace_root}/trial-new/" in transferred_sources
    assert f"Jupiter:{trace_root}/trial-older/" in transferred_sources


def test_jupiter_ray_sync_ignores_untransferable_session_sockets(tmp_path: Path) -> None:
    experiment_dir = "/e/data1/experiments/tasktrove-x6"

    def fake_run(arguments: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if arguments[0] == "ssh" and "ls -1t" in arguments[-1]:
            return subprocess.CompletedProcess(
                arguments,
                0,
                stdout=f"{experiment_dir}/logs/tasktrove-x6_1170543.out\n",
                stderr="",
            )
        if arguments[0] == "rsync" and "/ray_logs/" in " ".join(arguments):
            if "--exclude=sockets/" not in arguments:
                return subprocess.CompletedProcess(
                    arguments,
                    23,
                    stdout="",
                    stderr='rsync: recv_generator: mknod "session_latest/sockets/raylet" failed: File name too long',
                )
        return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")

    result = sync_jupiter_artifacts(
        JupiterRunSpec("1170543", experiment_dir),
        tmp_path,
        host="Jupiter",
        trace_sync_limit=20,
        max_non_log_bytes=100 * 1024 * 1024,
        scope="full",
        finelog_tail_lines=600_000,
        runner=fake_run,
    )

    assert result.ray_logs == "4 directories synced"
    assert result.errors == ()


def test_jupiter_sync_works_when_openrsync_rejects_protect_args(tmp_path: Path) -> None:
    experiment_dir = "/e/data1/experiments/tasktrove-x6"

    def fake_run(arguments: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if arguments[0] == "ssh" and "ls -1t" in arguments[-1]:
            return subprocess.CompletedProcess(
                arguments,
                0,
                stdout=f"{experiment_dir}/logs/tasktrove-x6_1170543.out\n",
                stderr="",
            )
        if arguments[0] == "rsync" and ({"--protect-args", "-s"} & set(arguments)):
            return subprocess.CompletedProcess(
                arguments,
                1,
                stdout="",
                stderr="rsync: unrecognized option `--protect-args'",
            )
        return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")

    result = sync_jupiter_artifacts(
        JupiterRunSpec("1170543", experiment_dir),
        tmp_path,
        host="Jupiter",
        trace_sync_limit=20,
        max_non_log_bytes=100 * 1024 * 1024,
        scope="full",
        finelog_tail_lines=600_000,
        runner=fake_run,
    )

    assert result.finelog == "synced"
    assert result.slurm_logs == "synced"
    assert result.ray_logs == "4 directories synced"
    assert result.errors == ()


def test_jupiter_run_spec_preserves_the_explicit_absolute_experiment_path() -> None:
    assert parse_jupiter_run_spec("1170543=/e/data1/experiments/tasktrove-x6") == JupiterRunSpec(
        "1170543",
        "/e/data1/experiments/tasktrove-x6",
    )


def test_jupiter_status_only_tails_finelog_without_recursive_transfer(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(arguments: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        if "ls -1t" in arguments[-1]:
            stdout = "/e/data1/experiments/tasktrove-x6/logs/tasktrove-x6_1170543.out\n"
        else:
            stdout = "WANDB_MIRROR kind=train step=8 metrics={}\n" if "tail -n" in arguments[-1] else ""
        return subprocess.CompletedProcess(arguments, 0, stdout=stdout, stderr="")

    result = sync_jupiter_artifacts(
        JupiterRunSpec("1170543", "/e/data1/experiments/tasktrove-x6"),
        tmp_path,
        host="Jupiter",
        trace_sync_limit=20,
        max_non_log_bytes=100 * 1024 * 1024,
        scope="status",
        finelog_tail_lines=600_000,
        runner=fake_run,
    )

    assert result.finelog == "current tail (1 line)"
    assert (tmp_path / "finelog.log").read_text().startswith("WANDB_MIRROR")
    assert all(call[0] == "ssh" for call in calls)


def test_jupiter_sync_reports_ssh_transport_failure_instead_of_missing_artifacts(tmp_path: Path) -> None:
    def failed_ssh(arguments: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(arguments, 255, stdout="", stderr="ssh: connect to host failed")

    with pytest.raises(RuntimeError, match="connect to host failed"):
        sync_jupiter_artifacts(
            JupiterRunSpec("1170543", "/e/data1/experiments/tasktrove-x6"),
            tmp_path,
            host="Jupiter",
            trace_sync_limit=20,
            max_non_log_bytes=100 * 1024 * 1024,
            scope="full",
            finelog_tail_lines=600_000,
            runner=failed_ssh,
        )


def test_jupiter_status_queries_only_the_requested_slurm_job() -> None:
    calls: list[list[str]] = []

    def fake_run(arguments: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, stdout="RUNNING|tasktrove-x6\n", stderr="")

    status = query_jupiter_job_status(
        JupiterRunSpec("1170543", "/e/data1/experiments/tasktrove-x6"),
        host="Jupiter",
        runner=fake_run,
    )

    assert status.state == "running"
    assert status.job_name == "tasktrove-x6"
    assert len(calls) == 1
    assert "squeue" in calls[0][-1]
    assert "1170543" in calls[0][-1]


def test_jupiter_only_run_uses_the_shared_report_and_skips_iris_trace_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_dir = "/e/data1/experiments/tasktrove-x6"

    def fake_sync(run: JupiterRunSpec, destination: Path, **_: object) -> JupiterArtifactSyncResult:
        assert run.job_name == "tasktrove-x6"
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "finelog.log").write_text(
            'WANDB_MIRROR kind=train step=9 metrics={"generate/tis/exact_match_fraction": 0.99}\n'
        )
        return JupiterArtifactSyncResult("synced", "synced", "1 directory synced", "newest 1 selected", 1, ())

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "watch_coreweave_rl.py",
            "--bundle-root",
            str(tmp_path),
            "--jupiter-only",
            "--jupiter-run",
            f"1170543={experiment_dir}",
            "--quiet-progress",
        ],
    )
    monkeypatch.setattr(
        watcher,
        "query_jupiter_job_status",
        lambda *_args, **_kwargs: JupiterJobStatus("running", "tasktrove-x6"),
    )
    monkeypatch.setattr(watcher, "sync_jupiter_artifacts", fake_sync)
    monkeypatch.setattr(
        watcher,
        "sync_fleet_trace_jobs",
        lambda *_args, **_kwargs: pytest.fail("Jupiter traces must not enter the Iris object-store fleet sync"),
    )

    assert watcher.main() == 0

    report_dir = tmp_path / "reports" / "rl"
    report_json = json.loads((report_dir / "latest.json").read_text())
    assert "jsc-jupiter/1170543" in report_json["jobs"]
    assert report_json["jobs"]["jsc-jupiter/1170543"]["artifacts"]["traces"] == "newest 1 selected"
    assert report_json["jobs"]["jsc-jupiter/1170543"]["artifacts"]["pod_logs"] == "not applicable (Jupiter)"
    assert report_json["jobs"]["jsc-jupiter/1170543"]["artifacts"]["slurm_logs"] == "synced"
    assert report_json["jobs"]["jsc-jupiter/1170543"]["artifacts"]["trace_started"] is None
    assert report_json["jobs"]["jsc-jupiter/1170543"]["artifacts"]["trace_selected"] == 1
