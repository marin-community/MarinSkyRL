from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from cloud.iris import job, runtime_environment  # noqa: E402
from cloud.iris import runtime_bundle  # noqa: E402
from cloud.iris.job import JobBackend, execute_job  # noqa: E402
from cloud.iris.protocol import (  # noqa: E402
    AttemptState,
    DataLocator,
    IrisLaunchOptions,
    ModelLocator,
    RuntimeIdentity,
    SkyRLLaunchRequest,
    SkyRLJobSpec,
    SkyRLOutputPaths,
    SkyRLRolePlan,
    SkyRLTerminalResponse,
    SkyRLTopology,
)
from cloud.iris.iris_backend import IrisLaunchOutcome, create_parser, job_launch_argv  # noqa: E402
from cloud.iris.runtime_environment import RuntimeProfile, task_setup_script  # noqa: E402
from cloud.iris.task_runtime import materialize_model_export  # noqa: E402
from iris.client import JobFailedError  # noqa: E402
from iris.cluster.types import JobName  # noqa: E402
from iris.rpc import job_pb2  # noqa: E402


@dataclass(frozen=True)
class FakeLaunchBackend(JobBackend):
    outcome: IrisLaunchOutcome

    def validate(self, spec: SkyRLJobSpec, config_path: str) -> None:
        assert spec.request.config_yaml
        assert Path(config_path).is_file()

    def launch(self, spec: SkyRLJobSpec, config_path: str) -> IrisLaunchOutcome:
        self.validate(spec, config_path)
        return self.outcome


@dataclass(frozen=True)
class FailedLaunchBackend(JobBackend):
    error: JobFailedError

    def validate(self, spec: SkyRLJobSpec, config_path: str) -> None:
        assert spec.request.config_yaml
        assert Path(config_path).is_file()

    def launch(self, spec: SkyRLJobSpec, config_path: str) -> IrisLaunchOutcome:
        self.validate(spec, config_path)
        raise self.error


def _git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_runtime_files(root: Path, marker: str) -> None:
    runtime_package = root / "cloud" / "iris"
    runtime_package.mkdir(parents=True)
    (runtime_package / "runtime_bundle_files.txt").write_text(
        "cloud/iris/__init__.py\ncloud/iris/task_runtime.py\nchat_templates/delphi_v0.jinja2\n"
    )
    (runtime_package / "__init__.py").write_text("")
    (runtime_package / "task_runtime.py").write_text(f'RUNTIME_MARKER = "{marker}"\n')
    chat_templates = root / "chat_templates"
    chat_templates.mkdir()
    (chat_templates / "delphi_v0.jinja2").write_text(f"{marker.replace('-', ' ')} template\n")


def _runtime_checkout(tmp_path: Path) -> tuple[Path, str]:
    checkout = tmp_path / "checkout"
    _write_runtime_files(checkout, "selected-checkout")
    (checkout / "pyproject.toml").write_text('[project]\nname = "marinskyrl"\nversion = "0.1.0"\n')
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(["git", "config", "user.email", "tests@marin.community"], cwd=checkout, check=True)
    subprocess.run(["git", "config", "user.name", "MarinSkyRL tests"], cwd=checkout, check=True)
    subprocess.run(["git", "add", "."], cwd=checkout, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=checkout, check=True)
    return checkout, _git_commit(checkout)


@pytest.fixture
def runtime_checkout(tmp_path: Path, monkeypatch) -> tuple[Path, str]:
    checkout, commit = _runtime_checkout(tmp_path)

    class Distribution:
        def read_text(self, name: str) -> str:
            assert name == "direct_url.json"
            return json.dumps({"url": checkout.as_uri(), "dir_info": {"editable": True}})

    monkeypatch.setattr(runtime_bundle.importlib.metadata, "distribution", lambda name: Distribution())
    monkeypatch.chdir(tmp_path)
    return checkout, commit


def _cluster_config(tmp_path: Path) -> Path:
    config = tmp_path / "cluster.yaml"
    config.write_text(
        """
name: cw-us-east-08a
storage:
  remote_state_dir: s3://example/iris/state
scale_groups:
  gb200:
    resources:
      device_type: gpu
      device_variant: GB200
      device_count: 4
      cpu: 64
""".strip()
    )
    return config


def _spec(tmp_path: Path) -> SkyRLJobSpec:
    output = tmp_path / "output"
    return SkyRLJobSpec(
        request=SkyRLLaunchRequest(
            run_id="iceball-test",
            attempt_id="attempt-1",
            config_yaml="trainer:\n  strategy: fsdp2\n  placement:\n    colocate_all: true\n",
            runtime=RuntimeIdentity(
                commit=_git_commit(_REPOSITORY_ROOT),
                profile=RuntimeProfile.FSDP,
            ),
            model=ModelLocator(
                uri=(tmp_path / "input-model").as_uri(),
                identity="sft-step@abc123",
                local_path="/tmp/iceball-input-model",
                tokenizer_uri="Qwen/Qwen3-0.6B-Base",
                tokenizer_revision="da87bfb",
            ),
            train_data=(
                DataLocator(
                    uri=(tmp_path / "input-data").as_uri(),
                    identity="gsm8k@e53f048:first-1024",
                    local_path="/tmp/iceball-gsm8k",
                    relative_path="train.parquet",
                ),
            ),
            validation_data=(),
            topology=SkyRLTopology(
                num_nodes=1,
                gpus_per_node=4,
                gpu_variant="GB200",
                role_plan=SkyRLRolePlan(
                    colocate_all=True,
                    policy_num_nodes=1,
                    policy_num_gpus_per_node=4,
                    num_inference_engines=4,
                    inference_engine_tensor_parallel_size=1,
                    train_batch_size=16,
                    policy_mini_batch_size=16,
                    micro_train_batch_size_per_gpu=1,
                    n_samples_per_prompt=4,
                ),
            ),
            output=SkyRLOutputPaths(
                checkpoint_root=(output / "checkpoints").as_uri(),
                export_root=(output / "exports").as_uri(),
                attempts_root=(output / "attempts").as_uri(),
                resolved_config_uri=(output / "resolved-skyrl.json").as_uri(),
                terminal_manifest_uri=(output / "terminal.json").as_uri(),
            ),
            seed=7,
            overrides=("++trainer.max_steps=8",),
        ),
        execution=IrisLaunchOptions(
            cluster="cw-us-east-08a",
            cluster_config=str(_cluster_config(tmp_path)),
            cpu=128,
            memory="800GB",
            disk="4TB",
            target_cluster=None,
            parent_cluster_config=None,
            priority="interactive",
            max_retries=3,
            job_name="iceball-test-attempt-1",
            wandb_entity="marin-community",
        ),
    )


def _write_terminal_training_outputs(envelope: SkyRLJobSpec) -> None:
    output = Path(envelope.request.output.checkpoint_root.removeprefix("file://"))
    output.mkdir(parents=True)
    (output / "latest_ckpt_global_step.txt").write_text("8")
    export = Path(envelope.request.output.export_root.removeprefix("file://")) / "global_step_8" / "policy"
    export.mkdir(parents=True)
    (export / "config.json").write_text("{}")
    (export / "model.safetensors").write_bytes(b"weights")
    (export / "tokenizer.json").write_text("{}")
    resolved = Path(envelope.request.output.resolved_config_uri.removeprefix("file://"))
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text('{"entrypoint":"skyrl_train.entrypoints.main_base","hydra_args":[]}')


def test_execute_job_commits_validated_terminal_model(tmp_path: Path) -> None:
    envelope = _spec(tmp_path)
    _write_terminal_training_outputs(envelope)
    backend = FakeLaunchBackend(
        IrisLaunchOutcome(
            job_id="01KTEST",
            job_state="succeeded",
            exit_code=0,
        )
    )

    response = execute_job(envelope, backend=backend)

    assert response.state == AttemptState.SUCCEEDED
    assert response.model is not None
    assert response.model.global_step == 8
    terminal = json.loads(Path(envelope.request.output.terminal_manifest_uri.removeprefix("file://")).read_text())
    assert terminal["response"] == asdict(response)
    assert terminal["request"]["runtime"] == asdict(envelope.request.runtime)
    attempt = Path(envelope.request.output.attempts_root.removeprefix("file://")) / "attempt-1.json"
    assert json.loads(attempt.read_text()) == terminal


def test_launcher_argv_includes_staged_data_role_plan_and_seed(tmp_path: Path) -> None:
    envelope = _spec(tmp_path)

    argv = job_launch_argv(envelope, "config.yaml")

    assert json.loads(argv[argv.index("--train-data") + 1]) == ["/tmp/iceball-gsm8k/train.parquet"]
    overrides = [argv[index + 1] for index, value in enumerate(argv) if value == "--skyrl-override"]
    assert "++trainer.placement.policy_num_nodes=1" in overrides
    assert "++trainer.placement.ref_num_nodes=1" in overrides
    assert "++trainer.placement.policy_num_gpus_per_node=4" in overrides
    assert "++trainer.placement.ref_num_gpus_per_node=4" in overrides
    assert "++generator.num_inference_engines=4" in overrides
    assert "++trainer.train_batch_size=16" in overrides
    assert "++trainer.seed=7" in overrides


def test_launcher_argv_satisfies_standalone_required_options(tmp_path: Path) -> None:
    argv = job_launch_argv(_spec(tmp_path), "config.yaml")

    args = create_parser().parse_args(argv)

    assert args.rl_config == "config.yaml"
    assert args.model_path == "/tmp/iceball-input-model"
    assert args.cpu == 128
    assert args.memory == "800GB"
    assert args.disk == "4TB"
    assert args.wandb_entity == "marin-community"


def test_launcher_rejects_data_entry_outside_staged_source_root(tmp_path: Path) -> None:
    envelope = _spec(tmp_path)
    locator = DataLocator(
        uri=envelope.request.train_data[0].uri,
        identity="bad",
        local_path="/tmp/iceball-gsm8k",
        relative_path="../escape.parquet",
    )
    envelope = SkyRLJobSpec(
        request=replace(envelope.request, train_data=(locator,)),
        execution=envelope.execution,
    )

    with pytest.raises(ValueError, match="stay below"):
        job_launch_argv(envelope, "config.yaml")


def test_execute_job_failure_records_attempt_without_terminal_model(tmp_path: Path) -> None:
    envelope = _spec(tmp_path)
    backend = FakeLaunchBackend(
        IrisLaunchOutcome(
            job_id="01KFAILED",
            job_state="failed",
            exit_code=1,
        )
    )

    response = execute_job(envelope, backend=backend)

    assert response.state == AttemptState.FAILED
    assert response.model is None
    assert not Path(envelope.request.output.terminal_manifest_uri.removeprefix("file://")).exists()
    attempt = Path(envelope.request.output.attempts_root.removeprefix("file://")) / "attempt-1.json"
    assert json.loads(attempt.read_text())["response"]["iris_job_state"] == "failed"


def test_execute_job_serializes_iris_job_failure(tmp_path: Path) -> None:
    envelope = _spec(tmp_path)
    status = job_pb2.JobStatus(state=job_pb2.JOB_STATE_KILLED, error="Terminated by user")
    backend = FailedLaunchBackend(JobFailedError(JobName.from_string("/power/iceball-test"), status))

    response = execute_job(envelope, backend=backend)

    assert response.state == AttemptState.FAILED
    assert response.iris_job_id == "/power/iceball-test"
    assert response.iris_job_state == "killed"
    assert response.failure == "Iris job reached killed"
    attempt = Path(envelope.request.output.attempts_root.removeprefix("file://")) / "attempt-1.json"
    assert json.loads(attempt.read_text())["response"] == asdict(response)


def test_execute_job_rejects_overwriting_terminal_manifest(tmp_path: Path) -> None:
    envelope = _spec(tmp_path)
    terminal = Path(envelope.request.output.terminal_manifest_uri.removeprefix("file://"))
    terminal.parent.mkdir(parents=True)
    terminal.write_text("{}")

    with pytest.raises(ValueError, match="immutable and already exists"):
        execute_job(envelope, dry_run=True)


def test_materialize_model_export_copies_and_validates_hf_directory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "model.safetensors").write_bytes(b"weights")
    (source / "tokenizer.json").write_text("{}")
    destination = tmp_path / "destination"

    materialize_model_export(source.as_uri(), str(destination), "sft-step@abc123")

    assert (destination / "model.safetensors").read_bytes() == b"weights"
    manifest = json.loads((destination / ".marinskyrl-source.json").read_text())
    assert manifest["source_identity"] == "sft-step@abc123"
    assert {entry["path"] for entry in manifest["files"]} == {
        "config.json",
        "model.safetensors",
        "tokenizer.json",
    }


def test_materialize_model_export_replaces_a_stale_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "model.safetensors").write_bytes(b"new weights")
    (source / "tokenizer.json").write_text("{}")
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "stale.bin").write_bytes(b"old weights")

    materialize_model_export(source.as_uri(), str(destination), "sft-step@new")

    assert not (destination / "stale.bin").exists()
    assert (destination / "model.safetensors").read_bytes() == b"new weights"


def test_runtime_bundle_uses_selected_checkout_when_imported_package_is_stale(
    runtime_checkout: tuple[Path, str],
) -> None:
    _, commit = runtime_checkout
    workspace = runtime_bundle.build_runtime_bundle(commit)

    assert (workspace / "cloud" / "iris" / "task_runtime.py").read_text() == ('RUNTIME_MARKER = "selected-checkout"\n')
    assert (workspace / "chat_templates" / "delphi_v0.jinja2").read_text() == "selected checkout template\n"
    identity = json.loads((workspace / ".marinskyrl-runtime.json").read_text())
    assert identity["launcher_commit"] == commit
    assert {entry["path"] for entry in identity["files"]} == {
        "chat_templates/delphi_v0.jinja2",
        "cloud/iris/__init__.py",
        "cloud/iris/task_runtime.py",
    }
    assert runtime_bundle.validate_bundled_runtime(workspace) == commit


def test_runtime_bundle_uses_files_from_installed_vcs_distribution(tmp_path: Path, monkeypatch) -> None:
    site_packages = tmp_path / "site-packages"
    _write_runtime_files(site_packages, "installed-vcs-distribution")
    commit = "1" * 40

    class Distribution:
        def read_text(self, name: str) -> str:
            assert name == "direct_url.json"
            return json.dumps(
                {
                    "url": "https://github.com/marin-community/MarinSkyRL.git",
                    "vcs_info": {"vcs": "git", "commit_id": commit, "requested_revision": commit},
                }
            )

        def locate_file(self, path: str) -> Path:
            return site_packages / path

    monkeypatch.setattr(runtime_bundle.importlib.metadata, "distribution", lambda name: Distribution())
    monkeypatch.chdir(tmp_path)

    workspace = runtime_bundle.build_runtime_bundle(commit)

    assert (workspace / "cloud" / "iris" / "task_runtime.py").read_text() == (
        'RUNTIME_MARKER = "installed-vcs-distribution"\n'
    )
    assert runtime_bundle.validate_bundled_runtime(workspace) == commit


def test_runtime_bundle_rejects_a_synced_file_that_differs_from_its_identity(
    runtime_checkout: tuple[Path, str],
) -> None:
    _, commit = runtime_checkout
    workspace = runtime_bundle.build_runtime_bundle(commit)
    (workspace / "cloud" / "iris" / "task_runtime.py").write_text('RUNTIME_MARKER = "corrupt"\n')

    with pytest.raises(RuntimeError, match="task_runtime.py"):
        runtime_bundle.validate_bundled_runtime(workspace)


def test_runtime_bundle_rejects_requested_commit_that_differs_from_selected_checkout(
    runtime_checkout: tuple[Path, str],
) -> None:
    with pytest.raises(ValueError, match="does not match requested"):
        runtime_bundle.build_runtime_bundle("0" * 40)


def test_runtime_bundle_rejects_uncommitted_runtime_bytes(runtime_checkout: tuple[Path, str]) -> None:
    checkout, commit = runtime_checkout
    (checkout / "cloud" / "iris" / "task_runtime.py").write_text('RUNTIME_MARKER = "dirty"\n')

    with pytest.raises(RuntimeError, match="task_runtime.py"):
        runtime_bundle.build_runtime_bundle(commit)


def test_cli_reserves_stdout_for_terminal_json(tmp_path: Path, monkeypatch, capsys) -> None:
    envelope = _spec(tmp_path)
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(asdict(envelope)))

    def fake_launch(_spec: SkyRLJobSpec, *, dry_run: bool) -> SkyRLTerminalResponse:
        assert dry_run
        print("human launcher log")
        return SkyRLTerminalResponse(
            run_id=envelope.request.run_id,
            attempt_id=envelope.request.attempt_id,
            state=AttemptState.PREPARED,
            iris_job_id=None,
            iris_job_state=None,
            runtime=envelope.request.runtime,
            model=None,
            failure=None,
        )

    monkeypatch.setattr(job, "execute_job", fake_launch)

    exit_code = job.main(["iris", "launch", "--request", str(request_path), "--dry-run"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert json.loads(captured.out)["state"] == "prepared"
    assert "human launcher log" in captured.err


def test_write_json_supports_a_filename_without_a_parent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    job._write_json("result.json", {"state": "prepared"})

    assert json.loads((tmp_path / "result.json").read_text()) == {"state": "prepared"}


def test_task_setup_executes_the_pinned_checkout_bootstrap(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    bootstrap = source / runtime_environment.MARINSKYRL_BOOTSTRAP_SCRIPT
    bootstrap.parent.mkdir(parents=True)
    bootstrap.write_bytes((_REPOSITORY_ROOT / runtime_environment.MARINSKYRL_BOOTSTRAP_SCRIPT).read_bytes())
    bootstrap.chmod(0o755)
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=source, check=True)
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    subprocess.run(["git", "commit", "-qm", "bootstrap fixture"], cwd=source, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    checkout = tmp_path / "checkout"
    runtime_file = checkout / ".iris-runtime-env"
    environment = tmp_path / "environment"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (environment / "bin").mkdir(parents=True)
    cuda_library_path = tmp_path / "cuda" / "lib"
    cuda_library_path.mkdir(parents=True)
    uv_args = tmp_path / "uv-args"
    fake_uv = fake_bin / "uv"
    fake_uv.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$FAKE_UV_ARGS"\n')
    fake_uv.chmod(0o755)
    fake_python = environment / "bin" / "python"
    fake_python.write_text(
        "#!/bin/sh\n"
        'case "$2" in\n'
        '  *"import site"*) printf "%s\\n" "$FAKE_CUDA_LIBRARY_PATH" ;;\n'
        "  *) exit 0 ;;\n"
        "esac\n"
    )
    fake_python.chmod(0o755)

    monkeypatch.setattr(runtime_environment, "MARINSKYRL_REPOSITORY", str(source))
    monkeypatch.setattr(runtime_environment, "MARINSKYRL_TASK_ROOT", str(checkout))
    monkeypatch.setattr(runtime_environment, "MARINSKYRL_ACTIVATION_FILE", str(runtime_file))
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "IRIS_VENV": str(environment),
        "FAKE_UV_ARGS": str(uv_args),
        "FAKE_CUDA_LIBRARY_PATH": str(cuda_library_path),
    }

    subprocess.run(["bash", "-c", task_setup_script(commit, RuntimeProfile.FSDP)], env=env, check=True)

    assert _git_commit(checkout) == commit
    assert uv_args.read_text().splitlines() == [
        "sync",
        "--project",
        str(checkout),
        "--frozen",
        "--link-mode",
        "symlink",
        "--no-group",
        "dev",
        "--extra",
        "fsdp",
        "--extra",
        "vllm",
        "--extra",
        "telemetry",
    ]
    assert runtime_file.read_text().startswith("export LD_LIBRARY_PATH=")
