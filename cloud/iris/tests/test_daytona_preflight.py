"""Snapshot launch gates against an in-memory Daytona API."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from daytona import DaytonaNotFoundError
from daytona_api_client_async import SnapshotState
from omegaconf import OmegaConf
from taskcompendium.models import CommandSemantics, EnvironmentRequirements, Source

from cloud.iris.daytona_preflight import prepare_daytona_snapshots, purge_stale_snapshots
from cloud.iris.iris_backend import launch
from skyrl_train.dataset.harbor_import import harbor_trial_task
from tests.cpu.task_specs import lowered_task, text_task


@dataclass
class SnapshotAPI:
    snapshots: dict = field(default_factory=dict)
    build_failure: bool = False

    async def get(self, name):
        if name not in self.snapshots:
            raise DaytonaNotFoundError("missing snapshot", status_code=404)
        return self.snapshots[name]

    async def list(self, page, limit):
        values = list(self.snapshots.values())
        return SimpleNamespace(
            items=values[(page - 1) * limit : page * limit], total_pages=max(1, (len(values) + limit - 1) // limit)
        )

    async def delete(self, snapshot):
        del self.snapshots[snapshot.name]

    async def create(self, params):
        snapshot = SimpleNamespace(
            name=params.name,
            state=SnapshotState.BUILD_FAILED if self.build_failure else SnapshotState.ACTIVE,
            error_reason="registry denied image",
            image=params.image,
            resources=params.resources,
            general=False,
            last_used_at=datetime.now(timezone.utc),
        )
        self.snapshots[params.name] = snapshot
        return snapshot


class DaytonaAPI:
    def __init__(self, snapshots):
        self.snapshot = snapshots

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        pass


def launch_config(tmp_path, *, backend="daytona"):
    package = tmp_path / "package"
    package.mkdir()
    (package / "instruction.md").write_text("Set up the correct branch.")
    (package / "task.toml").write_text(
        '[environment]\ndocker_image = "registry.example/stage1@sha256:abc"\n'
        "cpus = 2\nmemory_mb = 4096\nstorage_mb = 10240\n"
    )
    task = harbor_trial_task(package, source=Source(dataset="stage1", revision="1", row="0", importer_revision="1"))
    lowered = lowered_task(task, "harbor_trial")
    parquet = tmp_path / "tasks.parquet"
    pq.write_table(pa.table({"lowered_task_spec": [lowered.model_dump_json()] * 2}), parquet)
    source = dict(
        kind="directory", uri=str(tmp_path), identity="test-data", local_path=str(tmp_path), relative_path=parquet.name
    )
    return OmegaConf.create(
        {
            "inputs": {"train_data": [source], "validation_data": [source]},
            "skyrl": {
                "terminal_bench_config": {
                    "harbor": {
                        "environment_type": backend,
                        "daytona_api_key_env": "DAYTONA_RL_API_KEY",
                        "eval_daytona_api_key_env": "EVAL_KEY",
                    },
                    "installed_agent": {
                        "environment_snapshots": {"registry:registry.example/stage1@sha256:abc": "stage1-approved"}
                    },
                }
            },
        }
    )


@pytest.mark.asyncio
async def test_launch_prebuilds_named_snapshot_in_both_accounts(tmp_path, monkeypatch):
    config = launch_config(tmp_path)
    accounts = {"rl-secret": SnapshotAPI(), "eval-secret": SnapshotAPI()}
    monkeypatch.setenv("DAYTONA_RL_API_KEY", "rl-secret")
    monkeypatch.setenv("EVAL_KEY", "eval-secret")
    monkeypatch.setattr("cloud.iris.daytona_preflight.AsyncDaytona", lambda cfg: DaytonaAPI(accounts[cfg.api_key]))

    await prepare_daytona_snapshots(config)
    await prepare_daytona_snapshots(config)

    for account in accounts.values():
        snapshot = account.snapshots["stage1-approved"]
        assert len(account.snapshots) == 1
        assert snapshot.state == SnapshotState.ACTIVE
        assert snapshot.image == "registry.example/stage1@sha256:abc"
        assert (snapshot.resources.cpu, snapshot.resources.memory, snapshot.resources.disk) == (2, 4, 10)


@pytest.mark.asyncio
async def test_snapshot_build_failure_aborts_preflight(tmp_path, monkeypatch):
    config = launch_config(tmp_path)
    account = SnapshotAPI(build_failure=True)
    monkeypatch.setenv("DAYTONA_RL_API_KEY", "rl-secret")
    monkeypatch.setattr("cloud.iris.daytona_preflight.AsyncDaytona", lambda cfg: DaytonaAPI(account))

    with pytest.raises(RuntimeError, match="Daytona preflight failed for stage1-approved") as failure:
        await prepare_daytona_snapshots(config)
    assert "registry denied image" in str(failure.value.__cause__)


def test_snapshot_build_failure_prevents_gpu_submission(tmp_path, monkeypatch):
    config = launch_config(tmp_path)
    config.runtime = {"entrypoint": "skyrl_train.entrypoints.terminal_bench"}
    account = SnapshotAPI(build_failure=True)
    monkeypatch.setenv("DAYTONA_RL_API_KEY", "rl-secret")
    monkeypatch.setattr("cloud.iris.daytona_preflight.AsyncDaytona", lambda cfg: DaytonaAPI(account))
    # Git/runtime packaging is a filesystem boundary unrelated to snapshot readiness.
    monkeypatch.setattr("cloud.iris.iris_backend.build_runtime_bundle", lambda _commit: tmp_path)
    submitted = []
    monkeypatch.setattr("cloud.iris.iris_backend.IrisClient", lambda *args, **kwargs: submitted.append(args))

    with pytest.raises(RuntimeError, match="GPU job was not submitted"):
        launch(
            SimpleNamespace(config=config, job_name="snapshot-regression", entrypoint=config.runtime.entrypoint),
            "commit",
        )

    assert submitted == []
    assert account.snapshots["stage1-approved"].state == SnapshotState.BUILD_FAILED


def test_missing_source_prompt_prevents_gpu_submission(tmp_path, monkeypatch):
    monkeypatch.setenv("DAYTONA_RL_API_KEY", "rl-secret")
    config = launch_config(tmp_path)
    package = tmp_path / "package"
    with (package / "task.toml").open("a") as stream:
        stream.write(
            '\n[metadata.agent_profile]\nschema_version = 1\nconfig_file = "missing.yaml"\n'
            f'config_sha256 = "{"a" * 64}"\n'
            '[metadata.agent_profile.agent]\nname = "mini-swe-agent"\n'
        )
    task = harbor_trial_task(package, source=Source(dataset="stage1", revision="2", row="0", importer_revision="2"))
    pq.write_table(
        pa.table({"lowered_task_spec": [lowered_task(task, "harbor_trial").model_dump_json()]}),
        tmp_path / "tasks.parquet",
    )
    config.runtime = {"entrypoint": "skyrl_train.entrypoints.terminal_bench"}
    monkeypatch.setattr("cloud.iris.iris_backend.build_runtime_bundle", lambda _commit: tmp_path)
    submitted = []
    monkeypatch.setattr("cloud.iris.iris_backend.IrisClient", lambda *args, **kwargs: submitted.append(args))
    with pytest.raises(FileNotFoundError, match="missing.yaml"):
        launch(
            SimpleNamespace(config=config, job_name="missing-prompt", entrypoint=config.runtime.entrypoint), "commit"
        )
    assert submitted == []


@pytest.mark.asyncio
async def test_missing_shared_snapshot_does_not_build_from_an_arbitrary_alias(tmp_path, monkeypatch):
    config = launch_config(tmp_path)
    original = pq.read_table(tmp_path / "tasks.parquet")["lowered_task_spec"][0].as_py()
    (tmp_path / "package" / "task.toml").write_text('[environment]\ndocker_image = "another:image"\n')
    task = harbor_trial_task(
        tmp_path / "package", source=Source(dataset="other", revision="1", row="0", importer_revision="1")
    )
    other = lowered_task(task, "harbor_trial")
    pq.write_table(pa.table({"lowered_task_spec": [original, other.model_dump_json()]}), tmp_path / "tasks.parquet")
    config.skyrl.terminal_bench_config.installed_agent.environment_snapshots["registry:another:image"] = (
        "stage1-approved"
    )
    account = SnapshotAPI()
    monkeypatch.setenv("DAYTONA_RL_API_KEY", "rl-secret")
    monkeypatch.setattr("cloud.iris.daytona_preflight.AsyncDaytona", lambda cfg: DaytonaAPI(account))

    with pytest.raises(RuntimeError) as failure:
        await prepare_daytona_snapshots(config)

    assert isinstance(failure.value.__cause__, ValueError)
    assert account.snapshots == {}


@pytest.mark.asyncio
async def test_docker_run_does_not_contact_daytona(tmp_path, monkeypatch):
    config = launch_config(tmp_path, backend="docker")

    def unavailable(_cfg):
        raise AssertionError("A Docker run must not contact Daytona")

    monkeypatch.setattr("cloud.iris.daytona_preflight.AsyncDaytona", unavailable)
    await prepare_daytona_snapshots(config)
    assert config.skyrl.terminal_bench_config.harbor.environment_type == "docker"


@pytest.mark.asyncio
async def test_native_daytona_tasks_prebuild_but_shellsim_tasks_do_not(tmp_path, monkeypatch):
    image = "image@sha256:" + "a" * 64
    task = text_task().model_copy(
        update={
            "environment_requirements": EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=image
            )
        }
    )
    rows = [lowered_task(task, backend=backend).model_dump_json() for backend in ("daytona", "shellsim")]
    parquet = tmp_path / "tasks.parquet"
    pq.write_table(pa.table({"lowered_task_spec": rows}), parquet)
    source = dict(
        kind="directory", uri=str(tmp_path), identity="native", local_path=str(tmp_path), relative_path=parquet.name
    )
    config = OmegaConf.create(
        {
            "inputs": {"train_data": [source], "validation_data": []},
            "skyrl": {"environment": {"task_sessions": {"daytona": {"api_key_env": "NATIVE_KEY"}}}},
        }
    )
    account = SnapshotAPI()
    monkeypatch.setenv("NATIVE_KEY", "native-secret")
    monkeypatch.setattr("cloud.iris.daytona_preflight.AsyncDaytona", lambda cfg: DaytonaAPI(account))

    await prepare_daytona_snapshots(config)

    assert len(account.snapshots) == 1
    assert next(iter(account.snapshots.values())).image == image


@pytest.mark.asyncio
async def test_purge_all_stale_snapshots_across_pages_without_created_time_fallback():
    now = datetime.now(timezone.utc)
    account = SnapshotAPI()
    for index in range(101):
        name = f"snapshot-{index}"
        account.snapshots[name] = SimpleNamespace(
            name=name, id=name, organization_id="rl", general=False, last_used_at=now - timedelta(hours=2)
        )
    account.snapshots["unknown"] = SimpleNamespace(name="unknown", general=False, last_used_at=None)
    account.snapshots["fresh"] = SimpleNamespace(name="fresh", general=False, last_used_at=now)
    account.snapshots["provider"] = SimpleNamespace(
        name="provider", general=True, last_used_at=now - timedelta(days=30)
    )

    await purge_stale_snapshots(DaytonaAPI(account))

    assert set(account.snapshots) == {"unknown", "fresh", "provider"}
