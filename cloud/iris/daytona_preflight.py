"""Prepare the Daytona environments a run will use before allocating GPUs."""

import asyncio
import math
import os
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow.parquet as pq
from harbor_config.models.trial.config import AgentConfig
from daytona import AsyncDaytona, DaytonaConfig, DaytonaNotFoundError, Resources
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from rolloutengine.lowering import SHELLBOX_SESSION
from rolloutengine.spec import LoweredTaskSpec, TaskSessionSpec
from shellbox.backends.daytona.machine import ensure_snapshot
from shellbox.image import DockerfileSource, RegistryImage, image_source_key
from taskcompendium.models import ScriptGrader, Source, VerifyitGrader

from cloud.iris.artifacts import ArtifactSource, fs_and_path, materialize
from marinskyrl.resource_locator import join_resource_path
from marinskyrl.packed_tasks import PackedTaskMaterializer, select_task_references
from marinskyrl.task_sources import DirectoryDataSource, data_source
from skyrl_train.dataset.harbor_import import (
    HARBOR_TRIAL_SESSION,
    TaskConfig,
    harbor_task,
    harbor_trial_task,
    lower_harbor_task,
    restore_harbor_trial_task,
)
from skyrl_train.dataset.agent_profile import task_agent_config
from skyrl_train.rollouts.harbor_settings import EnvironmentConfig

SNAPSHOT_TIMEOUT = 600
STALE_SNAPSHOT_AGE = 3600


@dataclass(frozen=True)
class SnapshotRequirement:
    source: RegistryImage | DockerfileSource
    resources: Resources
    api_key_env: str
    name: str | None
    timeout: float


def _lowered_tasks(config: DictConfig, field: str, root: Path) -> Iterator[LoweredTaskSpec]:
    for index, value in enumerate(config.inputs[field]):
        source = data_source(value)
        uri = join_resource_path(source.uri, source.relative_path)
        if isinstance(source, DirectoryDataSource) and uri.endswith(".parquet"):
            filesystem, path = fs_and_path(uri)
            with filesystem.open(path, "rb") as stream:
                parquet = pq.ParquetFile(stream)
                if "lowered_task_spec" not in parquet.schema_arrow.names:
                    continue
                for batch in parquet.iter_batches(columns=["lowered_task_spec"]):
                    for row in batch.column(0).to_pylist():
                        yield LoweredTaskSpec.model_validate_json(row)
            continue
        session = dict(OmegaConf.to_container(config.skyrl.environment.task_sessions.session, resolve=True))
        installed = OmegaConf.select(config, "skyrl.terminal_bench_config.installed_agent") is not None
        session["task_session"] = HARBOR_TRIAL_SESSION if installed else SHELLBOX_SESSION
        limits = TaskSessionSpec.model_validate(session)

        def lowered(directory: Path) -> LoweredTaskSpec:
            importer = harbor_trial_task if installed else harbor_task
            task = importer(
                directory,
                source=Source(
                    dataset=source.identity,
                    revision=source.identity,
                    row=directory.name,
                    importer_revision="skyrl-harbor-v1",
                ),
            )
            return lower_harbor_task(directory, task, session=limits)

        if isinstance(source, DirectoryDataSource):
            directory = root / field / str(index)
            materialize(ArtifactSource(source.uri, source.identity, str(directory)))
            selected = Path(replace(source, local_path=str(directory)).resolved_path())
            candidates = [selected] if (selected / "instruction.md").is_file() else sorted(selected.iterdir())
            for task_directory in candidates:
                if task_directory.is_dir() and (task_directory / "task.toml").is_file():
                    yield lowered(task_directory)
            continue
        source = replace(source, local_path=source.uri)
        summary = select_task_references(source, dataset_path=source.resolved_path())
        with closing(PackedTaskMaterializer(root / "archives")) as materializer:
            for reference in summary.references:
                directories = materializer.materialize_batch([reference])
                yield lowered(directories[reference])


def snapshot_requirements(config: DictConfig, root: Path) -> Iterator[SnapshotRequirement]:
    """Resolve task images and phase-specific accounts without starting sandboxes."""
    harbor_config = OmegaConf.select(config, "skyrl.terminal_bench_config")
    harbor = None if harbor_config is None else EnvironmentConfig.from_harbor_config(harbor_config.get("harbor", {}))
    mappings = (
        {}
        if harbor_config is None
        else OmegaConf.select(harbor_config, "installed_agent.environment_snapshots", default={}) or {}
    )
    native_key = OmegaConf.select(config, "skyrl.environment.task_sessions.daytona.api_key_env")
    for field, phase in (("train_data", "train"), ("validation_data", "eval")):
        for index, lowered in enumerate(_lowered_tasks(config, field, root)):
            settings = harbor if "harbor" in lowered.task.tags else None
            if lowered.session.task_session == HARBOR_TRIAL_SESSION:
                if settings is None or settings.type != "daytona":
                    continue
                directory = root / field / f"package-{index}"
                restore_harbor_trial_task(lowered.task, directory)
                agent = OmegaConf.to_container(harbor_config.installed_agent, resolve=True).get("agent", {})
                task_agent_config(directory, AgentConfig.model_validate(agent))
                package = TaskConfig.model_validate_toml((directory / "task.toml").read_text())
                environment = package.environment
                context = directory / "environment"
                image = (
                    RegistryImage(environment.docker_image)
                    if environment.docker_image
                    else DockerfileSource(context, context / "Dockerfile")
                )
                name = mappings.get(image_source_key(image))
                if name is None:
                    raise ValueError(f"Harbor environment has no approved snapshot: {image_source_key(image)}")
                resources = Resources(
                    cpu=settings.override_cpus or environment.cpus,
                    memory=math.ceil((settings.override_memory_mb or environment.memory_mb) / 1024)
                    if settings.override_memory_mb or environment.memory_mb
                    else None,
                    disk=math.ceil((settings.override_storage_mb or environment.storage_mb) / 1024)
                    if settings.override_storage_mb or environment.storage_mb
                    else None,
                    gpu=environment.gpus if settings.override_gpus is None else settings.override_gpus,
                )
                key = settings.kwargs.get("eval_api_key_env" if phase == "eval" else "api_key_env")
                key = key or settings.kwargs.get("api_key_env", "DAYTONA_API_KEY")
                yield SnapshotRequirement(image, resources, key, name, environment.build_timeout_sec)
                continue
            pairs = [(lowered.runtime.task_machine, lowered.task.environment_requirements)]
            grader = lowered.task.grader
            if isinstance(grader, (ScriptGrader, VerifyitGrader)) and grader.environment is not None:
                pairs.append((lowered.runtime.verifier_machine, grader.environment))
            for machine, environment in pairs:
                if (
                    machine is not None
                    and machine.backend == "docker"
                    and settings is not None
                    and settings.type == "daytona"
                ):
                    fields = {
                        name: value
                        for name in ("cpus", "memory_mb", "storage_mb", "gpus")
                        if (value := getattr(settings, f"override_{name}")) is not None
                    }
                    fields["backend"] = (
                        "harbor_eval" if phase == "eval" and settings.kwargs.get("eval_api_key_env") else "harbor"
                    )
                    machine = machine.model_copy(update=fields)
                if machine is None or machine.backend not in {"daytona", "harbor", "harbor_eval"}:
                    continue
                if machine.backend == "daytona":
                    key = native_key
                else:
                    assert settings is not None
                    if settings.type != "daytona":
                        continue
                    key = settings.kwargs.get(
                        "eval_api_key_env" if machine.backend == "harbor_eval" else "api_key_env", "DAYTONA_API_KEY"
                    )
                if key is None or environment.docker_image is None:
                    raise ValueError(f"Daytona task {lowered.task.id} needs a credential and registry image")
                yield SnapshotRequirement(
                    RegistryImage(environment.docker_image),
                    Resources(
                        cpu=machine.cpus,
                        memory=None if machine.memory_mb is None else math.ceil(machine.memory_mb / 1024),
                        disk=None if machine.storage_mb is None else math.ceil(machine.storage_mb / 1024),
                        gpu=machine.gpus or None,
                    ),
                    key,
                    None,
                    machine.startup_timeout or SNAPSHOT_TIMEOUT,
                )


async def purge_stale_snapshots(client: AsyncDaytona) -> None:
    """Purge all stale org-owned RL snapshots; unknown last-use times are not evidence."""
    inventory = []
    page = 1
    while True:
        result = await client.snapshot.list(page=page, limit=100)
        inventory.extend(result.items)
        if page >= result.total_pages:
            break
        page += 1
    for original in inventory:
        if original.general or original.last_used_at is None:
            continue
        if (datetime.now(timezone.utc) - original.last_used_at).total_seconds() <= STALE_SNAPSHOT_AGE:
            continue
        current = await client.snapshot.get(original.name)
        if current.id != original.id or current.organization_id != original.organization_id or current.general:
            raise RuntimeError(f"Snapshot identity changed during cleanup: {original.name}")
        if current.last_used_at is None:
            continue
        if (datetime.now(timezone.utc) - current.last_used_at).total_seconds() <= STALE_SNAPSHOT_AGE:
            continue
        logger.info("Deleting stale Daytona RL snapshot {}", current.name)
        await client.snapshot.delete(current)
        async with asyncio.timeout(SNAPSHOT_TIMEOUT):
            while True:
                try:
                    await client.snapshot.get(current.name)
                except DaytonaNotFoundError:
                    break
                await asyncio.sleep(1)


async def prepare_daytona_snapshots(config: DictConfig) -> None:
    """Fail before GPU submission if any required Daytona snapshot cannot become active."""
    with TemporaryDirectory(prefix="skyrl-snapshot-preflight-") as directory:
        requirements = list(snapshot_requirements(config, Path(directory)))
        unique: dict[tuple, SnapshotRequirement] = {}
        aliases = set()
        for requirement in requirements:
            key = (requirement.api_key_env, requirement.name or image_source_key(requirement.source))
            if requirement.name is None:
                key += (repr(requirement.resources),)
            previous = unique.get(key)
            if previous is not None and image_source_key(previous.source) != image_source_key(requirement.source):
                # Several task images can alias an externally prepared common image. Do not
                # rebuild that name from an arbitrary alias if its original recipe is absent.
                aliases.add(key)
                continue
            unique[key] = requirement
        logger.info("Daytona preflight: {} distinct snapshot requirements", len(unique))
        cleaned = set()
        for key, requirement in unique.items():
            async with AsyncDaytona(DaytonaConfig(api_key=os.environ[requirement.api_key_env])) as client:
                # The eval org is exempt from the RL snapshot purge policy.
                if requirement.api_key_env == "DAYTONA_RL_API_KEY" and requirement.api_key_env not in cleaned:
                    await purge_stale_snapshots(client)
                    cleaned.add(requirement.api_key_env)
                logger.info(
                    "Preparing Daytona snapshot {} (credential {})",
                    requirement.name or image_source_key(requirement.source),
                    requirement.api_key_env,
                )
                try:
                    async with asyncio.timeout(requirement.timeout):
                        if key in aliases:
                            try:
                                await client.snapshot.get(requirement.name)
                            except DaytonaNotFoundError as error:
                                raise ValueError(
                                    f"Missing shared snapshot {requirement.name} has several source-image aliases; "
                                    "restore it from its original build recipe, not an arbitrary alias"
                                ) from error
                        name = await ensure_snapshot(
                            client, requirement.source, requirement.resources, name=requirement.name
                        )
                except Exception as error:
                    raise RuntimeError(
                        f"Daytona preflight failed for {requirement.name or image_source_key(requirement.source)} "
                        f"using {requirement.api_key_env}; GPU job was not submitted"
                    ) from error
                logger.info("Daytona snapshot ready: {}", name)
