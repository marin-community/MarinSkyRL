"""Validate, submit, and record one resolved SkyRL launch config."""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from iris.client.client import JobFailedError
from iris.resources.state import JobState
from omegaconf import DictConfig, OmegaConf

from cloud.iris.artifacts import resource_exists, resource_file_names, terminal_checkpoint_step, write_json
from cloud.iris.iris_backend import IrisBackend, IrisLaunchOutcome
from cloud.iris.launch_config import load_launch_config
from cloud.iris.runtime_bundle import runtime_bundle_inputs
from marinskyrl.checkpoint_paths import policy_export_path
from marinskyrl.hf_model import validate_portable_hf_model_files
from marinskyrl.packed_tasks import select_task_references
from marinskyrl.resource_locator import join_resource_path
from marinskyrl.task_sources import TaskTroveParquetSource, TaskTroveSelectionSnapshot, data_source


class LaunchState(StrEnum):
    PREPARED = "prepared"
    SUBMITTED = "submitted"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class ExportedPolicy:
    policy_export_uri: str
    global_step: int
    tokenizer_uri: str
    tokenizer_revision: str
    checkpoint_root: str
    terminal_manifest_uri: str


@dataclass(frozen=True)
class LaunchResult:
    run_id: str
    attempt_id: str
    state: LaunchState
    iris_job_id: str | None
    iris_job_state: str | None
    launcher_commit: str
    runtime_profile: str
    model: ExportedPolicy | None
    failure: str | None


class LaunchBackend(Protocol):
    """I/O boundary for one validated launch config."""

    def validate(self, config_path: Path) -> None: ...

    def launch(self, config_path: Path) -> IrisLaunchOutcome: ...

    def export_terminal_policy(self, config_path: Path) -> None: ...


def _path_exists(uri: str) -> bool:
    return resource_exists(uri)


def _prepared_sources(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared = []
    for value in values:
        source = data_source(value)
        if isinstance(source, TaskTroveParquetSource):
            summary = select_task_references(source)
            source = replace(
                source,
                snapshot=TaskTroveSelectionSnapshot(
                    count=len(summary.references),
                    digest=summary.digest,
                    distinct_environment_count=summary.distinct_environment_count,
                ),
            )
        prepared.append(asdict(source))
    return prepared


def _prepare_config(config: DictConfig) -> DictConfig:
    prepared = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    OmegaConf.set_struct(prepared, False)
    prepared.inputs.train_data = _prepared_sources(list(config.inputs.train_data))
    prepared.inputs.validation_data = _prepared_sources(list(config.inputs.validation_data))
    OmegaConf.set_struct(prepared, True)
    return prepared


def _exported_policy(config: DictConfig) -> ExportedPolicy:
    checkpoint_root = str(config.artifacts.checkpoint_root)
    global_step = terminal_checkpoint_step(checkpoint_root)
    policy_uri = policy_export_path(str(config.artifacts.export_root), global_step)
    names = resource_file_names(policy_uri)
    validate_portable_hf_model_files(names, policy_uri)
    return ExportedPolicy(
        policy_export_uri=policy_uri,
        global_step=global_step,
        tokenizer_uri=str(config.inputs.model.tokenizer_uri),
        tokenizer_revision=str(config.inputs.model.tokenizer_revision),
        checkpoint_root=checkpoint_root,
        terminal_manifest_uri=str(config.artifacts.terminal_manifest_uri),
    )


def _result(
    config: DictConfig,
    state: LaunchState,
    *,
    outcome: IrisLaunchOutcome | None = None,
    model: ExportedPolicy | None = None,
    failure: str | None = None,
) -> LaunchResult:
    return LaunchResult(
        run_id=str(config.run.id),
        attempt_id=str(config.run.attempt_id),
        state=state,
        iris_job_id=outcome.job_id if outcome else None,
        iris_job_state=outcome.job_state if outcome else None,
        launcher_commit=str(config.runtime.launcher_commit),
        runtime_profile=str(config.runtime.profile),
        model=model,
        failure=failure,
    )


def _manifest(config: DictConfig, result: LaunchResult) -> dict[str, Any]:
    return {
        "config": OmegaConf.to_container(config, resolve=True),
        "result": asdict(result),
    }


def _record_failure(
    config: DictConfig,
    outcome: IrisLaunchOutcome,
    failure: str,
) -> LaunchResult:
    result = _result(config, LaunchState.FAILED, outcome=outcome, failure=failure)
    attempt_uri = join_resource_path(str(config.artifacts.attempts_root), f"{config.run.attempt_id}.json")
    write_json(attempt_uri, _manifest(config, result))
    return result


def execute_launch(config_path: Path, *, backend: LaunchBackend | None = None) -> LaunchResult:
    """Execute the lifecycle declared by one resolved launch config."""
    config = _prepare_config(load_launch_config(config_path))
    runtime_bundle_inputs(str(config.runtime.launcher_commit))
    terminal_manifest_uri = str(config.artifacts.terminal_manifest_uri)
    if _path_exists(terminal_manifest_uri):
        raise ValueError(f"Terminal manifest is immutable and already exists: {terminal_manifest_uri}")

    active_backend = backend or IrisBackend()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", encoding="utf-8") as resolved_file:
        OmegaConf.save(config, resolved_file.name, resolve=False)
        resolved_path = Path(resolved_file.name)
        if config.run.submission == "prepare":
            active_backend.validate(resolved_path)
            return _result(config, LaunchState.PREPARED)
        try:
            outcome = active_backend.launch(resolved_path)
        except JobFailedError as error:
            outcome = IrisLaunchOutcome(job_id=str(error.job_id), job_state=error.status.state.value, exit_code=1)

        if outcome.exit_code != 0:
            return _record_failure(config, outcome, f"Iris job reached {outcome.job_state}")
        if config.run.submission == "detach":
            return _result(config, LaunchState.SUBMITTED, outcome=outcome)
        if outcome.job_state != JobState.SUCCEEDED.value:
            return _record_failure(config, outcome, f"Iris job reached {outcome.job_state}")
        if config.run.export_hf:
            try:
                active_backend.export_terminal_policy(resolved_path)
            except (OSError, subprocess.CalledProcessError, ValueError) as error:
                return _record_failure(config, outcome, f"Terminal policy export failed: {error}")

    model = None
    if config.run.export_hf:
        try:
            model = _exported_policy(config)
        except ValueError as error:
            return _record_failure(config, outcome, str(error))
    if not _path_exists(str(config.artifacts.resolved_config_uri)):
        return _record_failure(
            config,
            outcome,
            f"Successful Iris job did not persist resolved config: {config.artifacts.resolved_config_uri}",
        )
    result = _result(config, LaunchState.SUCCEEDED, outcome=outcome, model=model)
    payload = _manifest(config, result)
    attempt_uri = join_resource_path(str(config.artifacts.attempts_root), f"{config.run.attempt_id}.json")
    write_json(attempt_uri, payload)
    write_json(terminal_manifest_uri, payload)
    return result


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch SkyRL from one resolved Hydra config")
    commands = parser.add_subparsers(dest="component", required=True)
    iris = commands.add_parser("iris")
    iris_commands = iris.add_subparsers(dest="action", required=True)
    launch = iris_commands.add_parser("launch")
    launch.add_argument("--config", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    with contextlib.redirect_stdout(sys.stderr):
        result = execute_launch(args.config)
    json.dump(asdict(result), sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if result.state in {LaunchState.PREPARED, LaunchState.SUBMITTED, LaunchState.SUCCEEDED} else 1


if __name__ == "__main__":
    sys.exit(main())
