"""Typed machine protocol for Marin-driven MarinSkyRL runs."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import fsspec

from cloud.iris.gpu_rl_images import CLUSTER_ARCHITECTURES, GPU_RL_IMAGES, GpuRlImage
from cloud.iris.launch_rl_iris import IrisLaunchOutcome, launch, resolved_launch_args


class AttemptState(StrEnum):
    PREPARED = "prepared"
    SUBMITTED = "submitted"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class RuntimeIdentity:
    launcher_commit: str
    task_image: str
    trainer_commit: str


@dataclass(frozen=True)
class ModelLocator:
    uri: str
    identity: str
    local_path: str
    tokenizer_uri: str
    tokenizer_revision: str


@dataclass(frozen=True)
class DataLocator:
    uri: str
    identity: str
    local_path: str
    relative_path: str


@dataclass(frozen=True)
class SkyRLTopology:
    num_nodes: int
    gpus_per_node: int
    gpu_variant: str
    role_plan: dict[str, Any]


@dataclass(frozen=True)
class SkyRLOutputPaths:
    checkpoint_root: str
    export_root: str
    attempts_root: str
    resolved_config_uri: str
    terminal_manifest_uri: str


@dataclass(frozen=True)
class SkyRLLaunchRequest:
    run_id: str
    attempt_id: str
    config_yaml: str
    runtime: RuntimeIdentity
    model: ModelLocator
    train_data: tuple[DataLocator, ...]
    validation_data: tuple[DataLocator, ...]
    topology: SkyRLTopology
    output: SkyRLOutputPaths
    seed: int
    overrides: tuple[str, ...]


@dataclass(frozen=True)
class IrisLaunchOptions:
    cluster: str
    cluster_config: str
    cpu: float
    memory: str
    disk: str
    target_cluster: str | None
    parent_cluster_config: str | None
    priority: str
    max_retries: int
    job_name: str


@dataclass(frozen=True)
class ArtifactLaunchEnvelope:
    request: SkyRLLaunchRequest
    execution: IrisLaunchOptions


@dataclass(frozen=True)
class SkyRLModel:
    policy_export_uri: str
    global_step: int
    tokenizer_uri: str
    tokenizer_revision: str
    checkpoint_root: str
    terminal_manifest_uri: str


@dataclass(frozen=True)
class SkyRLTerminalResponse:
    run_id: str
    attempt_id: str
    state: AttemptState
    iris_job_id: str | None
    iris_job_state: str | None
    runtime: RuntimeIdentity
    model: SkyRLModel | None
    failure: str | None


class LaunchBackend(Protocol):
    """I/O boundary used to submit one prepared Iris request."""

    def launch(self, argv: list[str]) -> IrisLaunchOutcome: ...


class IrisLaunchBackend:
    def launch(self, argv: list[str]) -> IrisLaunchOutcome:
        args = resolved_launch_args(argv)
        with contextlib.redirect_stdout(sys.stderr):
            return launch(args)


def _runtime_identity(value: dict[str, Any]) -> RuntimeIdentity:
    return RuntimeIdentity(**value)


def _model_locator(value: dict[str, Any]) -> ModelLocator:
    return ModelLocator(**value)


def _data_locator(value: dict[str, Any]) -> DataLocator:
    return DataLocator(**value)


def _resolved_data_path(locator: DataLocator) -> str:
    relative = Path(locator.relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Data relative_path must stay below its source root: {locator.relative_path!r}")
    return os.path.join(locator.local_path, *relative.parts)


def launch_envelope(value: dict[str, Any]) -> ArtifactLaunchEnvelope:
    """Parse one JSON-compatible launch envelope."""
    request = value["request"]
    return ArtifactLaunchEnvelope(
        request=SkyRLLaunchRequest(
            run_id=request["run_id"],
            attempt_id=request["attempt_id"],
            config_yaml=request["config_yaml"],
            runtime=_runtime_identity(request["runtime"]),
            model=_model_locator(request["model"]),
            train_data=tuple(_data_locator(locator) for locator in request["train_data"]),
            validation_data=tuple(_data_locator(locator) for locator in request["validation_data"]),
            topology=SkyRLTopology(**request["topology"]),
            output=SkyRLOutputPaths(**request["output"]),
            seed=int(request["seed"]),
            overrides=tuple(request.get("overrides", ())),
        ),
        execution=IrisLaunchOptions(**value["execution"]),
    )


def _registered_image(runtime: RuntimeIdentity, cluster: str) -> GpuRlImage:
    launcher_commit = _installed_launcher_commit()
    if launcher_commit != runtime.launcher_commit:
        raise ValueError(
            f"Installed launcher commit {launcher_commit} does not match requested {runtime.launcher_commit}"
        )
    matches = [image for image in GPU_RL_IMAGES.values() if image.reference == runtime.task_image]
    if len(matches) != 1:
        raise ValueError(f"Task image is not registered by this launcher: {runtime.task_image}")
    image = matches[0]
    if image.source_commit != runtime.trainer_commit:
        raise ValueError(
            f"Task image embeds trainer commit {image.source_commit}, not requested {runtime.trainer_commit}"
        )
    architecture = CLUSTER_ARCHITECTURES.get(cluster)
    if architecture is None:
        raise ValueError(f"No GPU architecture is registered for cluster {cluster!r}")
    if image.architecture != architecture:
        raise ValueError(
            f"Task image architecture {image.architecture} is incompatible with cluster {cluster} ({architecture})"
        )
    return image


def _installed_launcher_commit() -> str:
    try:
        direct_url = importlib.metadata.distribution("marinskyrl").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        direct_url = None
    if direct_url:
        commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
        if commit:
            return str(commit)
    repository_root = str(Path(__file__).resolve().parents[2])
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_json(uri: str, value: dict[str, Any]) -> None:
    filesystem, path = fsspec.core.url_to_fs(uri)
    parent = path.rsplit("/", 1)[0]
    if parent:
        filesystem.makedirs(parent, exist_ok=True)
    with filesystem.open(path, "w") as destination:
        json.dump(value, destination, sort_keys=True)


def _read_text(uri: str) -> str:
    with fsspec.open(uri, "r") as source:
        return source.read()


def _path_exists(uri: str) -> bool:
    filesystem, path = fsspec.core.url_to_fs(uri)
    return filesystem.exists(path)


def _policy_export(request: SkyRLLaunchRequest) -> SkyRLModel:
    checkpoint_marker = f"{request.output.checkpoint_root.rstrip('/')}/latest_ckpt_global_step.txt"
    if not _path_exists(checkpoint_marker):
        raise ValueError(f"Successful Iris job did not commit a checkpoint marker: {checkpoint_marker}")
    global_step = int(_read_text(checkpoint_marker).strip())
    policy_uri = f"{request.output.export_root.rstrip('/')}/global_step_{global_step}/policy"
    filesystem, policy_path = fsspec.core.url_to_fs(policy_uri)
    files = sorted(path for path in filesystem.find(policy_path) if not filesystem.isdir(path))
    names = {path.removeprefix(policy_path.rstrip("/") + "/") for path in files}
    if "config.json" not in names:
        raise ValueError(f"Terminal policy export is missing config.json: {policy_uri}")
    if not any(name.endswith((".safetensors", ".bin")) for name in names):
        raise ValueError(f"Terminal policy export has no weight shards: {policy_uri}")
    if not any(name.startswith("tokenizer") or name.endswith(".model") for name in names):
        raise ValueError(f"Terminal policy export has no tokenizer files: {policy_uri}")
    return SkyRLModel(
        policy_export_uri=policy_uri,
        global_step=global_step,
        tokenizer_uri=request.model.tokenizer_uri,
        tokenizer_revision=request.model.tokenizer_revision,
        checkpoint_root=request.output.checkpoint_root,
        terminal_manifest_uri=request.output.terminal_manifest_uri,
    )


def _launcher_argv(envelope: ArtifactLaunchEnvelope, config_path: str) -> list[str]:
    request = envelope.request
    execution = envelope.execution
    data_sources = [asdict(locator) for locator in (*request.train_data, *request.validation_data)]
    role_plan = request.topology.role_plan
    required_role_fields = {
        "colocate_all",
        "policy_num_nodes",
        "policy_num_gpus_per_node",
        "num_inference_engines",
        "inference_engine_tensor_parallel_size",
        "train_batch_size",
        "policy_mini_batch_size",
        "micro_train_batch_size_per_gpu",
        "n_samples_per_prompt",
    }
    missing_role_fields = required_role_fields - set(role_plan)
    if missing_role_fields:
        raise ValueError(f"SkyRL role plan is missing fields: {sorted(missing_role_fields)}")
    role_overrides = (
        f"++trainer.placement.colocate_all={str(role_plan['colocate_all']).lower()}",
        f"++trainer.placement.policy_num_nodes={role_plan['policy_num_nodes']}",
        f"++trainer.placement.policy_num_gpus_per_node={role_plan['policy_num_gpus_per_node']}",
        f"++generator.num_inference_engines={role_plan['num_inference_engines']}",
        f"++generator.inference_engine_tensor_parallel_size={role_plan['inference_engine_tensor_parallel_size']}",
        f"++trainer.train_batch_size={role_plan['train_batch_size']}",
        f"++trainer.policy_mini_batch_size={role_plan['policy_mini_batch_size']}",
        f"++trainer.micro_train_batch_size_per_gpu={role_plan['micro_train_batch_size_per_gpu']}",
        f"++generator.n_samples_per_prompt={role_plan['n_samples_per_prompt']}",
    )
    argv = [
        "--rl_config",
        config_path,
        "--model_path",
        request.model.local_path,
        "--model-source-uri",
        request.model.uri,
        "--model-source-identity",
        request.model.identity,
        "--train-data",
        json.dumps([_resolved_data_path(locator) for locator in request.train_data]),
        "--val-data",
        json.dumps([_resolved_data_path(locator) for locator in request.validation_data]),
        "--data-sources-json",
        json.dumps(data_sources, sort_keys=True),
        "--num-nodes",
        str(request.topology.num_nodes),
        "--gpus-per-node",
        str(request.topology.gpus_per_node),
        "--gpu-variant",
        request.topology.gpu_variant,
        "--cpu",
        str(execution.cpu),
        "--memory",
        execution.memory,
        "--disk",
        execution.disk,
        "--cluster",
        execution.cluster,
        "--cluster-config",
        execution.cluster_config,
        "--task-image",
        request.runtime.task_image,
        "--priority",
        execution.priority,
        "--max-retries",
        str(execution.max_retries),
        "--job-name",
        execution.job_name,
        "--resolved-config-uri",
        request.output.resolved_config_uri,
        "--skyrl-override",
        f"++trainer.ckpt_path={request.output.checkpoint_root}",
        "--skyrl-override",
        f"++trainer.export_path={request.output.export_root}",
        "--skyrl-override",
        "++trainer.resume_mode=latest",
        "--skyrl-override",
        f"++seed={request.seed}",
    ]
    for override in role_overrides:
        argv.extend(["--skyrl-override", override])
    if execution.target_cluster:
        argv.extend(["--target-cluster", execution.target_cluster])
    if execution.parent_cluster_config:
        argv.extend(["--parent-cluster-config", execution.parent_cluster_config])
    for override in request.overrides:
        argv.extend(["--skyrl-override", override])
    return argv


def _attempt_uri(request: SkyRLLaunchRequest) -> str:
    return f"{request.output.attempts_root.rstrip('/')}/{request.attempt_id}.json"


def _manifest_payload(envelope: ArtifactLaunchEnvelope, response: SkyRLTerminalResponse) -> dict[str, Any]:
    return {
        "request": asdict(envelope.request),
        "execution": asdict(envelope.execution),
        "response": asdict(response),
    }


def launch_artifact(
    envelope: ArtifactLaunchEnvelope,
    *,
    dry_run: bool = False,
    backend: LaunchBackend | None = None,
) -> SkyRLTerminalResponse:
    """Validate, submit, monitor, and commit one MarinSkyRL artifact attempt."""
    request = envelope.request
    _registered_image(request.runtime, envelope.execution.target_cluster or envelope.execution.cluster)
    if _path_exists(request.output.terminal_manifest_uri):
        raise ValueError(f"Terminal manifest is immutable and already exists: {request.output.terminal_manifest_uri}")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", encoding="utf-8") as config_file:
        config_file.write(request.config_yaml)
        config_file.flush()
        argv = _launcher_argv(envelope, config_file.name)
        if dry_run:
            resolved_launch_args(argv)
            return SkyRLTerminalResponse(
                run_id=request.run_id,
                attempt_id=request.attempt_id,
                state=AttemptState.PREPARED,
                iris_job_id=None,
                iris_job_state=None,
                runtime=request.runtime,
                model=None,
                failure=None,
            )
        outcome = (backend or IrisLaunchBackend()).launch(argv)

    if outcome.exit_code != 0:
        response = SkyRLTerminalResponse(
            run_id=request.run_id,
            attempt_id=request.attempt_id,
            state=AttemptState.FAILED,
            iris_job_id=outcome.job_id,
            iris_job_state=outcome.job_state,
            runtime=request.runtime,
            model=None,
            failure=f"Iris job reached {outcome.job_state}",
        )
        _write_json(_attempt_uri(request), _manifest_payload(envelope, response))
        return response

    try:
        model = _policy_export(request)
    except ValueError as error:
        response = SkyRLTerminalResponse(
            run_id=request.run_id,
            attempt_id=request.attempt_id,
            state=AttemptState.FAILED,
            iris_job_id=outcome.job_id,
            iris_job_state=outcome.job_state,
            runtime=request.runtime,
            model=None,
            failure=str(error),
        )
        _write_json(_attempt_uri(request), _manifest_payload(envelope, response))
        return response
    if not _path_exists(request.output.resolved_config_uri):
        response = SkyRLTerminalResponse(
            run_id=request.run_id,
            attempt_id=request.attempt_id,
            state=AttemptState.FAILED,
            iris_job_id=outcome.job_id,
            iris_job_state=outcome.job_state,
            runtime=request.runtime,
            model=None,
            failure=f"Successful Iris job did not persist resolved config: {request.output.resolved_config_uri}",
        )
        _write_json(_attempt_uri(request), _manifest_payload(envelope, response))
        return response
    response = SkyRLTerminalResponse(
        run_id=request.run_id,
        attempt_id=request.attempt_id,
        state=AttemptState.SUCCEEDED,
        iris_job_id=outcome.job_id,
        iris_job_state=outcome.job_state,
        runtime=request.runtime,
        model=model,
        failure=None,
    )
    payload = _manifest_payload(envelope, response)
    _write_json(_attempt_uri(request), payload)
    _write_json(request.output.terminal_manifest_uri, payload)
    return response


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MarinSkyRL launch-host protocol")
    subcommands = parser.add_subparsers(dest="component", required=True)
    iris = subcommands.add_parser("iris")
    iris_commands = iris.add_subparsers(dest="action", required=True)
    launch_parser = iris_commands.add_parser("launch")
    launch_parser.add_argument("--request", required=True)
    launch_parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    with open(args.request) as source:
        envelope = launch_envelope(json.load(source))
    with contextlib.redirect_stdout(sys.stderr):
        response = launch_artifact(envelope, dry_run=args.dry_run)
    json.dump(asdict(response), sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if response.state in (AttemptState.PREPARED, AttemptState.SUCCEEDED) else 1


if __name__ == "__main__":
    sys.exit(main())
