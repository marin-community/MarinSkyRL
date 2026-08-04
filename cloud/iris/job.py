"""Execute typed MarinSkyRL jobs and commit their terminal results."""

from __future__ import annotations

import argparse
import contextlib
import json
import posixpath
import sys
import tempfile
from dataclasses import asdict
from typing import Any, Protocol

from iris.client import JobFailedError

from cloud.iris.gpu_rl_images import CLUSTER_ARCHITECTURES, GPU_RL_IMAGES, GpuRlImage
from cloud.iris.artifacts import (
    CHECKPOINT_MARKER_FILENAME,
    fs_and_path,
    policy_export_uri,
    relative_object_key,
    validate_hf_export,
)
from cloud.iris.iris_backend import IrisBackend, IrisLaunchOutcome, iris_job_state_name
from cloud.iris.runtime_bundle import launcher_source_at_commit
from cloud.iris.protocol import (
    AttemptState,
    RuntimeIdentity,
    SkyRLJobSpec,
    SkyRLLaunchRequest,
    SkyRLModel,
    SkyRLTerminalResponse,
    job_spec,
)


class JobBackend(Protocol):
    """I/O boundary used to submit one prepared Iris request."""

    def validate(self, spec: SkyRLJobSpec, config_path: str) -> None: ...

    def launch(self, spec: SkyRLJobSpec, config_path: str) -> IrisLaunchOutcome: ...


def _registered_image(runtime: RuntimeIdentity, cluster: str) -> GpuRlImage:
    launcher_source_at_commit(runtime.launcher_commit)
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


def _write_json(uri: str, value: dict[str, Any]) -> None:
    filesystem, path = fs_and_path(uri)
    parent = posixpath.dirname(path)
    if parent:
        filesystem.makedirs(parent, exist_ok=True)
    with filesystem.open(path, "w") as destination:
        json.dump(value, destination, sort_keys=True)


def _read_text(uri: str) -> str:
    filesystem, path = fs_and_path(uri)
    with filesystem.open(path, "r") as source:
        return source.read()


def _path_exists(uri: str) -> bool:
    filesystem, path = fs_and_path(uri)
    return filesystem.exists(path)


def _policy_export(request: SkyRLLaunchRequest) -> SkyRLModel:
    checkpoint_marker = posixpath.join(request.output.checkpoint_root, CHECKPOINT_MARKER_FILENAME)
    if not _path_exists(checkpoint_marker):
        raise ValueError(f"Successful Iris job did not commit a checkpoint marker: {checkpoint_marker}")
    global_step = int(_read_text(checkpoint_marker).strip())
    policy_uri = policy_export_uri(request.output.export_root, global_step)
    filesystem, policy_path = fs_and_path(policy_uri)
    files = sorted(path for path in filesystem.find(policy_path) if not filesystem.isdir(path))
    names = {relative_object_key(policy_path, path) for path in files}
    validate_hf_export(names, policy_uri)
    return SkyRLModel(
        policy_export_uri=policy_uri,
        global_step=global_step,
        tokenizer_uri=request.model.tokenizer_uri,
        tokenizer_revision=request.model.tokenizer_revision,
        checkpoint_root=request.output.checkpoint_root,
        terminal_manifest_uri=request.output.terminal_manifest_uri,
    )


def _attempt_uri(request: SkyRLLaunchRequest) -> str:
    return f"{request.output.attempts_root.rstrip('/')}/{request.attempt_id}.json"


def _manifest_payload(spec: SkyRLJobSpec, response: SkyRLTerminalResponse) -> dict[str, Any]:
    return {
        "request": asdict(spec.request),
        "execution": asdict(spec.execution),
        "response": asdict(response),
    }


def _record_failed_attempt(
    spec: SkyRLJobSpec,
    outcome: IrisLaunchOutcome,
    failure: str,
) -> SkyRLTerminalResponse:
    request = spec.request
    response = SkyRLTerminalResponse(
        run_id=request.run_id,
        attempt_id=request.attempt_id,
        state=AttemptState.FAILED,
        iris_job_id=outcome.job_id,
        iris_job_state=outcome.job_state,
        runtime=request.runtime,
        model=None,
        failure=failure,
    )
    _write_json(_attempt_uri(request), _manifest_payload(spec, response))
    return response


def execute_job(
    spec: SkyRLJobSpec,
    *,
    dry_run: bool = False,
    backend: JobBackend | None = None,
) -> SkyRLTerminalResponse:
    """Validate, submit, monitor, and commit one MarinSkyRL artifact attempt."""
    request = spec.request
    _registered_image(request.runtime, spec.execution.target_cluster or spec.execution.cluster)
    if _path_exists(request.output.terminal_manifest_uri):
        raise ValueError(f"Terminal manifest is immutable and already exists: {request.output.terminal_manifest_uri}")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", encoding="utf-8") as config_file:
        config_file.write(request.config_yaml)
        config_file.flush()
        active_backend = backend or IrisBackend()
        if dry_run:
            active_backend.validate(spec, config_file.name)
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
        try:
            outcome = active_backend.launch(spec, config_file.name)
        except JobFailedError as error:
            job_state = iris_job_state_name(error.status.state)
            outcome = IrisLaunchOutcome(job_id=str(error.job_id), job_state=job_state, exit_code=1)

    if outcome.exit_code != 0:
        return _record_failed_attempt(spec, outcome, f"Iris job reached {outcome.job_state}")

    try:
        model = _policy_export(request)
    except ValueError as error:
        return _record_failed_attempt(spec, outcome, str(error))
    if not _path_exists(request.output.resolved_config_uri):
        return _record_failed_attempt(
            spec,
            outcome,
            f"Successful Iris job did not persist resolved config: {request.output.resolved_config_uri}",
        )
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
    payload = _manifest_payload(spec, response)
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
        spec = job_spec(json.load(source))
    with contextlib.redirect_stdout(sys.stderr):
        response = execute_job(spec, dry_run=args.dry_run)
    json.dump(asdict(response), sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if response.state in (AttemptState.PREPARED, AttemptState.SUCCEEDED) else 1


if __name__ == "__main__":
    sys.exit(main())
