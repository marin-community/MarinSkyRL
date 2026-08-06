"""Execute typed MarinSkyRL jobs and commit their terminal results."""

from __future__ import annotations

import argparse
import contextlib
import json
import posixpath
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

from iris.client import JobFailedError

from cloud.iris.artifacts import (
    CHECKPOINT_MARKER_FILENAME,
    fs_and_path,
    policy_export_uri,
    relative_object_key,
    validate_hf_export,
)
from cloud.iris.runtime_bundle import runtime_bundle_inputs
from cloud.iris.iris_backend import IrisBackend, IrisLaunchOutcome, iris_job_state_name
from cloud.iris.protocol import (
    AttemptState,
    SkyRLJobSpec,
    SkyRLLaunchRequest,
    SkyRLModel,
    SkyRLTerminalResponse,
    job_spec,
)
from cloud.iris.request_builder import build_job_spec


class JobBackend(Protocol):
    """I/O boundary used to submit one prepared Iris request."""

    def validate(self, spec: SkyRLJobSpec, config_path: str) -> None: ...

    def launch(self, spec: SkyRLJobSpec, config_path: str) -> IrisLaunchOutcome: ...


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
    runtime_bundle_inputs(request.runtime.commit)
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

    build_parser = iris_commands.add_parser("build-request", help="Build a SkyRLJobSpec JSON from an RL YAML config.")
    build_parser.add_argument("--config", required=True, help="Path to the RL YAML config.")
    build_parser.add_argument("--run-id", required=True, help="Experiment run identifier.")
    build_parser.add_argument("--model-uri", required=True)
    build_parser.add_argument("--model-identity", required=True)
    build_parser.add_argument("--model-local-path", required=True)
    build_parser.add_argument("--tokenizer-uri", required=True)
    build_parser.add_argument("--tokenizer-revision", required=True)
    build_parser.add_argument(
        "--train-data",
        required=True,
        help='JSON list of data locators, e.g. [{"uri":"s3://...","identity":"...","local_path":"...","relative_path":"train.parquet"}]',
    )
    build_parser.add_argument("--validation-data", default="[]", help="JSON list of validation data locators.")
    build_parser.add_argument("--cluster", required=True)
    build_parser.add_argument("--cluster-config", required=True)
    build_parser.add_argument("--cpu", type=float, required=True)
    build_parser.add_argument("--memory", required=True)
    build_parser.add_argument("--disk", required=True)
    build_parser.add_argument("--gpu-variant", default=None)
    build_parser.add_argument("--target-cluster", default=None)
    build_parser.add_argument("--parent-cluster-config", default=None)
    build_parser.add_argument("--wandb-entity", default=None)
    build_parser.add_argument("--priority", default=None)
    build_parser.add_argument("--max-retries", type=int, default=None)
    build_parser.add_argument("--seed", type=int, default=None)
    build_parser.add_argument("--run-prefix", required=True, help="Canonical output root (e.g. s3://bucket/run-id).")
    build_parser.add_argument("--overrides", default=None, help="JSON list of Hydra ++ override strings.")
    build_parser.add_argument("--attempt-id", default=None)
    build_parser.add_argument("--out", default=None, help="Write JSON to this path; stdout if omitted.")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)

    if args.action == "build-request":
        optional_fields = (
            "gpu_variant",
            "target_cluster",
            "parent_cluster_config",
            "wandb_entity",
            "priority",
            "max_retries",
            "seed",
            "attempt_id",
        )
        build_kwargs = {k: getattr(args, k) for k in optional_fields if getattr(args, k) is not None}
        if args.overrides is not None:
            build_kwargs["overrides"] = json.loads(args.overrides)

        spec = build_job_spec(
            config_path=Path(args.config),
            run_id=args.run_id,
            model_uri=args.model_uri,
            model_identity=args.model_identity,
            model_local_path=args.model_local_path,
            tokenizer_uri=args.tokenizer_uri,
            tokenizer_revision=args.tokenizer_revision,
            train_data=json.loads(args.train_data),
            validation_data=json.loads(args.validation_data),
            cluster=args.cluster,
            cluster_config=args.cluster_config,
            cpu=args.cpu,
            memory=args.memory,
            disk=args.disk,
            run_prefix=args.run_prefix,
            **build_kwargs,
        )
        payload = json.dumps(asdict(spec), indent=2, sort_keys=True)
        if args.out:
            Path(args.out).write_text(payload + "\n")
            print(f"[build-request] wrote {args.out} for run_id={args.run_id}", file=sys.stderr)
        else:
            sys.stdout.write(payload + "\n")
        return 0

    with open(args.request) as source:
        spec = job_spec(json.load(source))
    with contextlib.redirect_stdout(sys.stderr):
        response = execute_job(spec, dry_run=args.dry_run)
    json.dump(asdict(response), sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if response.state in (AttemptState.PREPARED, AttemptState.SUCCEEDED) else 1


if __name__ == "__main__":
    sys.exit(main())
