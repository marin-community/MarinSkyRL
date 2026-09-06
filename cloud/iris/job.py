"""Prepare, submit, and optionally monitor typed MarinSkyRL jobs."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

from iris.client.client import JobFailedError

from cloud.iris.artifacts import (
    fs_and_path,
)
from marinskyrl.training_completion import CompletionMode
from cloud.iris.training_result import read_training_result
from cloud.iris.runtime_bundle import runtime_bundle_inputs
from cloud.iris.iris_backend import IrisBackend, IrisLaunchOutcome
from cloud.iris.protocol import (
    AttemptState,
    LaunchMode,
    SkyRLJobSpec,
    SkyRLLaunchResponse,
    SkyRLLaunchRequest,
    SkyRLTrainingResult,
    export_spec,
    job_spec,
)
from cloud.iris.request_builder import build_job_spec


class JobBackend(Protocol):
    """I/O boundary used to submit one prepared Iris request."""

    def validate(self, spec: SkyRLJobSpec, config_path: str) -> None: ...

    def launch(
        self,
        spec: SkyRLJobSpec,
        config_path: str,
        *,
        mode: LaunchMode = LaunchMode.WAIT,
    ) -> IrisLaunchOutcome: ...


def _write_json(uri: str, value: dict[str, Any]) -> None:
    from skyrl_train.io.io import write_bytes_atomic

    _, path = fs_and_path(uri)
    destination = path if uri.startswith("file://") else uri
    write_bytes_atomic(destination, json.dumps(value, sort_keys=True).encode())


def _path_exists(uri: str) -> bool:
    filesystem, path = fs_and_path(uri)
    return filesystem.exists(path)


def _attempt_uri(request: SkyRLLaunchRequest) -> str:
    return f"{request.output.attempts_root.rstrip('/')}/{request.attempt_id}.json"


def _manifest_payload(spec: SkyRLJobSpec, response: SkyRLLaunchResponse) -> dict[str, Any]:
    return {
        "schema_version": spec.schema_version,
        "request": asdict(spec.request),
        "execution": asdict(spec.execution),
        "response": asdict(response),
    }


def _launch_response(
    spec: SkyRLJobSpec,
    state: AttemptState,
    *,
    outcome: IrisLaunchOutcome | None = None,
    training: SkyRLTrainingResult | None = None,
    failure: str | None = None,
) -> SkyRLLaunchResponse:
    return SkyRLLaunchResponse(
        run_id=spec.request.run_id,
        attempt_id=spec.request.attempt_id,
        state=state,
        iris_job_id=outcome.job_id if outcome else None,
        iris_job_state=outcome.job_state if outcome else None,
        runtime=spec.request.runtime,
        training=training,
        failure=failure,
    )


def _record_failed_attempt(
    spec: SkyRLJobSpec,
    outcome: IrisLaunchOutcome,
    failure: str,
) -> SkyRLLaunchResponse:
    request = spec.request
    response = _launch_response(spec, AttemptState.FAILED, outcome=outcome, failure=failure)
    _write_json(_attempt_uri(request), _manifest_payload(spec, response))
    return response


def execute_job(
    spec: SkyRLJobSpec,
    *,
    mode: LaunchMode = LaunchMode.WAIT,
    backend: JobBackend | None = None,
) -> SkyRLLaunchResponse:
    """Validate, detach from, or monitor one job according to ``mode``."""
    request = spec.request
    runtime_bundle_inputs(request.runtime.commit)
    if _path_exists(request.output.terminal_manifest_uri):
        raise ValueError(f"Terminal manifest is immutable and already exists: {request.output.terminal_manifest_uri}")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", encoding="utf-8") as config_file:
        config_file.write(request.config_yaml)
        config_file.flush()
        active_backend = backend or IrisBackend()
        if mode is LaunchMode.PREPARE:
            active_backend.validate(spec, config_file.name)
            return _launch_response(spec, AttemptState.PREPARED)
        try:
            outcome = active_backend.launch(spec, config_file.name, mode=mode)
        except JobFailedError as error:
            job_state = error.status.state.value
            outcome = IrisLaunchOutcome(job_id=str(error.job_id), job_state=job_state, exit_code=1)

        if outcome.exit_code != 0:
            return _record_failed_attempt(spec, outcome, f"Iris job reached {outcome.job_state}")
        if mode is LaunchMode.DETACH:
            return _launch_response(spec, AttemptState.SUBMITTED, outcome=outcome)
    try:
        training = read_training_result(request)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return _record_failed_attempt(spec, outcome, f"Invalid training result: {error}")
    response = _launch_response(spec, AttemptState.SUCCEEDED, outcome=outcome, training=training)
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
    launch_mode = launch_parser.add_mutually_exclusive_group()
    launch_mode.add_argument("--dry-run", action="store_true")
    launch_mode.add_argument("--no-wait", action="store_true", help="Submit and return without following job logs.")

    export_parser = iris_commands.add_parser("export", help="Export a successful native checkpoint without retraining.")
    export_parser.add_argument("--request", required=True)
    export_parser.add_argument("--dry-run", action="store_true")

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
    build_parser.add_argument(
        "--timeout-seconds", type=int, default=None, help="Job deadline in seconds (0 disables it)."
    )
    build_parser.add_argument("--seed", type=int, default=None)
    build_parser.add_argument("--run-prefix", required=True, help="Canonical output root (e.g. s3://bucket/run-id).")
    build_parser.add_argument("--overrides", default=None, help="JSON list of Hydra ++ override strings.")
    build_parser.add_argument("--attempt-id", default=None)
    build_parser.add_argument(
        "--completion-mode", choices=[mode.value for mode in CompletionMode], default="checkpoint"
    )
    build_parser.add_argument("--checkpoint-retention-days", type=int, default=None)
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
            "timeout_seconds",
            "seed",
            "attempt_id",
            "completion_mode",
            "checkpoint_retention_days",
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

    if args.action == "export":
        from cloud.iris.export_job import execute_export

        with open(args.request) as source:
            spec = export_spec(json.load(source))
        with contextlib.redirect_stdout(sys.stderr):
            response = execute_export(spec, mode=LaunchMode.PREPARE if args.dry_run else LaunchMode.WAIT)
        json.dump(asdict(response), sys.stdout, sort_keys=True)
        sys.stdout.write("\n")
        return 0 if response.state in (AttemptState.PREPARED, AttemptState.SUCCEEDED) else 1

    with open(args.request) as source:
        spec = job_spec(json.load(source))
    if args.dry_run:
        mode = LaunchMode.PREPARE
    elif args.no_wait:
        mode = LaunchMode.DETACH
    else:
        mode = LaunchMode.WAIT
    with contextlib.redirect_stdout(sys.stderr):
        response = execute_job(spec, mode=mode)
    json.dump(asdict(response), sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if response.state in (AttemptState.PREPARED, AttemptState.SUBMITTED, AttemptState.SUCCEEDED) else 1


if __name__ == "__main__":
    sys.exit(main())
