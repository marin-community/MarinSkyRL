#!/usr/bin/env python
"""Export a banked training checkpoint to HF safetensors as an Iris job.

WHY THIS EXISTS
---------------
Normal training records ``hf_export_request.json`` beside each checkpoint selected by
``hf_save_interval``. It never gathers live weights for Hugging Face serialization. Pass
that checkpoint directory with ``--request`` to run the conversion on a separate Iris
gang. The request remains pending after failure and becomes complete only after Iris
reports success, so an interrupted or partial export is explicitly rerunnable.

Converting those checkpoints on a laptop is not practical for the Megatron strategy. It
writes a ``torch.distributed.checkpoint`` set (``__N_0.distcp`` + ``.metadata``) whose
tensors are Megatron-native — layer-stacked keys, grouped experts — and the conversion
to HF layout runs through ``bridge.save_hf_weights`` (mbridge), which needs the Megatron
runtime and a live process group at the saved parallel geometry.

The task enters ``skyrl_train.entrypoints.checkpoint_export`` instead of an RL entrypoint.
It creates only the saved policy worker group, restores only model tensors, invokes the
strategy's existing HF converter, and exits. It does not create rollout engines, datasets,
generators, tracking, reference models, critics, optimizers, or schedulers. ``--timeout``
belongs to this export job and is independent of the source training gang's process-group
timeout.

The export lands in ``export_path`` on durable storage. When the request carries the
training run's Hub destination, the export-only job publishes the completed artifact.
"""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from skyrl_train.hf_export_schema import (
    DEFAULT_HF_EXPORT_TIMEOUT,
    DEFAULT_HF_HUB_REVISION,
    DEFAULT_HF_UPLOAD_MODE,
    HFExportRequest,
    HFExportStatus,
    HFUploadMode,
)

from cloud.iris.model_paths import model_source_cli_args
from cloud.iris.rl_config_translation import format_hydra_arg
from cloud.iris.runtime_environment import CHECKPOINT_EXPORT_ENTRYPOINT
from marinskyrl.checkpoint_paths import GLOBAL_STEP_PREFIX, policy_export_path
from marinskyrl.resource_locator import ModelLocatorError

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_hf_export_request(checkpoint_path: str) -> HFExportRequest | None:
    from skyrl_train.hf_export import read_hf_export_request  # noqa: PLC0415 - keep launcher imports Torch-free

    return read_hf_export_request(checkpoint_path)


def _write_hf_export_request(request: HFExportRequest) -> None:
    from skyrl_train.hf_export import write_hf_export_request  # noqa: PLC0415 - keep launcher imports Torch-free

    write_hf_export_request(request)


def _verify_hf_model_export(export_path: str) -> None:
    from skyrl_train.hf_model_io import verify_hf_model_export  # noqa: PLC0415 - keep launcher imports Torch-free

    verify_hf_model_export(export_path)


@dataclass(frozen=True)
class ExportJobSpec:
    request: HFExportRequest
    rl_config: str
    cluster: str
    priority: str
    job_name: str | None
    timeout: int
    no_wait: bool
    cluster_config: str | None = None
    target_cluster: str | None = None
    parent_cluster_config: str | None = None
    cpu: float | None = None
    memory: str | None = None
    disk: str | None = None
    storage_user: str | None = None
    receipt_uri: str | None = None
    request_fingerprint: str | None = None
    attempt_id: str | None = None


def build_command(spec: ExportJobSpec) -> list[str]:
    """Return the Iris backend command that performs an export-only run."""
    request = spec.request

    overrides = [
        format_hydra_arg("checkpoint_export.step", request.step, prefix="++"),
        format_hydra_arg("checkpoint_export.checkpoint_path", request.checkpoint_path, prefix="++"),
        format_hydra_arg("checkpoint_export.export_root", request.export_path, prefix="++"),
        format_hydra_arg("checkpoint_export.hf_hub_repo_id", request.hf_hub_repo_id or "null", prefix="++"),
        format_hydra_arg("checkpoint_export.hf_hub_private", request.hf_hub_private, prefix="++"),
        format_hydra_arg("checkpoint_export.hf_hub_revision", request.hf_hub_revision, prefix="++"),
        format_hydra_arg("checkpoint_export.hf_upload_mode", request.hf_upload_mode, prefix="++"),
    ]
    receipt_values = (spec.receipt_uri, spec.request_fingerprint, spec.attempt_id)
    if any(value is not None for value in receipt_values) and not all(
        isinstance(value, str) and value.strip() for value in receipt_values
    ):
        raise ValueError("Export completion receipt URI, request fingerprint, and attempt ID must be provided together")
    if spec.receipt_uri is not None:
        overrides.extend(
            [
                format_hydra_arg("checkpoint_export.completion_receipt_uri", spec.receipt_uri, prefix="++"),
                format_hydra_arg("checkpoint_export.request_fingerprint", spec.request_fingerprint, prefix="++"),
                format_hydra_arg("checkpoint_export.attempt_id", spec.attempt_id, prefix="++"),
            ]
        )

    cmd = [
        sys.executable,
        "-m",
        "cloud.iris.iris_backend",
        "--rl_config",
        spec.rl_config,
        "--model_path",
        request.model_path,
        "--num-nodes",
        str(request.num_nodes),
        "--gpus-per-node",
        str(request.gpus_per_node),
        "--cluster",
        spec.cluster,
        "--entrypoint",
        CHECKPOINT_EXPORT_ENTRYPOINT,
        "--priority",
        spec.priority,
        # An export job must not be retried into a second export.
        "--max-retries",
        "0",
        "--timeout",
        str(spec.timeout),
    ]
    if spec.cluster_config:
        cmd += ["--cluster-config", spec.cluster_config]
    if spec.target_cluster:
        cmd += ["--target-cluster", spec.target_cluster]
    if spec.parent_cluster_config:
        cmd += ["--parent-cluster-config", spec.parent_cluster_config]
    if spec.cpu is not None:
        cmd += ["--cpu", str(spec.cpu)]
    if spec.memory:
        cmd += ["--memory", spec.memory]
    if spec.disk:
        cmd += ["--disk", spec.disk]
    if spec.storage_user:
        cmd += ["--storage-user", spec.storage_user]
    cmd.extend(model_source_cli_args(request.model_source_uri, request.model_source_identity))
    if spec.no_wait:
        cmd.append("--no-wait")
    if spec.job_name:
        cmd += ["--job-name", spec.job_name]
    for override in overrides:
        cmd += ["--skyrl_override", override]
    return cmd


def argument_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--request", help="global_step_N checkpoint directory containing hf_export_request.json")
    ap.add_argument("--ckpt_path", help="checkpoint root holding global_step_N/")
    ap.add_argument("--step", type=int, help="checkpoint step to export")
    ap.add_argument("--rl_config", required=True, help="the RL config the run was trained with")
    ap.add_argument("--model_path", help="base model path, as at training time")
    ap.add_argument("--model-source-uri", help="object-store model source for a task-local --model_path")
    ap.add_argument("--model-source-identity", help="immutable identity for --model-source-uri")
    ap.add_argument("--cluster", default="cw-rno2a")
    ap.add_argument("--cluster-config")
    ap.add_argument("--target-cluster")
    ap.add_argument("--parent-cluster-config")
    ap.add_argument("--cpu", type=float)
    ap.add_argument("--memory")
    ap.add_argument("--disk")
    ap.add_argument("--storage-user")
    ap.add_argument("--export-receipt-uri")
    ap.add_argument("--export-request-fingerprint")
    ap.add_argument("--export-attempt-id")
    ap.add_argument("--num-nodes", type=int)
    ap.add_argument("--gpus-per-node", type=int)
    ap.add_argument("--priority", default="batch")
    ap.add_argument("--export_path", help="defaults to <ckpt_path parent>/exports")
    ap.add_argument("--job-name", dest="job_name")
    ap.add_argument("--hf-hub-repo-id")
    ap.add_argument("--hf-hub-private", action="store_true", default=None)
    ap.add_argument("--hf-hub-revision")
    ap.add_argument("--hf-upload-mode", choices=tuple(HFUploadMode))
    ap.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_HF_EXPORT_TIMEOUT,
        help=(
            "export-job timeout in seconds, independent of the training process group "
            f"(default: {DEFAULT_HF_EXPORT_TIMEOUT})"
        ),
    )
    ap.add_argument(
        "--no-wait",
        action="store_true",
        help="submit a manual export and return immediately; request-driven exports must record completion",
    )
    ap.add_argument("--dry-run", action="store_true", help="print the command and exit")
    return ap


def request_spec(args: argparse.Namespace, parser: argparse.ArgumentParser) -> ExportJobSpec:
    hydrated_options = {
        "--ckpt_path": args.ckpt_path,
        "--step": args.step,
        "--export_path": args.export_path,
        "--model_path": args.model_path,
        "--model-source-uri": args.model_source_uri,
        "--model-source-identity": args.model_source_identity,
        "--num-nodes": args.num_nodes,
        "--gpus-per-node": args.gpus_per_node,
        "--hf-hub-repo-id": args.hf_hub_repo_id,
        "--hf-hub-private": args.hf_hub_private,
        "--hf-hub-revision": args.hf_hub_revision,
        "--hf-upload-mode": args.hf_upload_mode,
    }
    conflicts = [name for name, value in hydrated_options.items() if value is not None]
    if conflicts:
        parser.error(f"--request cannot be combined with request-owned options: {', '.join(conflicts)}")
    if args.no_wait:
        parser.error("--no-wait cannot be used with --request because completion must be recorded")
    try:
        request = _read_hf_export_request(args.request)
    except ValueError as error:
        parser.error(f"invalid HF export request: {error}")
    if request is None:
        parser.error(f"no hf_export_request.json found under {args.request}")
    return operational_spec(args, request, no_wait=False)


def operational_spec(args: argparse.Namespace, request: HFExportRequest, *, no_wait: bool) -> ExportJobSpec:
    return ExportJobSpec(
        request=request,
        rl_config=args.rl_config,
        cluster=args.cluster,
        priority=args.priority,
        job_name=args.job_name,
        timeout=args.timeout,
        no_wait=no_wait,
        cluster_config=args.cluster_config,
        target_cluster=args.target_cluster,
        parent_cluster_config=args.parent_cluster_config,
        cpu=args.cpu,
        memory=args.memory,
        disk=args.disk,
        storage_user=args.storage_user,
        receipt_uri=args.export_receipt_uri,
        request_fingerprint=args.export_request_fingerprint,
        attempt_id=args.export_attempt_id,
    )


def manual_spec(args: argparse.Namespace, parser: argparse.ArgumentParser) -> ExportJobSpec:
    required = {
        "--ckpt_path": args.ckpt_path,
        "--step": args.step,
        "--model_path": args.model_path,
        "--num-nodes": args.num_nodes,
        "--gpus-per-node": args.gpus_per_node,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error(f"provide --request or all manual export options; missing {', '.join(missing)}")
    checkpoint_base_path = args.ckpt_path.rstrip("/")
    export_path = args.export_path or os.path.join(os.path.dirname(checkpoint_base_path), "exports")
    try:
        request = HFExportRequest(
            step=args.step,
            checkpoint_base_path=checkpoint_base_path,
            checkpoint_path=os.path.join(checkpoint_base_path, f"{GLOBAL_STEP_PREFIX}{args.step}"),
            export_path=export_path,
            model_path=args.model_path,
            num_nodes=args.num_nodes,
            gpus_per_node=args.gpus_per_node,
            model_source_uri=args.model_source_uri,
            model_source_identity=args.model_source_identity,
            hf_hub_repo_id=args.hf_hub_repo_id,
            hf_hub_private=bool(args.hf_hub_private),
            hf_hub_revision=args.hf_hub_revision or DEFAULT_HF_HUB_REVISION,
            hf_upload_mode=HFUploadMode(args.hf_upload_mode or DEFAULT_HF_UPLOAD_MODE),
        )
    except ModelLocatorError as error:
        parser.error(str(error))
    return operational_spec(args, request, no_wait=args.no_wait)


def _run_export(spec: ExportJobSpec, command: list[str]) -> None:
    print(
        f"[export-hf] geometry {spec.request.num_nodes}x{spec.request.gpus_per_node} GPU — this MUST match "
        f"the training geometry or the sharded load will not resolve"
    )
    exit_code = subprocess.call(command, cwd=str(_REPO_ROOT))
    if exit_code != 0:
        raise subprocess.CalledProcessError(exit_code, command)
    if not spec.no_wait:
        export_path = policy_export_path(spec.request.export_path, spec.request.step)
        _verify_hf_model_export(export_path)


def submit_requested_export(spec: ExportJobSpec, command: list[str]) -> None:
    """Submit one export job, verify synchronous output, and persist request state."""
    request = spec.request.with_status(HFExportStatus.IN_PROGRESS, timeout=spec.timeout, increment_attempts=True)
    _write_hf_export_request(request)
    try:
        _run_export(spec, command)
    except BaseException as error:
        exit_code = error.returncode if isinstance(error, subprocess.CalledProcessError) else 1
        _write_hf_export_request(request.with_status(HFExportStatus.PENDING, last_exit_code=exit_code))
        raise
    _write_hf_export_request(request.with_status(HFExportStatus.COMPLETE, last_exit_code=0))


def main() -> None:
    parser = argument_parser()
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive for an export job")
    if args.request:
        spec = request_spec(args, parser)
        request = spec.request
    else:
        request = None
        spec = manual_spec(args, parser)

    if request is not None and request.status is HFExportStatus.COMPLETE:
        print(f"[export-hf] global_step_{request.step} is already complete")
        return

    cmd = build_command(spec)

    print("[export-hf] converting checkpoint step", spec.request.step)
    print("[export-hf]", " ".join(cmd))
    if args.dry_run:
        return
    if request is None:
        _run_export(spec, cmd)
    else:
        submit_requested_export(spec, cmd)


if __name__ == "__main__":
    main()
