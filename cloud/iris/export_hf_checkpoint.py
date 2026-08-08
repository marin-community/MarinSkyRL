#!/usr/bin/env python
"""Export a banked training checkpoint to HF safetensors as an Iris job.

WHY THIS EXISTS
---------------
Normal training records ``hf_export_request.json`` beside each checkpoint selected by
``hf_save_interval``. It never gathers live weights for Hugging Face serialization. Pass
that checkpoint directory with ``--request`` to run the conversion on a separate Iris
gang. The request remains pending after failure and becomes complete only after Iris
reports success, so an interrupted or partial export is explicitly rerunnable.

Converting those checkpoints offline is not practical for the megatron strategy. It
writes a ``torch.distributed.checkpoint`` set (``__N_0.distcp`` + ``.metadata``) whose
tensors are Megatron-native — layer-stacked keys, grouped experts — and the conversion
to HF layout runs through ``bridge.save_hf_weights`` (mbridge), which needs the Megatron
runtime and a live process group at the original parallel geometry. There is no laptop
path.

This command does not reimplement conversion. It re-runs the trainer's own export by
using its resume-at-max-steps path:

``FullyAsyncTrainer._train_loop`` checks, right after resuming, whether the resumed step
is at or past ``max_steps``. If it is, the run is complete: it calls
``_handle_resume_at_max_steps``, which fires ``on_train_end``, and the checkpoint
callback then requests an HF save (``save_on_train_end and save_steps > 0``). That runs
``save_models`` → ``save_hf_model`` → ``bridge.save_hf_weights``, followed by
``_flush_hf_uploads``, and exits 0.

Setting ``max_steps`` to the checkpoint's own step produces a job that loads the weights,
exports them, and stops without training a step. ``--timeout`` belongs to this export job;
it is independent of the source training gang's process-group timeout.

The export lands in ``export_path`` on durable storage. When the request carries the
training run's Hub destination, the export-only job publishes the completed artifact.
"""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from skyrl_train.hf_export import read_hf_export_request, write_hf_export_request
from skyrl_train.hf_export_schema import (
    DEFAULT_HF_EXPORT_TIMEOUT,
    DEFAULT_HF_HUB_REVISION,
    DEFAULT_HF_UPLOAD_MODE,
    HFExportRequest,
    HFExportStatus,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Any reachable dataset satisfies the trainer's construction-time size assert. This one
# is small (514 rows) and is already part of the sweep, so an export pulls nothing new.
DEFAULT_EXPORT_TRAIN_DATA = '["DCAgent/exp_rpt_curriculum-easy"]'


@dataclass(frozen=True)
class ExportJobSpec:
    checkpoint_base_path: str
    step: int
    export_path: str
    rl_config: str
    model_path: str
    cluster: str
    num_nodes: int
    gpus_per_node: int
    priority: str
    train_data: str
    job_name: str | None
    timeout: int
    no_wait: bool
    hf_hub_repo_id: str | None
    hf_hub_private: bool
    hf_hub_revision: str
    hf_upload_mode: str


def build_command(spec: ExportJobSpec) -> list[str]:
    """Return the Iris backend command that performs an export-only run."""
    resume_path = f"{spec.checkpoint_base_path}/global_step_{spec.step}"

    overrides = [
        # Resume this exact step, not whatever is newest. The step to publish is chosen
        # by trailing-EMA reward, which is rarely the last one a run banked.
        "++trainer.resume_mode=from_path",
        f"++trainer.resume_path={resume_path}",
        # The load-bearing argument: resuming AT max_steps takes the
        # already-complete branch, which finalizes and exits instead of training.
        f"++trainer.max_steps={spec.step}",
        f"++trainer.ckpt_path={spec.checkpoint_base_path}",
        f"++trainer.export_path={spec.export_path}",
        # Force legacy callbacks so an explicit training callback list cannot
        # resave the source checkpoint or suppress this one-shot export.
        "++trainer.callbacks=[]",
        "++trainer.ckpt_interval=-1",
        "++trainer.hf_save_interval=1",
        "++trainer.hf_export_execution=true",
        f"++trainer.hf_hub_repo_id={spec.hf_hub_repo_id or 'null'}",
        f"++trainer.hf_hub_private={str(spec.hf_hub_private).lower()}",
        f"++trainer.hf_hub_revision={spec.hf_hub_revision}",
        f"++trainer.hf_upload_mode={spec.hf_upload_mode}",
        "++trainer.enable_db_registration=false",
    ]

    cmd = [
        sys.executable,
        "-m",
        "cloud.iris.iris_backend",
        "--rl_config",
        spec.rl_config,
        "--model_path",
        spec.model_path,
        "--num-nodes",
        str(spec.num_nodes),
        "--gpus-per-node",
        str(spec.gpus_per_node),
        # The trainer builds its prompts dataset before it can reach the
        # resume-at-max-steps branch, and asserts the dataset is at least
        # train_batch_size. The sweep configs carry data.train_data: [], so an
        # export that passes no data dies at
        # "dataset should be atleast as large as train_batch_size 64, got size 0"
        # roughly eight minutes in, after the Ray head is already up. The rows are
        # never consumed: max_steps is already reached, so zero steps run.
        "--train_data",
        spec.train_data,
        "--cluster",
        spec.cluster,
        "--target-cluster",
        spec.cluster,
        "--priority",
        spec.priority,
        # An export job must not be retried into a second export.
        "--max-retries",
        "0",
        "--timeout",
        str(spec.timeout),
    ]
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
    ap.add_argument("--cluster", default="cw-rno2a")
    ap.add_argument("--num-nodes", type=int)
    ap.add_argument("--gpus-per-node", type=int)
    ap.add_argument("--priority", default="batch")
    ap.add_argument(
        "--train_data",
        default=DEFAULT_EXPORT_TRAIN_DATA,
        help=(
            "JSON list of dataset paths. Never trained on — it only has to be large enough "
            "for the trainer to construct. Override only if the default is unreachable."
        ),
    )
    ap.add_argument("--export_path", help="defaults to <ckpt_path parent>/exports")
    ap.add_argument("--job-name", dest="job_name")
    ap.add_argument("--hf-hub-repo-id")
    ap.add_argument("--hf-hub-private", action="store_true", default=None)
    ap.add_argument("--hf-hub-revision")
    ap.add_argument("--hf-upload-mode", choices=("latest", "all"))
    ap.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_HF_EXPORT_TIMEOUT,
        help=(
            "export-job timeout in seconds, independent of the training process group "
            f"(default: {DEFAULT_HF_EXPORT_TIMEOUT})"
        ),
    )
    ap.add_argument("--no-wait", action="store_true", help="submit without recording request completion")
    ap.add_argument("--dry-run", action="store_true", help="print the command and exit")
    return ap


def request_spec(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[HFExportRequest, ExportJobSpec]:
    hydrated_options = {
        "--ckpt_path": args.ckpt_path,
        "--step": args.step,
        "--export_path": args.export_path,
        "--model_path": args.model_path,
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
    request = read_hf_export_request(args.request)
    if request is None:
        parser.error(f"no hf_export_request.json found under {args.request}")
    return request, ExportJobSpec(
        checkpoint_base_path=request.checkpoint_base_path.rstrip("/"),
        step=request.step,
        export_path=request.export_path,
        rl_config=args.rl_config,
        model_path=request.model_path,
        cluster=args.cluster,
        num_nodes=request.num_nodes,
        gpus_per_node=request.gpus_per_node,
        priority=args.priority,
        train_data=args.train_data,
        job_name=args.job_name,
        timeout=args.timeout,
        no_wait=False,
        hf_hub_repo_id=request.hf_hub_repo_id,
        hf_hub_private=request.hf_hub_private,
        hf_hub_revision=request.hf_hub_revision,
        hf_upload_mode=request.hf_upload_mode,
    )


def manual_spec(args: argparse.Namespace, parser: argparse.ArgumentParser) -> ExportJobSpec:
    if args.ckpt_path is None or args.step is None or args.model_path is None:
        parser.error("provide --request or all of --ckpt_path, --step, and --model_path")
    checkpoint_base_path = args.ckpt_path.rstrip("/")
    export_path = args.export_path or f"{checkpoint_base_path.rsplit('/', 1)[0]}/exports"
    return ExportJobSpec(
        checkpoint_base_path=checkpoint_base_path,
        step=args.step,
        export_path=export_path,
        rl_config=args.rl_config,
        model_path=args.model_path,
        cluster=args.cluster,
        num_nodes=args.num_nodes if args.num_nodes is not None else 4,
        gpus_per_node=args.gpus_per_node if args.gpus_per_node is not None else 8,
        priority=args.priority,
        train_data=args.train_data,
        job_name=args.job_name,
        timeout=args.timeout,
        no_wait=args.no_wait,
        hf_hub_repo_id=args.hf_hub_repo_id,
        hf_hub_private=bool(args.hf_hub_private),
        hf_hub_revision=args.hf_hub_revision or DEFAULT_HF_HUB_REVISION,
        hf_upload_mode=args.hf_upload_mode or DEFAULT_HF_UPLOAD_MODE,
    )


def submit_export(spec: ExportJobSpec, request: HFExportRequest | None, command: list[str]) -> int:
    """Submit one export job and persist the request's terminal state."""
    if request is not None:
        request = request.with_status(
            HFExportStatus.IN_PROGRESS,
            timeout_seconds=spec.timeout,
            increment_attempts=True,
        )
        write_hf_export_request(request)

    print(
        f"[export-hf] geometry {spec.num_nodes}x{spec.gpus_per_node} GPU — this MUST match "
        f"the training geometry or the sharded load will not resolve"
    )
    exit_code = subprocess.call(command, cwd=str(_REPO_ROOT))
    if request is not None:
        status = HFExportStatus.COMPLETE if exit_code == 0 else HFExportStatus.PENDING
        write_hf_export_request(request.with_status(status, last_exit_code=exit_code))
    return exit_code


def main() -> None:
    parser = argument_parser()
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive for an export job")
    if args.request:
        request, spec = request_spec(args, parser)
    else:
        request = None
        spec = manual_spec(args, parser)

    if request is not None and request.status is HFExportStatus.COMPLETE:
        print(f"[export-hf] global_step_{request.step} is already complete")
        return

    cmd = build_command(spec)

    print("[export-hf] resuming step", spec.step, "and exporting without training")
    print("[export-hf]", " ".join(cmd))
    if args.dry_run:
        return
    raise SystemExit(submit_export(spec, request, cmd))


if __name__ == "__main__":
    main()
