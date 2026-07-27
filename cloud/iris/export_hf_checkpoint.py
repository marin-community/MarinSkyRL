#!/usr/bin/env python
"""Export a banked training checkpoint to HF safetensors, as an iris job.

WHY THIS EXISTS
---------------
A run can finish with training checkpoints and no publishable model: ``save_hf_model``
is gated on ``hf_save_interval``, and before the durable-``export_path`` fix it wrote to
node-local disk that vanished with the job. Several completed runs are in that state.

Converting those checkpoints offline is not practical for the megatron strategy. It
writes a ``torch.distributed.checkpoint`` set (``__N_0.distcp`` + ``.metadata``) whose
tensors are Megatron-native — layer-stacked keys, grouped experts — and the conversion
to HF layout runs through ``bridge.save_hf_weights`` (mbridge), which needs the Megatron
runtime and a live process group at the original parallel geometry. There is no laptop
path.

So this does not reimplement the conversion. It re-runs **the trainer's own export**,
which is already tested and already correct for both strategies, by exploiting a path
the trainer has for a different purpose:

``FullyAsyncTrainer._train_loop`` checks, right after resuming, whether the resumed step
is at or past ``max_steps``. If it is, the run is complete: it calls
``_handle_resume_at_max_steps``, which fires ``on_train_end``, and the checkpoint
callback then requests an HF save (``save_on_train_end and save_steps > 0``). That runs
``save_models`` → ``save_hf_model`` → ``bridge.save_hf_weights``, followed by
``_flush_hf_uploads``, and exits 0.

Setting ``max_steps`` to the checkpoint's own step therefore produces a job that loads
the weights, exports them, and stops — **without training a single step**. Every piece is
existing, exercised code; this script only supplies the argument combination.

WHAT IT DOES NOT DO
-------------------
It does not publish. The export lands in ``export_path`` on durable storage. Pushing to
the Hub is a separate, owner-authorized step.
"""

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def build_command(args: argparse.Namespace) -> list[str]:
    """Return the launch_rl_iris command that performs an export-only run."""
    ckpt_root = args.ckpt_path.rstrip("/")
    resume_path = f"{ckpt_root}/global_step_{args.step}"
    export_path = args.export_path or f"{ckpt_root.rsplit('/', 1)[0]}/exports"

    overrides = [
        # Resume this exact step, not whatever is newest. The step to publish is chosen
        # by trailing-EMA reward, which is rarely the last one a run banked.
        "++trainer.resume_mode=from_path",
        f"++trainer.resume_path={resume_path}",
        # The load-bearing argument: resuming AT max_steps takes the
        # already-complete branch, which finalizes and exits instead of training.
        f"++trainer.max_steps={args.step}",
        f"++trainer.ckpt_path={ckpt_root}",
        f"++trainer.export_path={export_path}",
        # on_train_end only requests an HF save when save_steps > 0.
        "++trainer.hf_save_interval=1",
        # Nothing should be pushed anywhere by an export job.
        "++trainer.hf_hub_repo_id=null",
        "++trainer.enable_db_registration=false",
    ]

    cmd = [
        sys.executable,
        "-m",
        "cloud.iris.launch_rl_iris",
        "--rl_config",
        args.rl_config,
        "--model_path",
        args.model_path,
        "--num-nodes",
        str(args.num_nodes),
        "--gpus-per-node",
        str(args.gpus_per_node),
        "--cluster",
        args.cluster,
        "--target-cluster",
        args.cluster,
        "--priority",
        args.priority,
        # An export job must not be retried into a second export.
        "--max-retries",
        "0",
        "--no-wait",
    ]
    if args.job_name:
        cmd += ["--job-name", args.job_name]
    for override in overrides:
        cmd += ["--skyrl_override", override]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt_path", required=True, help="s3:// checkpoint root holding global_step_N/")
    ap.add_argument("--step", required=True, type=int, help="checkpoint step to export")
    ap.add_argument("--rl_config", required=True, help="the RL config the run was trained with")
    ap.add_argument("--model_path", required=True, help="base model path, as at training time")
    ap.add_argument("--cluster", default="cw-rno2a")
    ap.add_argument("--num-nodes", type=int, default=4)
    ap.add_argument("--gpus-per-node", type=int, default=8)
    ap.add_argument("--priority", default="batch")
    ap.add_argument("--export_path", help="defaults to <ckpt_path parent>/exports")
    ap.add_argument("--job-name", dest="job_name")
    ap.add_argument("--dry-run", action="store_true", help="print the command and exit")
    args = ap.parse_args()

    cmd = build_command(args)

    print("[export-hf] resuming step", args.step, "and exporting without training")
    print("[export-hf]", " ".join(cmd))
    if args.dry_run:
        return

    # The geometry must match training: the checkpoint is sharded to the parallel layout
    # the run used, and bridge.save_hf_weights gathers across that same mesh.
    print(
        f"[export-hf] geometry {args.num_nodes}x{args.gpus_per_node} GPU — this MUST match "
        f"the training geometry or the sharded load will not resolve"
    )
    raise SystemExit(subprocess.call(cmd, cwd=str(_REPO_ROOT)))


if __name__ == "__main__":
    main()
