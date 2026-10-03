"""Opt-in Ray/GPU worker abort through the Iris task runtime's failure teardown.

The fixture replaces only the training-driver command. Ray startup, subprocess
supervision, failure uploads and Ray shutdown use the production task runtime.
It does not exercise an NCCL collective or generate an NCCL timeout dump.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
from unittest.mock import patch

from cloud.iris import task_runtime
from cloud.iris.artifacts import fs_and_path
from cloud.iris.ray_storage import RaySpillBackend
from marinskyrl.environment_contract import DEBUG_ARTIFACT_DIR_ENV, DebugMode, EnvVarManager, EnvVarScope
from marinskyrl.process_diagnostics import ProcessOutcome, initialize_process_diagnostics


WORKER_ARTIFACT = "runs/worker-abort.json"
LOSS_ARTIFACT = "runs/worker-loss.json"
FAILURE_EXIT_CODE = 42
WORKER_TIMEOUT = 120


def failure_marker(identity: dict) -> str:
    return f"HERO_CAPTURE_ABORT run={identity['run_id']} worker={identity['worker_id']} pid={identity['pid']}"


def debug_environment(run_id: str) -> EnvVarManager:
    return EnvVarManager.for_debug_launch(
        job_name=run_id,
        mode=DebugMode.LIGHT,
        artifact_root=os.environ[DEBUG_ARTIFACT_DIR_ENV],
    )


def fixture_driver_command(_config_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "scripts.hero_failure_capture",
        "driver",
        "--run-id",
        os.environ["HERO_CAPTURE_RUN_ID"],
    ]


def run_controller(output: str, run_id: str) -> int:
    debug_environment(run_id).apply_to_process(EnvVarScope.TASK_RUNTIME)
    initialize_process_diagnostics("capture-task-runtime")
    args = argparse.Namespace(
        ray_port=6379,
        rendezvous_dir=f"{output}/rendezvous",
        ray_log_dir=f"{output}/ray-logs",
        ray_spill_backend=RaySpillBackend.LOCAL,
        ray_spill_dir="/tmp/ray-spill",
        cluster_join_timeout=120,
        driver_liveness_timeout=180,
    )
    # The arbitrary-command interface used by the custom Hero launcher predates
    # main's config-only driver. Adapt only its subprocess command in this fixture.
    with patch.object(task_runtime, "training_driver_command", fixture_driver_command):
        return task_runtime.run_head(args, Path("unused-fixture-config"))


def supervise(command: list[str]) -> int:
    """Forward task signals and wait for the runtime's final uploads before exit."""
    child = None
    termination_signal = None

    def forward_signal(signum, _frame):
        nonlocal termination_signal
        termination_signal = signum
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGTERM, forward_signal)
    signal.signal(signal.SIGINT, forward_signal)
    child = subprocess.Popen(command, start_new_session=True)
    if termination_signal is not None:
        child.send_signal(termination_signal)
    return ProcessOutcome.from_returncode(child.wait()).public_exit_code


def run_driver(run_id: str) -> int:
    # These GPU dependencies are deliberately absent from the acceptance reader.
    import ray
    import torch

    @ray.remote(num_gpus=1, max_restarts=0)
    class AbortWorker:
        def ready(self) -> dict:
            context = ray.get_runtime_context()
            value = torch.ones(8, device="cuda")
            torch.cuda.synchronize()
            identity = {
                "run_id": run_id,
                "worker_id": context.get_worker_id(),
                "actor_id": context.get_actor_id(),
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "debug_root": os.environ[DEBUG_ARTIFACT_DIR_ENV],
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu_sum": value.sum().item(),
                "artifact_kind": "synthetic transport receipt",
            }
            initialize_process_diagnostics("capture-abort-worker")
            (Path(identity["debug_root"]) / WORKER_ARTIFACT).write_text(json.dumps(identity, sort_keys=True) + "\n")
            return identity

        def abort(self) -> None:
            identity = json.loads((Path(os.environ[DEBUG_ARTIFACT_DIR_ENV]) / WORKER_ARTIFACT).read_text())
            print(failure_marker(identity), file=sys.stderr, flush=True)
            os.abort()

    ray.init(
        address=os.environ["RAY_ADDRESS"],
        runtime_env={"env_vars": debug_environment(run_id).environment_for(EnvVarScope.RAY_WORKER)},
    )
    try:
        worker = AbortWorker.remote()
        identity = ray.get(worker.ready.remote(), timeout=WORKER_TIMEOUT)
        assert identity["debug_root"] == os.environ[DEBUG_ARTIFACT_DIR_ENV]
        assert identity["gpu_sum"] == 8
        try:
            ray.get(worker.abort.remote(), timeout=WORKER_TIMEOUT)
        except ray.exceptions.RayActorError as error:
            loss = {
                "run_id": run_id,
                "worker_id": identity["worker_id"],
                "pid": identity["pid"],
                "exception": type(error).__name__,
            }
            (Path(os.environ[DEBUG_ARTIFACT_DIR_ENV]) / LOSS_ARTIFACT).write_text(
                json.dumps(loss, sort_keys=True) + "\n"
            )
            print("HERO_CAPTURE_WORKER_LOST " + json.dumps(loss), flush=True)
            return FAILURE_EXIT_CODE
        raise RuntimeError("worker survived os.abort")
    finally:
        ray.shutdown()


def check_artifacts(output: str, run_id: str) -> dict:
    """Read retained bytes and reject absent, stale or incomplete failure evidence."""
    filesystem, root = fs_and_path(output)
    manifests = filesystem.glob(f"{root}/rendezvous/debug_artifacts/*/sync-manifest.json")
    if len(manifests) != 1:
        raise ValueError(f"expected one upload receipt, found {len(manifests)}")
    manifest_path = manifests[0]
    manifest = json.loads(filesystem.cat(manifest_path))
    debug_root = str(Path(manifest_path).parent)
    worker_bytes = filesystem.cat(f"{debug_root}/{WORKER_ARTIFACT}")
    identity = json.loads(worker_bytes)
    loss_bytes = filesystem.cat(f"{debug_root}/{LOSS_ARTIFACT}")
    loss = json.loads(loss_bytes)
    if (
        identity["run_id"] != run_id
        or loss["run_id"] != run_id
        or loss["worker_id"] != identity["worker_id"]
        or loss["pid"] != identity["pid"]
    ):
        raise ValueError("failure artifact identity does not match this run")
    copied = {item["path"]: item["bytes"] for item in manifest["copied"]}
    if (
        manifest["skipped"]
        or copied.get(WORKER_ARTIFACT) != len(worker_bytes)
        or copied.get(LOSS_ARTIFACT) != len(loss_bytes)
    ):
        raise ValueError("upload receipt does not cover the failure artifacts")
    if manifest["source_root"] != identity["debug_root"] or not manifest["reason"].startswith("driver exit_code=42"):
        raise ValueError("missing final failure upload receipt")
    logs = filesystem.glob(f"{root}/ray-logs/{manifest['node_id']}/session_*/worker-*{identity['pid']}.err")
    marker = failure_marker(identity).encode()
    matching_logs = [path for path in logs if marker in filesystem.cat(path)]
    if not matching_logs:
        raise ValueError("no retained worker stderr identifies the deliberate abort")
    return {
        "run_id": run_id,
        "identity": identity,
        "worker_artifact_sha256": hashlib.sha256(worker_bytes).hexdigest(),
        "loss_artifact_sha256": hashlib.sha256(loss_bytes).hexdigest(),
        "manifest": manifest,
        "failure_logs": matching_logs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("task", "controller", "driver", "check"))
    parser.add_argument("--output")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    os.environ["HERO_CAPTURE_RUN_ID"] = args.run_id
    if args.mode == "driver":
        return run_driver(args.run_id)
    if not args.output:
        parser.error("--output is required")
    if args.mode == "check":
        print(json.dumps(check_artifacts(args.output, args.run_id), indent=2, sort_keys=True))
        return 0
    if args.mode == "controller":
        return run_controller(args.output, args.run_id)
    return supervise(
        [
            sys.executable,
            "-m",
            "scripts.hero_failure_capture",
            "controller",
            "--output",
            args.output,
            "--run-id",
            args.run_id,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
