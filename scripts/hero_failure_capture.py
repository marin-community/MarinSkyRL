"""Opt-in worker abort and retained-evidence check through the existing Iris runtime."""

import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

from cloud.iris import task_runtime
from cloud.iris.artifacts import fs_and_path
from marinskyrl.environment_contract import DEBUG_ARTIFACT_DIR_ENV, DebugMode, EnvVarManager, EnvVarScope
from marinskyrl.resource_locator import join_resource_path


WORKER_ARTIFACT = "runs/worker-abort.json"
FAILURE_EXIT_CODE = 42


def debug_environment(run_id: str) -> EnvVarManager:
    return EnvVarManager.for_debug_launch(
        job_name=run_id, mode=DebugMode.LIGHT, artifact_root=os.environ.get(DEBUG_ARTIFACT_DIR_ENV, "/tmp/debug")
    )


def failure_marker(identity: dict) -> str:
    return f"HERO_CAPTURE_ABORT run={identity['run_id']} worker={identity['worker_id']} pid={identity['pid']}"


def run_driver(run_id: str) -> int:
    import ray
    import torch

    @ray.remote(num_gpus=1, max_restarts=0)
    class AbortWorker:
        def abort(self) -> None:
            assert torch.ones(8, device="cuda").sum().item() == 8
            torch.cuda.synchronize()
            identity = {
                "run_id": run_id,
                "worker_id": ray.get_runtime_context().get_worker_id(),
                "pid": os.getpid(),
                "debug_root": os.environ[DEBUG_ARTIFACT_DIR_ENV],
                "gpu": torch.cuda.get_device_name(0),
                "artifact_kind": "synthetic transport receipt",
            }
            (Path(identity["debug_root"]) / WORKER_ARTIFACT).write_text(json.dumps(identity))
            print(failure_marker(identity), file=sys.stderr, flush=True)
            os.abort()

    ray.init(
        address=os.environ["RAY_ADDRESS"],
        runtime_env={"env_vars": debug_environment(run_id).environment_for(EnvVarScope.RAY_WORKER)},
    )
    try:
        worker = AbortWorker.remote()
        try:
            ray.get(worker.abort.remote(), timeout=120)
        except ray.exceptions.RayActorError:
            return FAILURE_EXIT_CODE
        raise RuntimeError("worker survived os.abort")
    finally:
        ray.shutdown()


def run_capture(output: str, run_id: str) -> int:
    os.environ["SKYRL_HOME"] = str(Path(__file__).resolve().parents[1])
    debug_environment(run_id).apply_to_process(EnvVarScope.TASK_RUNTIME)
    args = argparse.Namespace(
        ray_port=6379,
        rendezvous_dir=join_resource_path(output, "rendezvous"),
        ray_log_dir=join_resource_path(output, "ray-logs"),
        ray_spill_backend=task_runtime.RaySpillBackend.LOCAL,
        ray_spill_dir="/tmp/ray-spill",
        cluster_join_timeout=120,
        driver_liveness_timeout=180,
    )
    command = [sys.executable, "-m", "scripts.hero_failure_capture", "driver", "--run-id", run_id]
    # Change only the subprocess command. The runtime owns signals, uploads and shutdown.
    with patch.object(task_runtime, "training_driver_command", lambda _config: command):
        return task_runtime.run_head(args, Path("unused-fixture-config"))


def check_artifacts(output: str, run_id: str) -> dict:
    filesystem, root = fs_and_path(output)
    (manifest_path,) = filesystem.glob(f"{root}/rendezvous/debug_artifacts/*/sync-manifest.json")
    manifest = json.loads(filesystem.cat(manifest_path))
    worker_bytes = filesystem.cat(f"{Path(manifest_path).parent}/{WORKER_ARTIFACT}")
    identity = json.loads(worker_bytes)
    if identity["run_id"] != run_id or manifest["source_root"] != identity["debug_root"]:
        raise ValueError("failure evidence belongs to another run or debug root")
    copied = {item["path"]: item["bytes"] for item in manifest["copied"]}
    if manifest["skipped"] or copied.get(WORKER_ARTIFACT) != len(worker_bytes):
        raise ValueError("upload receipt does not cover the worker artifact")
    if not manifest["reason"].startswith(f"driver exit_code={FAILURE_EXIT_CODE}"):
        raise ValueError("missing final failure upload")
    marker = failure_marker(identity).encode()
    for path in filesystem.glob(f"{root}/ray-logs/{manifest['node_id']}/session_*/worker-*{identity['pid']}.err"):
        payload = filesystem.cat(path)
        offset = payload.find(marker)
        if offset >= 0 and b"Fatal Python error: Aborted" in payload[offset + len(marker) :]:
            return {"identity": identity, "manifest": manifest, "stderr": path}
    raise ValueError("missing worker abort and fatal traceback")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "driver", "check"))
    parser.add_argument("--output")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.mode == "driver":
        sys.exit(run_driver(args.run_id))
    if not args.output:
        parser.error("--output is required")
    if args.mode == "run":
        sys.exit(run_capture(args.output, args.run_id))
    print(json.dumps(check_artifacts(args.output, args.run_id), indent=2))
