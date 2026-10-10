"""Run the task-owned driver under the current production Ray supervisor."""

import argparse
import os
from pathlib import Path

from cloud.iris import task_runtime as runtime
from cloud.iris.ray_storage import RaySpillBackend
from marinskyrl.environment_contract import TrainingType


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rendezvous-dir", required=True)
    parser.add_argument("--ray-log-dir", required=True)
    parser.add_argument("--rendezvous-timeout", type=int, default=7200)
    parser.add_argument("--cluster-join-timeout", type=int, default=7200)
    parser.add_argument("--driver-liveness-timeout", type=int, default=3600)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a driver command is required")
    args.ray_port = 6379
    args.ray_spill_dir = "/tmp/hero-ray-spill"
    args.ray_spill_backend = RaySpillBackend.LOCAL
    runtime._pin_boto3_s3_addressing_style()
    interface = runtime.pin_socket_ifname()
    runtime.export_telemetry_environment(os.environ["HERO_RUN_NAME"], TrainingType.ASYNC)
    runtime.ensure_fr_dump_dir()
    # Only the campaign driver command differs; gang lifecycle, logs, failure
    # capture, runtime matching and cleanup remain the production supervisor's.
    runtime.training_driver_command = lambda _config_path: command
    if runtime._rank() == 0:
        return runtime.run_head(args, Path("/tmp/hero-task-driver"), interface)
    return runtime.run_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
