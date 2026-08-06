#!/usr/bin/env python3
"""Submit the fixed-replay Grug benchmark through the pinned Iris runtime."""

from __future__ import annotations

import argparse
import os
import shlex
from pathlib import Path

from iris.cli.connect import open_iris_client
from iris.cluster.types import Entrypoint, EnvironmentSpec
from iris.rpc import job_pb2

from cloud.iris.iris_backend import _gpu_constraints, _gpu_multinode, _gpu_resources
from cloud.iris.runtime_bundle import build_runtime_bundle, resolve_launcher_source
from cloud.iris.runtime_environment import (
    MARINSKYRL_ACTIVATION_FILE,
    MARINSKYRL_TASK_ROOT,
    RuntimeProfile,
    task_setup_script,
)

APP_DIR = str(Path(MARINSKYRL_TASK_ROOT).parent)
RAY_SPILL_DIR = "/tmp/ray_spill"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-config", type=Path, required=True)
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--task-image", required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--rendezvous-dir", required=True)
    parser.add_argument("--nodes", choices=(1, 4), type=int, required=True)
    parser.add_argument("--priority", choices=("production", "interactive", "batch"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--cpu", type=float, default=48.0)
    parser.add_argument("--memory", default="1500GB")
    parser.add_argument("--disk", default="1000GB")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the immutable job request without contacting Iris.",
    )
    args, benchmark_argv = parser.parse_known_args()
    if benchmark_argv and benchmark_argv[0] == "--":
        benchmark_argv = benchmark_argv[1:]
    if not benchmark_argv:
        parser.error("pass benchmark arguments after `--`")
    return args, benchmark_argv


def argument_value(argv: list[str], name: str) -> str:
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"benchmark command is missing {name}") from error


def validate_request(args: argparse.Namespace, benchmark_argv: list[str]) -> None:
    source = resolve_launcher_source()
    if source.commit != args.runtime_commit:
        raise ValueError(f"launcher checkout is {source.commit}, requested {args.runtime_commit}")
    if "@sha256:" not in args.task_image:
        raise ValueError("--task-image must use an immutable sha256 digest")
    expected = {
        "--source-revision": args.runtime_commit,
        "--image": args.task_image,
        "--model": args.model,
        "--model-revision": args.model_revision,
        "--sample": "1",
    }
    for name, value in expected.items():
        actual = argument_value(benchmark_argv, name)
        if actual != value:
            raise ValueError(f"benchmark {name}={actual!r}, expected {value!r}")
    mode = argument_value(benchmark_argv, "--mode")
    expected_nodes = 1 if mode == "preflight" else 4 if mode == "headline" else None
    if expected_nodes is None or args.nodes != expected_nodes:
        raise ValueError(f"benchmark mode {mode!r} requires {expected_nodes} nodes, got {args.nodes}")


def controller_command(args: argparse.Namespace, benchmark_argv: list[str]) -> list[str]:
    benchmark = [
        "python",
        f"{MARINSKYRL_TASK_ROOT}/skyrl-train/scripts/grug_fixed_replay_benchmark.py",
        *benchmark_argv,
    ]
    return [
        "python",
        "cloud/iris/task_runtime.py",
        "--ray-spill-dir",
        RAY_SPILL_DIR,
        "--rendezvous-dir",
        args.rendezvous_dir,
        "--prestage-model",
        args.model,
        "--prestage-model-revision",
        args.model_revision,
        "--",
        *benchmark,
    ]


def task_command(args: argparse.Namespace, benchmark_argv: list[str]) -> list[str]:
    controller = controller_command(args, benchmark_argv)
    pythonpath = f"{APP_DIR}:{MARINSKYRL_TASK_ROOT}:{MARINSKYRL_TASK_ROOT}/skyrl-train"
    shell = (
        "set -euo pipefail; "
        f"mkdir -p {shlex.quote(RAY_SPILL_DIR)}; "
        f"cd {APP_DIR}; "
        f"source {shlex.quote(MARINSKYRL_ACTIVATION_FILE)}; "
        f"export PYTHONPATH={shlex.quote(pythonpath)}:${{PYTHONPATH:-}}; "
        f"exec {shlex.join(controller)}"
    )
    return ["bash", "-c", shell]


def main() -> None:
    args, benchmark_argv = parse_args()
    validate_request(args, benchmark_argv)
    workspace = build_runtime_bundle(args.runtime_commit)
    command = task_command(args, benchmark_argv)
    resources = _gpu_resources("H100", 8, cpu=args.cpu, memory=args.memory, disk=args.disk)
    replicas = args.nodes
    coscheduling = _gpu_multinode("H100", 8, replicas)
    constraints = _gpu_constraints(
        resources.to_proto(),
        replicas=replicas,
        preemptible=False,
        target_cluster=None,
    )
    priority = job_pb2.PriorityBand.Value(f"PRIORITY_BAND_{args.priority.upper()}")
    env_vars = {
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    print(f"[grug-fixed-replay] job=/{os.environ.get('USER', 'user')}/{args.job_name}")
    print(f"[grug-fixed-replay] runtime={args.runtime_commit} image={args.task_image}")
    print(f"[grug-fixed-replay] topology={replicas}x8 H100 priority={args.priority}")
    print(f"[grug-fixed-replay] command={shlex.join(command)}", flush=True)

    if args.dry_run:
        print("[grug-fixed-replay] dry-run: validated; no job submitted", flush=True)
        return

    with open_iris_client(config_file=args.cluster_config, workspace=workspace) as client:
        job = client.submit(
            entrypoint=Entrypoint.from_command(*command),
            name=args.job_name,
            resources=resources,
            environment=EnvironmentSpec(
                env_vars=env_vars,
                extras=["gpu"],
                setup_scripts=[task_setup_script(args.runtime_commit, RuntimeProfile.FSDP)],
            ),
            constraints=constraints or None,
            coscheduling=coscheduling,
            replicas=replicas,
            max_retries_failure=0,
            priority_band=priority,
            task_image=args.task_image,
        )
        print(f"[grug-fixed-replay] submitted={job.job_id}", flush=True)


if __name__ == "__main__":
    main()
