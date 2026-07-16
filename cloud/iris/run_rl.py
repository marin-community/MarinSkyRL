#!/usr/bin/env python3
"""In-container MarinSkyRL RL training runner (Iris path).

Runs on rank 0 inside the gpu-rl container after the controller
(``start_rl_iris_controller.py``) has bootstrapped one cross-node Ray cluster and
exported ``RAY_ADDRESS``. This runner parses the RL config, resolves HF task data,
builds the SkyRL Hydra args, and execs the MarinSkyRL entrypoint attached to that
Ray cluster (SkyRL's bare ``ray.init()`` honors ``RAY_ADDRESS``).

Usage::

    python -m cloud.iris.run_rl \
        --rl_config configs/56gpu_qwen3_8b.yaml \
        --train_data '["org/my-dataset"]' \
        --model_path Qwen/Qwen3-8B \
        --job_name my_rl_run \
        --num_nodes 7 --gpus 56 --gpus_per_node 8
"""

from __future__ import annotations

import argparse
import ast
import os
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from cloud.iris.paths import PROJECT_ROOT
from cloud.iris.rl_config_translation import (
    build_skyrl_hydra_args,
    get_skyrl_command_preview,
    parse_rl_config,
)
from cloud.iris.rl_data import (
    check_rl_environment,
    compute_num_inference_engines,
    derive_skyrl_export_path,
    resolve_rl_train_data,
)


@dataclass
class LocalRLConfig:
    """Configuration for the in-container RL runner."""

    rl_config_path: str
    job_name: str
    model_path: str
    train_data: List[str] = field(default_factory=list)
    val_data: List[str] = field(default_factory=list)
    experiments_dir: str = "experiments"
    gpus: int = 4
    cpus: int = 0  # 0 = auto-detect
    # Multi-node placement. The external controller has already bootstrapped one
    # cross-node Ray cluster and exported RAY_ADDRESS; this runner ATTACHES to it,
    # and gpus_per_node drives the SkyRL placement + num_inference_engines.
    num_nodes: int = 1
    gpus_per_node: int = 0  # 0 = use `gpus`
    ray_port: int = 6379
    master_port: int = 12345
    skyrl_overrides: List[str] = field(default_factory=list)
    dry_run: bool = False
    tensor_parallel_size: int = 1  # auto-derived


class LocalRLRunner:
    """Runs SkyRL training attached to an externally-managed Ray cluster."""

    def __init__(self, config: LocalRLConfig):
        self.config = config
        self._processes: List[subprocess.Popen] = []
        self.rl_env_path: Path | None = None

    def setup(self) -> None:
        """Validate configuration and set up directories."""
        rl_env = check_rl_environment()
        if rl_env:
            print(f"RL environment found: {rl_env}")
            self.rl_env_path = rl_env
        else:
            # On Iris, sys.executable is already the gpu-rl image's RL venv python.
            self.rl_env_path = None

        skyrl_home = os.environ.get("SKYRL_HOME")
        if skyrl_home and Path(skyrl_home).exists():
            print(f"SkyRL home: {skyrl_home}")
            skyrl_train = os.path.join(skyrl_home, "skyrl-train")
            if skyrl_train not in sys.path:
                sys.path.insert(0, skyrl_train)
            pythonpath = os.environ.get("PYTHONPATH", "")
            if skyrl_train not in pythonpath:
                os.environ["PYTHONPATH"] = f"{skyrl_train}:{pythonpath}"
        else:
            print("\nWARNING: SKYRL_HOME not found! Set SKYRL_HOME to the MarinSkyRL clone.")

        experiments_dir = Path(self.config.experiments_dir).expanduser().resolve()
        experiments_dir.mkdir(parents=True, exist_ok=True)
        self.config.experiments_dir = str(experiments_dir)

        if self.config.cpus <= 0:
            self.config.cpus = os.cpu_count() or 16

        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        def handle_signal(signum, _frame):
            print(f"\nSignal {signum} received; shutting down...", file=sys.stderr)
            self.cleanup()
            sys.exit(1)

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

    def cleanup(self) -> None:
        for proc in self._processes:
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()

    def print_banner(self) -> None:
        print("=== MarinSkyRL Iris Training Runner ===")
        print(f"  Job Name: {self.config.job_name}")
        print(f"  RL Config: {self.config.rl_config_path}")
        print(f"  Model: {self.config.model_path}")
        print(f"  GPUs: {self.config.gpus}")
        print(f"  Train Data: {self.config.train_data}")
        print(f"  Experiments Dir: {self.config.experiments_dir}")
        print("=======================================")

    def run(self) -> int:
        """Execute the RL training job. Returns an exit code (0 for success)."""
        self.print_banner()

        parsed = parse_rl_config(
            self.config.rl_config_path,
            model_override=self.config.model_path,
        )
        print(f"Loaded RL config: {parsed.config_path}")
        self.config.tensor_parallel_size = parsed.tensor_parallel_size

        if self.config.train_data:
            print(f"\nResolving train_data: {self.config.train_data}")
            resolved_train = resolve_rl_train_data(self.config.train_data)
            self.config.train_data = resolved_train
            print(f"Resolved train_data: {resolved_train}")

        exp_args = self._build_exp_args()

        hpc_stub = _LocalHPCStub(
            gpus_per_node=self.config.gpus,
            cpus_per_node=self.config.cpus,
        )
        hydra_args = build_skyrl_hydra_args(parsed, exp_args, hpc_stub)

        if self.config.skyrl_overrides:
            hydra_args.extend(self.config.skyrl_overrides)

        if self.config.dry_run:
            print("\n[DRY RUN] Would execute SkyRL with:")
            print(get_skyrl_command_preview(parsed.entrypoint, hydra_args))
            return 0

        self._setup_environment(exp_args)
        return self._run_skyrl(parsed.entrypoint, hydra_args)

    def _gpus_per_node(self) -> int:
        """GPUs per node, defaulting to total `gpus` for the single-node case."""
        return self.config.gpus_per_node or self.config.gpus

    def _build_exp_args(self) -> Dict[str, Any]:
        return {
            "job_name": self.config.job_name,
            "experiments_dir": self.config.experiments_dir,
            "model_path": self.config.model_path,
            "train_data": self.config.train_data,
            "val_data": self.config.val_data,
            "num_nodes": self.config.num_nodes,
            "gpus_per_node": self._gpus_per_node(),
            "cpus_per_node": self.config.cpus,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "ray_port": self.config.ray_port,
            "master_port": self.config.master_port,
        }

    def _setup_environment(self, exp_args: Dict[str, Any]) -> None:
        """Configure environment variables for RL training."""
        os.environ["TENSOR_PARALLEL_SIZE"] = str(self.config.tensor_parallel_size)
        os.environ["NUM_INFERENCE_ENGINES"] = str(
            compute_num_inference_engines(
                self.config.num_nodes,
                self._gpus_per_node(),
                self.config.tensor_parallel_size,
            )
        )
        os.environ["POLICY_NUM_NODES"] = str(self.config.num_nodes)

        export_path = derive_skyrl_export_path(
            self.config.experiments_dir,
            self.config.job_name,
        )
        os.environ["SKYRL_EXPORT_PATH"] = export_path

        os.environ["VLLM_USE_V1"] = "1"

        wandb_dir = os.path.join(self.config.experiments_dir, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        os.environ["WANDB_DIR"] = wandb_dir

        print("\nEnvironment configured:")
        print(f"  TENSOR_PARALLEL_SIZE={os.environ['TENSOR_PARALLEL_SIZE']}")
        print(f"  NUM_INFERENCE_ENGINES={os.environ['NUM_INFERENCE_ENGINES']}")
        print(f"  SKYRL_EXPORT_PATH={export_path}")
        print(f"  WANDB_DIR={wandb_dir}")

    def _run_skyrl(self, entrypoint: str, hydra_args: List[str]) -> int:
        """Exec the SkyRL entrypoint attached to the externally-managed Ray cluster.

        The controller exported RAY_ADDRESS, so SkyRL's ``initialize_ray()`` (a bare
        ``ray.init()``) attaches to the existing cluster; this runner must NOT call
        its own ``ray.init(num_cpus=, num_gpus=)`` (forbidden when attaching).
        """
        ray_address = os.environ.get("RAY_ADDRESS")
        if not ray_address:
            print("ERROR: RAY_ADDRESS is not set. This runner attaches to a Ray cluster "
                  "bootstrapped by start_rl_iris_controller.py; run it via that controller.",
                  file=sys.stderr)
            return 1
        print(f"\nAttaching to external Ray cluster at {ray_address} "
              f"(num_nodes={self.config.num_nodes}, gpus_per_node={self._gpus_per_node()})")

        python_exe = str(self.rl_env_path / "bin" / "python") if self.rl_env_path else sys.executable
        cmd = [python_exe, "-m", entrypoint] + hydra_args

        print("\nRunning SkyRL:")
        print(f"  Entrypoint: {entrypoint}")
        print(f"  Args: {len(hydra_args)} Hydra arguments")

        skyrl_home = os.environ.get("SKYRL_HOME")
        cwd = None
        if skyrl_home:
            candidate = os.path.join(skyrl_home, "skyrl-train")
            if os.path.isdir(candidate):
                cwd = candidate
                print(f"  Working dir: {cwd}")

        print(f"\nCommand: {' '.join(cmd[:3])} [... {len(cmd) - 3} more args]")
        sys.stdout.flush()

        proc = subprocess.Popen(cmd, cwd=cwd)
        self._processes.append(proc)
        return proc.wait()


@dataclass
class _LocalHPCStub:
    """Minimal HPC-like object for build_skyrl_hydra_args compatibility."""
    gpus_per_node: int = 4
    cpus_per_node: int = 48
    name: str = "local"


def parse_list_arg(value: str) -> List[str]:
    """Parse a list argument from the CLI (JSON or Python literal)."""
    if not value:
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
        return [str(parsed)]
    except (ValueError, SyntaxError):
        return [value]


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MarinSkyRL training inside the Iris container, attached to the "
        "controller-bootstrapped Ray cluster.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--rl_config", required=True, help="Path to a SkyRL config YAML.")
    parser.add_argument("--rl-config", dest="rl_config", help=argparse.SUPPRESS)

    parser.add_argument("--model_path", required=True, help="Model path or HuggingFace ID.")
    parser.add_argument("--model-path", dest="model_path", help=argparse.SUPPRESS)

    parser.add_argument("--job_name", required=True, help="Name for this training job.")
    parser.add_argument("--job-name", dest="job_name", help=argparse.SUPPRESS)

    parser.add_argument("--train_data", default="[]", help="Training data paths as a JSON list.")
    parser.add_argument("--train-data", dest="train_data", help=argparse.SUPPRESS)

    parser.add_argument("--val_data", default="[]", help="Validation data paths as a JSON list.")
    parser.add_argument("--val-data", dest="val_data", help=argparse.SUPPRESS)

    parser.add_argument("--gpus", type=int, default=4, help="Total number of GPUs to use.")
    parser.add_argument("--cpus", type=int, default=0, help="Number of CPUs (0 = auto-detect).")

    parser.add_argument(
        "--num_nodes", type=int, default=1,
        help="Number of nodes. >1 attaches to an external Ray cluster via RAY_ADDRESS.",
    )
    parser.add_argument("--num-nodes", dest="num_nodes", help=argparse.SUPPRESS)

    parser.add_argument(
        "--gpus_per_node", type=int, default=0,
        help="GPUs per node (0 = use --gpus; set for multi-node placement).",
    )
    parser.add_argument("--gpus-per-node", dest="gpus_per_node", help=argparse.SUPPRESS)

    parser.add_argument("--ray_port", type=int, default=6379, help="Port for the Ray cluster.")
    parser.add_argument("--ray-port", dest="ray_port", help=argparse.SUPPRESS)

    parser.add_argument("--master_port", type=int, default=12345, help="Master port for distributed training.")
    parser.add_argument("--master-port", dest="master_port", help=argparse.SUPPRESS)

    parser.add_argument("--skyrl_override", action="append", default=[], help="SkyRL Hydra override (repeatable).")
    parser.add_argument("--skyrl-override", dest="skyrl_override", action="append", help=argparse.SUPPRESS)

    parser.add_argument(
        "--experiments_dir", default=str(PROJECT_ROOT / "experiments"),
        help="Directory for experiment outputs.",
    )
    parser.add_argument("--experiments-dir", dest="experiments_dir", help=argparse.SUPPRESS)

    parser.add_argument("--dry_run", action="store_true", help="Print config and command without running.")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", help=argparse.SUPPRESS)

    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    train_data = parse_list_arg(args.train_data)
    val_data = parse_list_arg(args.val_data)
    skyrl_overrides = args.skyrl_override or []

    config = LocalRLConfig(
        rl_config_path=args.rl_config,
        job_name=args.job_name,
        model_path=args.model_path,
        train_data=train_data,
        val_data=val_data,
        experiments_dir=args.experiments_dir,
        gpus=args.gpus,
        cpus=args.cpus,
        num_nodes=int(args.num_nodes),
        gpus_per_node=int(args.gpus_per_node),
        ray_port=args.ray_port,
        master_port=args.master_port,
        skyrl_overrides=skyrl_overrides,
        dry_run=args.dry_run,
    )

    runner = LocalRLRunner(config)
    runner.setup()
    sys.exit(runner.run())


if __name__ == "__main__":
    main()
