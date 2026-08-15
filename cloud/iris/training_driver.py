#!/usr/bin/env python3
"""Drive SkyRL training from rank zero of an Iris-managed Ray cluster.

Runs on rank 0 inside the frozen Iris task environment after the controller
(``task_runtime.py``) has bootstrapped one cross-node Ray cluster and
exported ``RAY_ADDRESS``. This runner parses the RL config, resolves HF task data,
builds the SkyRL Hydra args, and execs the MarinSkyRL entrypoint attached to that
Ray cluster (SkyRL's bare ``ray.init()`` honors ``RAY_ADDRESS``).

Usage::

    python -m cloud.iris.training_driver \
        --rl_config configs/56gpu_qwen3_8b.yaml \
        --train_data '["org/my-dataset"]' \
        --model_path Qwen/Qwen3-8B \
        --job_name my_rl_run \
        --num_nodes 7 --gpus 56 --gpus_per_node 8
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List

from cloud.iris.artifacts import fs_and_path
from cloud.iris.paths import PROJECT_ROOT
from cloud.iris.rl_config_translation import (
    apply_context_budget_overrides,
    build_checkpoint_export_hydra_args,
    build_skyrl_hydra_args,
    get_skyrl_command_preview,
    materialize_rl_config,
    parse_checkpoint_export_config,
    parse_rl_config,
    write_resolved_context_budget,
)
from cloud.iris.rl_data import (
    check_rl_environment,
    compute_num_inference_engines,
    derive_skyrl_export_path,
    resolve_rl_train_data,
)
from cloud.iris.storage_policy import hydra_override_value
from marinskyrl.resource_locator import model_source_for_path
from cloud.iris.runtime_environment import CHECKPOINT_EXPORT_ENTRYPOINT


@dataclass
class LocalRLConfig:
    """Configuration for the in-container RL runner."""

    rl_config_path: str
    job_name: str
    model_path: str
    entrypoint: str | None = None
    model_source_uri: str | None = None
    model_source_identity: str | None = None
    train_data: List[str] = field(default_factory=list)
    val_data: List[str] = field(default_factory=list)
    experiments_dir: str = "experiments"
    resolved_config_uri: str | None = None
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
    # --- Cross-cluster ingress (Exp2 opencode-RL literal capture) ---
    # All default to the OFF/direct value so an all-defaults run stands up NO proxy,
    # registers NO endpoint, and touches NO env — byte-identical to today.
    ingress_mode: str = "direct"  # "direct" (off) | "controller"
    ingress_host: str = ""  # public controller-ingress host (marin: iris.oa.dev)
    record_literal: bool = False  # co-locate harbor RecordProxy for literal.jsonl capture
    target_cluster: str = ""  # set => federated: mint at the PARENT for the mirrored endpoint
    parent_controller_config: str = ""  # marin.yaml path for federated parent-minting
    vllm_http_port: int = 8000  # local vLLM HTTP endpoint (= generator.http_endpoint_port)

    def __post_init__(self) -> None:
        model_source_for_path(self.model_path, self.model_source_uri, self.model_source_identity)


class LocalRLRunner:
    """Runs SkyRL training attached to an externally-managed Ray cluster."""

    def __init__(self, config: LocalRLConfig):
        self.config = config
        self._processes: List[subprocess.Popen] = []
        self.rl_env_path: Path | None = None
        # Set by _ingress_context when ingress_mode=controller mints a capability
        # URL. Threaded into the SkyRL Hydra cfg (see run()) so it crosses the Ray
        # .remote() boundary as DATA — os.environ mutations here do NOT reach the
        # pre-existing Ray workers where TerminalBenchGenerator is constructed.
        self._minted_agent_api_base: str | None = None
        # Set by _ingress_context when record_literal stands up the co-located
        # RecordProxy. Threaded into the SkyRL Hydra cfg (see run()) for the SAME
        # process-boundary reason as _minted_agent_api_base: literal_proxy_utils
        # publishes the log path via os.environ["OTAGENT_LITERAL_LOG_PATH"] in THIS
        # driver, but the generator (which reads it to correlate opencode rollout
        # details / rebuild chat_history) runs in a pre-existing Ray worker that never
        # inherits this env → without the cfg thread every opencode trajectory loses
        # its logprobs and TIS degrades on 100% of the batch.
        self._literal_log_path: str | None = None

    def setup(self) -> None:
        """Validate configuration and set up directories."""
        rl_env = check_rl_environment()
        if rl_env:
            print(f"RL environment found: {rl_env}")
            self.rl_env_path = rl_env
        else:
            # On Iris, sys.executable is already the frozen task venv's Python.
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

    def _context_budget_artifact_destination(self, parsed, skyrl_overrides: List[str]) -> Path | str:
        """Choose the durable Harbor bundle when this run writes one."""
        trials_dir = (parsed.terminal_bench or {}).get("trials_dir")
        override_trials_dir = hydra_override_value(skyrl_overrides, "terminal_bench_config.trials_dir")
        if override_trials_dir is not None:
            trials_dir = override_trials_dir
        if trials_dir and str(trials_dir).startswith(("s3://", "gs://")) and not self.config.dry_run:
            return f"{str(trials_dir).rstrip('/')}/resolved-context-budget.json"
        return Path(self.config.experiments_dir) / self.config.job_name / "resolved-context-budget.json"

    def _record_context_budget(self, parsed, skyrl_overrides: List[str]) -> Path | str:
        """Print and persist the token contract resolved for this training run."""
        budget = parsed.context_budget
        artifact = self._context_budget_artifact_destination(parsed, skyrl_overrides)
        write_resolved_context_budget(budget, artifact, parsed.config_path)
        print("Resolved context budget:")
        print(f"  request window: {budget.request_window_tokens}")
        print(f"  client input:   {budget.max_input_tokens}")
        print(f"  turn output:    {budget.max_new_tokens_per_turn}")
        print(f"  max turns:      {budget.max_turns}")
        print(f"  artifact:       {artifact}")
        return artifact

    def _write_resolved_config(self, entrypoint: str, hydra_args: List[str], source_config: Path) -> None:
        if not self.config.resolved_config_uri:
            return
        filesystem, path = fs_and_path(self.config.resolved_config_uri)
        with filesystem.open(path, "w") as destination:
            json.dump(
                {
                    "entrypoint": entrypoint,
                    "hydra_args": hydra_args,
                    "source_config": str(source_config),
                },
                destination,
                sort_keys=True,
            )

    def _run_checkpoint_export(self, rl_config_path: Path, exp_args: dict, hpc: "_LocalHPCStub") -> int:
        """Run the policy-only conversion pipeline without training setup."""
        parsed = parse_checkpoint_export_config(rl_config_path, model_override=self.config.model_path)
        hydra_args = build_checkpoint_export_hydra_args(parsed, exp_args, hpc)
        hydra_args.extend(self.config.skyrl_overrides)
        print(f"Loaded RL config: {parsed.config_path}")
        self._write_resolved_config(CHECKPOINT_EXPORT_ENTRYPOINT, hydra_args, parsed.config_path)
        if self.config.dry_run:
            print("\n[DRY RUN] Would execute SkyRL with:")
            print(get_skyrl_command_preview(CHECKPOINT_EXPORT_ENTRYPOINT, hydra_args))
            return 0
        return self._run_skyrl(CHECKPOINT_EXPORT_ENTRYPOINT, hydra_args)

    def run(self) -> int:
        """Execute the RL training job. Returns an exit code (0 for success)."""
        self.print_banner()

        rl_config_path = materialize_rl_config(self.config.rl_config_path)
        exp_args = self._build_exp_args()
        hpc_stub = _LocalHPCStub(
            gpus_per_node=self.config.gpus,
            cpus_per_node=self.config.cpus,
        )
        if self.config.entrypoint == CHECKPOINT_EXPORT_ENTRYPOINT:
            return self._run_checkpoint_export(rl_config_path, exp_args, hpc_stub)

        parsed = parse_rl_config(
            rl_config_path,
            model_override=self.config.model_path,
        )
        parsed, skyrl_overrides = apply_context_budget_overrides(parsed, self.config.skyrl_overrides)
        entrypoint = self.config.entrypoint or parsed.entrypoint
        self.config.tensor_parallel_size = parsed.tensor_parallel_size
        if self.config.train_data:
            print(f"\nResolving train_data (kind={parsed.data_kind}): {self.config.train_data}")
            resolved_train = resolve_rl_train_data(self.config.train_data, kind=parsed.data_kind)
            self.config.train_data = resolved_train
            exp_args["train_data"] = resolved_train
            print(f"Resolved train_data: {resolved_train}")
        hydra_args = build_skyrl_hydra_args(parsed, exp_args, hpc_stub)
        self._record_context_budget(parsed, skyrl_overrides)

        print(f"Loaded RL config: {parsed.config_path}")

        if skyrl_overrides:
            hydra_args.extend(skyrl_overrides)

        self._write_resolved_config(entrypoint, hydra_args, parsed.config_path)

        if self.config.dry_run:
            print("\n[DRY RUN] Would execute SkyRL with:")
            print(get_skyrl_command_preview(entrypoint, hydra_args))
            return 0

        self._setup_environment(exp_args)
        # Cross-cluster ingress (opencode-RL literal capture): when enabled, stand up
        # the co-located RecordProxy + register the endpoint + mint the (parent, when
        # federated) capability URL and publish it as HARBOR_MODEL_ENDPOINT BEFORE the
        # SkyRL subprocess is spawned, so the generator (which inherits this env) points
        # opencode at iris.oa.dev/proxy/t/<token>/... and the sandbox traffic flows
        # controller -> RecordProxy -> vLLM. The default (direct) path is a null CM.
        with self._ingress_context():
            # If controller-ingress minted a capability URL, thread it into the cfg
            # so it crosses the Ray .remote() boundary as DATA (the generator is built
            # inside a Ray worker that never inherits this process's HARBOR_MODEL_ENDPOINT
            # env — see _ingress_context / __init__). Snapshot cadence matches the
            # existing design (one api_base string baked for the job's lifetime).
            if self._minted_agent_api_base:
                hydra_args = hydra_args + [f"++terminal_bench_config.agent_api_base={self._minted_agent_api_base}"]
            # Thread the RecordProxy log path as cfg DATA too (same Ray boundary): the
            # generator resolves the shared literal log from
            # terminal_bench_config.literal_log_path (env fallback) to correlate each
            # opencode trial's token_ids/logprobs + rebuild its chat_history. Without
            # this the worker's os.environ lacks the path and TIS skips 100% of the batch.
            if self._literal_log_path:
                hydra_args = hydra_args + [f"++terminal_bench_config.literal_log_path={self._literal_log_path}"]
            return self._run_skyrl(entrypoint, hydra_args)

    @contextlib.contextmanager
    def _ingress_context(self) -> Iterator[None]:
        """Guarded controller-ingress standup around the SkyRL subprocess.

        Mirrors the RLJobRunner controller-ingress path, consolidated into the
        canonical MarinSkyRL runner:

          1. co-locate harbor's RecordProxy (``record_literal``) in front of the local
             vLLM HTTP endpoint so agent completions are captured to ``literal.jsonl``;
          2. register that upstream (RecordProxy, else raw vLLM) with the in-pod iris
             controller under ENDPOINT_ACCESS_LINK (leased, kept alive for the run);
          3. mint a scoped capability token — at the PARENT (marin) for the MIRRORED
             endpoint when ``target_cluster`` is set (federated: cw-signed tokens 401 at
             iris.oa.dev), else at the local controller — and publish the
             ``/proxy/t/<token>/<name>/v1`` URL as ``HARBOR_MODEL_ENDPOINT`` +
             inject the inert sandbox agent key.

        Default ``ingress_mode=direct`` yields immediately (no proxy, no register, no
        env mutation) — byte-identical to today.
        """
        # Agent auth is DECOUPLED from ingress mode. Installed OpenAI-compatible agents
        # (opencode) refuse to start without a non-empty api_key; for the inert-capability
        # model that key is just a dummy placeholder. That is a per-AGENT requirement, NOT a
        # controller-ingress one — the served endpoint may be reached DIRECTLY (the agent's
        # api_base kwarg) with no controller/proxy at all. So publish the dummy key here,
        # regardless of ingress_mode. It never clobbers a real host OPENAI_API_KEY (see
        # inject_ingress_agent_key: OPENCODE_DUMMY_KEY set unconditionally, real keys only
        # setdefault'd). Previously this lived inside the controller branch only, so opencode
        # on ingress_mode=direct got NO key -> refused to start -> zero requests -> silent
        # empty rollouts (engine idle, "Sandbox not found" symptom).
        from cloud.iris.ingress_utils import inject_ingress_agent_key

        inject_ingress_agent_key()

        if self.config.ingress_mode != "controller":
            yield
            return

        # Lazy imports: these modules hard-import iris/harbor only at call time, and are
        # only needed on the (opt-in) controller-ingress path.
        from cloud.iris.ingress_utils import (
            capability_api_base,
            controller_registration_plan,
            federated_capability_api_base,
            register_controller_endpoint,
        )
        from cloud.iris.literal_proxy_utils import (
            DEFAULT_LITERAL_PROXY_PORT,
            maybe_serve_literal_proxy,
        )

        if not self.config.ingress_host:
            raise ValueError(
                "ingress_mode=controller requires --ingress_host (the public "
                "controller-ingress host; iris.oa.dev for the federated CoreWeave path)."
            )
        # Federated parent-minting reads the parent (marin) controller config from the
        # env the launcher forwards; surface it here so a misconfig fails loud early.
        if self.config.target_cluster:
            from cloud.iris.ingress_utils import PARENT_CONTROLLER_CONFIG_ENV

            if self.config.parent_controller_config:
                os.environ.setdefault(PARENT_CONTROLLER_CONFIG_ENV, self.config.parent_controller_config)
            if not os.environ.get(PARENT_CONTROLLER_CONFIG_ENV):
                raise ValueError(
                    "federated ingress (target_cluster set) requires the parent marin "
                    f"controller config via {PARENT_CONTROLLER_CONFIG_ENV} (or "
                    "--parent_controller_config); needed to mint at iris.oa.dev."
                )

        endpoint_name, register_address = controller_registration_plan(
            self.config.job_name,
            record_literal=self.config.record_literal,
            proxy_port=DEFAULT_LITERAL_PROXY_PORT,
            vllm_port=self.config.vllm_http_port,
        )
        vllm_local = f"http://localhost:{self.config.vllm_http_port}/v1"
        # RecordProxy binds 0.0.0.0 so the (remote) controller reaches it at
        # IRIS_ADVERTISE_HOST; record_literal off => maybe_serve_literal_proxy is a null
        # CM and the plan registered raw vLLM's port instead.
        with maybe_serve_literal_proxy(
            self.config.record_literal,
            vllm_local,
            experiments_dir=self.config.experiments_dir,
            job_name=self.config.job_name,
            host="0.0.0.0",
        ):
            registration = register_controller_endpoint(endpoint_name, register_address)
            try:
                if self.config.target_cluster:
                    api_base = federated_capability_api_base(endpoint_name, ingress_host=self.config.ingress_host)
                    mint_where = f"PARENT (federated -> {self.config.target_cluster})"
                else:
                    api_base = capability_api_base(self.config.ingress_host, endpoint_name)
                    mint_where = "local controller"
                # Publish the capability URL as the harbor-specific HARBOR_MODEL_ENDPOINT.
                # opencode (harbor agents/installed/opencode.py::_build_register_config_command)
                # reads its served-model baseURL from HARBOR_MODEL_ENDPOINT. We deliberately do
                # NOT touch OPENAI_BASE_URL: that var is reserved for genuine OpenAI traffic (the
                # LLM-judge verifiers on the worker read it), so overloading it with a vLLM
                # endpoint would silently misroute every judge call to vLLM.
                os.environ["HARBOR_MODEL_ENDPOINT"] = api_base
                # Also thread the minted URL through the SkyRL Hydra cfg. run() injects
                # ``++terminal_bench_config.agent_api_base=<api_base>`` from this, so the
                # value reaches the Ray tasks/actors (skyrl_entrypoint, RolloutCoordinator)
                # where TerminalBenchGenerator is built. The env var alone is insufficient:
                # this runner ATTACHES to a Ray cluster the controller started BEFORE the
                # mint, so its workers never inherit HARBOR_MODEL_ENDPOINT from this process
                # and the generator would fall back to the pod-local (unreachable) vLLM URL.
                self._minted_agent_api_base = api_base
                # Capture the RecordProxy log path that maybe_serve_literal_proxy just
                # published on os.environ, to thread it into the cfg alongside
                # agent_api_base (same Ray process-boundary — see run()). None when
                # record_literal is off (null CM), keeping the direct path byte-identical.
                self._literal_log_path = os.environ.get("OTAGENT_LITERAL_LOG_PATH")
                injected = True  # dummy key already published above (decoupled from ingress)
                print(
                    f"[training-driver] ingress_mode=controller record_literal="
                    f"{self.config.record_literal} target_cluster="
                    f"{self.config.target_cluster or '(direct)'}: registered "
                    f"{endpoint_name} -> {register_address} "
                    f"(id={registration.endpoint_id}, access=LINK); minted at {mint_where}; "
                    f"HARBOR_MODEL_ENDPOINT=/proxy/t/<token>/{endpoint_name}/v1 "
                    f"(dummy key injected={injected})",
                    flush=True,
                )
                yield
            finally:
                registration.close()

    def _gpus_per_node(self) -> int:
        """GPUs per node, defaulting to total `gpus` for the single-node case."""
        return self.config.gpus_per_node or self.config.gpus

    def _build_exp_args(self) -> Dict[str, Any]:
        return {
            "job_name": self.config.job_name,
            "experiments_dir": self.config.experiments_dir,
            "model_path": self.config.model_path,
            "model_source_uri": self.config.model_source_uri,
            "model_source_identity": self.config.model_source_identity,
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
            print(
                "ERROR: RAY_ADDRESS is not set. This runner attaches to a Ray cluster "
                "bootstrapped by task_runtime.py; run it via that controller.",
                file=sys.stderr,
            )
            return 1
        print(
            f"\nAttaching to external Ray cluster at {ray_address} "
            f"(num_nodes={self.config.num_nodes}, gpus_per_node={self._gpus_per_node()})"
        )

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
    parser.add_argument("--entrypoint", default=None, help="Override the RL config entrypoint.")

    parser.add_argument("--model_path", required=True, help="Model path or HuggingFace ID.")
    parser.add_argument("--model-path", dest="model_path", help=argparse.SUPPRESS)
    parser.add_argument("--model-source-uri")
    parser.add_argument("--model-source-identity")

    parser.add_argument("--job_name", required=True, help="Name for this training job.")
    parser.add_argument("--job-name", dest="job_name", help=argparse.SUPPRESS)

    parser.add_argument("--train_data", default="[]", help="Training data paths as a JSON list.")
    parser.add_argument("--train-data", dest="train_data", help=argparse.SUPPRESS)

    parser.add_argument("--val_data", default="[]", help="Validation data paths as a JSON list.")
    parser.add_argument("--val-data", dest="val_data", help=argparse.SUPPRESS)

    parser.add_argument("--gpus", type=int, default=4, help="Total number of GPUs to use.")
    parser.add_argument("--cpus", type=int, default=0, help="Number of CPUs (0 = auto-detect).")

    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="Number of nodes. >1 attaches to an external Ray cluster via RAY_ADDRESS.",
    )
    parser.add_argument("--num-nodes", dest="num_nodes", help=argparse.SUPPRESS)

    parser.add_argument(
        "--gpus_per_node",
        type=int,
        default=0,
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
        "--experiments_dir",
        default=str(PROJECT_ROOT / "experiments"),
        help="Directory for experiment outputs.",
    )
    parser.add_argument("--experiments-dir", dest="experiments_dir", help=argparse.SUPPRESS)
    parser.add_argument(
        "--resolved-config-uri",
        default=None,
        help="Durable JSON destination for the exact SkyRL entry point and Hydra arguments.",
    )

    parser.add_argument("--dry_run", action="store_true", help="Print config and command without running.")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", help=argparse.SUPPRESS)

    # --- Cross-cluster ingress (Exp2 opencode-RL literal capture) --- #
    # Forwarded by the launcher under --ingress-mode controller. Default off => the
    # standup context is a null CM (byte-identical).
    parser.add_argument(
        "--ingress_mode",
        default="direct",
        choices=["direct", "controller"],
        help="'controller' stands up the RecordProxy + registers the endpoint + mints "
        "the capability URL and publishes it as HARBOR_MODEL_ENDPOINT. Default 'direct' off.",
    )
    parser.add_argument("--ingress-mode", dest="ingress_mode", help=argparse.SUPPRESS)
    parser.add_argument(
        "--ingress_host",
        default="",
        help="Public controller-ingress host for the capability URL (iris.oa.dev for "
        "the federated CoreWeave path). Required with --ingress_mode controller.",
    )
    parser.add_argument("--ingress-host", dest="ingress_host", help=argparse.SUPPRESS)
    parser.add_argument(
        "--record_literal",
        action="store_true",
        help="Co-locate harbor's RecordProxy in front of vLLM to capture literal.jsonl.",
    )
    parser.add_argument("--record-literal", dest="record_literal", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--target_cluster",
        default="",
        help="Set for federated ingress: mint the capability token at the PARENT (marin) "
        "for the mirrored endpoint (a peer-signed token 401s at iris.oa.dev).",
    )
    parser.add_argument("--target-cluster", dest="target_cluster", help=argparse.SUPPRESS)
    parser.add_argument(
        "--parent_controller_config",
        default="",
        help="Parent (marin) cluster YAML for federated parent-minting; also honored via "
        "the OTAGENT_PARENT_CONTROLLER_CONFIG env.",
    )
    parser.add_argument("--parent-controller-config", dest="parent_controller_config", help=argparse.SUPPRESS)
    parser.add_argument(
        "--vllm_http_port",
        type=int,
        default=8000,
        help="Local vLLM HTTP endpoint port the RecordProxy relays to (= generator.http_endpoint_port; default 8000).",
    )
    parser.add_argument("--vllm-http-port", dest="vllm_http_port", type=int, help=argparse.SUPPRESS)

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
        entrypoint=args.entrypoint,
        model_source_uri=args.model_source_uri,
        model_source_identity=args.model_source_identity,
        train_data=train_data,
        val_data=val_data,
        experiments_dir=args.experiments_dir,
        resolved_config_uri=args.resolved_config_uri,
        gpus=args.gpus,
        cpus=args.cpus,
        num_nodes=int(args.num_nodes),
        gpus_per_node=int(args.gpus_per_node),
        ray_port=args.ray_port,
        master_port=args.master_port,
        skyrl_overrides=skyrl_overrides,
        dry_run=args.dry_run,
        ingress_mode=args.ingress_mode,
        ingress_host=args.ingress_host,
        record_literal=bool(args.record_literal),
        target_cluster=args.target_cluster,
        parent_controller_config=args.parent_controller_config,
        vllm_http_port=int(args.vllm_http_port),
    )

    runner = LocalRLRunner(config)
    runner.setup()
    sys.exit(runner.run())


if __name__ == "__main__":
    main()
