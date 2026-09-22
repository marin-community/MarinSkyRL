#!/usr/bin/env python3
"""Resolve task-local inputs and run SkyRL from one Hydra launch document.

This process runs on rank zero after ``task_runtime.py`` has bootstrapped the
cross-node Ray cluster. The only process boundary is ``--config``; staged model
paths, resolved datasets, ingress values, and SkyRL settings remain structured
configuration throughout the launch.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, List

from omegaconf import DictConfig, OmegaConf

from cloud.iris.artifacts import fs_and_path
from cloud.iris.rl_data import (
    check_rl_environment,
    compute_num_inference_engines,
    resolve_rl_train_data_with_sources,
)
from marinskyrl.process_diagnostics import ProcessOutcomeKind, write_process_outcome
from marinskyrl.resource_locator import model_source_for_path
from cloud.iris.launch_config import RunMode, load_launch_config
from cloud.iris.rl_config_translation import TaskLocalSkyRLValues, apply_task_local_values


@dataclass
class LocalRLConfig:
    """Configuration for the in-container RL runner."""

    job_name: str
    model_path: str
    model_source_uri: str | None = None
    model_source_identity: str | None = None
    train_data: List[str | dict[str, Any]] = field(default_factory=list)
    val_data: List[str | dict[str, Any]] = field(default_factory=list)
    experiments_dir: str = "experiments"
    resolved_config_uri: str | None = None
    gpus: int = 4
    # Multi-node placement. The external controller has already bootstrapped one
    # cross-node Ray cluster and exported RAY_ADDRESS; this runner ATTACHES to it,
    # and gpus_per_node drives the SkyRL placement + num_inference_engines.
    num_nodes: int = 1
    gpus_per_node: int = 0  # 0 = use `gpus`
    # --- Cross-cluster ingress (Exp2 opencode-RL literal capture) ---
    # All default to the OFF/direct value so an all-defaults run stands up NO proxy,
    # registers NO endpoint, and touches NO env — byte-identical to today.
    ingress_mode: str = "direct"  # "direct" (off) | "controller"
    ingress_host: str = ""  # public controller-ingress host (marin: iris.oa.dev)
    record_literal: bool = False  # co-locate harbor RecordProxy for literal.jsonl capture
    target_cluster: str = ""  # set => federated: mint at the PARENT for the mirrored endpoint
    parent_controller_config: str = ""  # marin.yaml path for federated parent-minting
    vllm_http_port: int = 8000
    tensor_parallel_size: int = 1
    launch_config: DictConfig | None = None

    def __post_init__(self) -> None:
        model_source_for_path(self.model_path, self.model_source_uri, self.model_source_identity)


class LocalRLRunner:
    """Runs SkyRL training attached to an externally-managed Ray cluster."""

    def __init__(self, config: LocalRLConfig):
        self.config = config
        self._train_data_sources = list(config.train_data)
        self._val_data_sources = list(config.val_data)
        self._processes: List[subprocess.Popen] = []
        self.rl_env_path: Path | None = None
        # Set by _ingress_context when ingress_mode=controller mints a capability
        # URL. Threaded into the SkyRL Hydra cfg (see run()) so it crosses the Ray
        # .remote() boundary as DATA — os.environ mutations here do NOT reach the
        # pre-existing Ray workers where HarborTrajectoryRunner is constructed.
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
        print("  Launch Config: loaded")
        print(f"  Model: {self.config.model_path}")
        print(f"  GPUs: {self.config.gpus}")
        print(f"  Train Data: {self.config.train_data}")
        print(f"  Validation Data: {self.config.val_data}")
        print(f"  Experiments Dir: {self.config.experiments_dir}")
        print("=======================================")

    def _write_resolved_config(self, config: DictConfig) -> None:
        if not self.config.resolved_config_uri:
            return
        filesystem, path = fs_and_path(self.config.resolved_config_uri)
        with filesystem.open(path, "w") as destination:
            json.dump(
                {
                    "config": OmegaConf.to_container(config, resolve=True),
                    "train_data_sources": self._train_data_sources,
                    "val_data_sources": self._val_data_sources,
                },
                destination,
                sort_keys=True,
            )

    def _resolve_data_inputs(self, data_kind: str) -> None:
        for role, attribute in (("train", "train_data"), ("validation", "val_data")):
            values = getattr(self.config, attribute)
            if not values:
                continue
            print(f"\nResolving {role} data (kind={data_kind}): {values}")
            resolved = resolve_rl_train_data_with_sources(values, kind=data_kind)
            paths = list(resolved.paths)
            sources = list(resolved.sources)
            setattr(self.config, attribute, paths)
            setattr(self, f"_{attribute}_sources", sources)
            print(f"Resolved {role} data: {paths}")

    def run(self) -> int:
        """Execute the RL training job. Returns an exit code (0 for success)."""
        self.print_banner()

        launch_config = self.config.launch_config
        if launch_config is None:
            raise ValueError("training driver requires a loaded launch config")
        skyrl_config = OmegaConf.create(OmegaConf.to_container(launch_config.skyrl, resolve=False))
        if launch_config.run.mode == RunMode.CHECKPOINT_EXPORT:
            self._write_resolved_config(launch_config)
            return self._run_skyrl(launch_config)
        self._resolve_data_inputs(str(launch_config.inputs.data_kind))
        terminal_bench_data = skyrl_config.get("data", {}).get("terminal_bench_data", ())
        if terminal_bench_data:
            terminal_bench_data = tuple(
                resolve_rl_train_data_with_sources(list(terminal_bench_data), kind="tasks").paths
            )
        self._setup_environment()
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
            # Thread the RecordProxy log path as cfg DATA too (same Ray boundary): the
            # generator resolves the shared literal log from
            # terminal_bench_config.literal_log_path (env fallback) to correlate each
            # opencode trial's token_ids/logprobs + rebuild its chat_history. Without
            # this the worker's os.environ lacks the path and TIS skips 100% of the batch.
            skyrl_config = apply_task_local_values(
                skyrl_config,
                TaskLocalSkyRLValues(
                    train_data=tuple(self.config.train_data),
                    validation_data=tuple(self.config.val_data),
                    terminal_bench_data=terminal_bench_data,
                    agent_api_base=self._minted_agent_api_base,
                    literal_log_path=self._literal_log_path,
                ),
            )
            launch_config.skyrl = skyrl_config
            self._write_resolved_config(launch_config)
            return self._run_skyrl(launch_config)

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
            DEFAULT_LITERAL_PROXY_HOST,
            maybe_serve_literal_proxy,
            select_literal_proxy_port,
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

        proxy_port = select_literal_proxy_port(self.config.job_name, host=DEFAULT_LITERAL_PROXY_HOST)
        endpoint_name, register_address = controller_registration_plan(
            self.config.job_name,
            record_literal=self.config.record_literal,
            proxy_port=proxy_port,
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
            host=DEFAULT_LITERAL_PROXY_HOST,
            port=proxy_port,
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
                # Also thread the minted URL through the structured SkyRL config so the
                # value reaches the Ray tasks/actors (skyrl_entrypoint, RolloutCoordinator)
                # where HarborTrajectoryRunner is built. The env var alone is insufficient:
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

    def _setup_environment(self) -> None:
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

        os.environ["VLLM_USE_V1"] = "1"

        wandb_dir = os.path.join(self.config.experiments_dir, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        os.environ["WANDB_DIR"] = wandb_dir

        print("\nEnvironment configured:")
        print(f"  TENSOR_PARALLEL_SIZE={os.environ['TENSOR_PARALLEL_SIZE']}")
        print(f"  NUM_INFERENCE_ENGINES={os.environ['NUM_INFERENCE_ENGINES']}")
        print(f"  WANDB_DIR={wandb_dir}")

    def _run_skyrl(self, launch_config: DictConfig) -> int:
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
        config_path = Path(tempfile.gettempdir()) / "marinskyrl" / "skyrl-launch.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(launch_config, config_path, resolve=True)
        cmd = [python_exe, "-m", "cloud.iris.skyrl_entrypoint", "--config", str(config_path)]

        print("\nRunning SkyRL:")
        print(f"  Entrypoint: {launch_config.runtime.entrypoint}")
        print(f"  Config: {config_path}")

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
        returncode = proc.wait()
        outcome, receipt = write_process_outcome(
            "skyrl-entrypoint",
            returncode,
            pid=proc.pid,
            metadata={"entrypoint": launch_config.runtime.entrypoint},
        )
        if outcome.kind is ProcessOutcomeKind.SIGNAL:
            print(
                f"SkyRL entrypoint pid={proc.pid} terminated by {outcome.signal_name} "
                f"(raw_returncode={returncode}, exit_code={outcome.public_exit_code}, receipt={receipt})",
                file=sys.stderr,
                flush=True,
            )
        return outcome.public_exit_code


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    return parser


def main() -> None:
    args = create_parser().parse_args()
    launch_config = load_launch_config(args.config)
    allocation = launch_config.iris.allocation
    config = LocalRLConfig(
        job_name=str(launch_config.iris.job_name),
        model_path=str(launch_config.skyrl.trainer.policy.model.path),
        model_source_uri=str(launch_config.inputs.model.uri),
        model_source_identity=str(launch_config.inputs.model.identity),
        train_data=list(launch_config.inputs.train_data),
        val_data=list(launch_config.inputs.validation_data),
        experiments_dir=str(launch_config.runtime.experiments_dir),
        resolved_config_uri=str(launch_config.artifacts.resolved_config_uri),
        gpus=int(allocation.num_nodes * allocation.gpus_per_node),
        num_nodes=int(allocation.num_nodes),
        gpus_per_node=int(allocation.gpus_per_node),
        tensor_parallel_size=int(launch_config.skyrl.generator.inference_engine_tensor_parallel_size),
        ingress_mode=str(launch_config.ingress.mode),
        ingress_host=str(launch_config.ingress.host),
        record_literal=bool(launch_config.ingress.record_literal),
        target_cluster=str(launch_config.iris.target_cluster or ""),
        parent_controller_config=str(launch_config.iris.parent_cluster_config or ""),
        vllm_http_port=int(launch_config.ingress.vllm_http_port),
        launch_config=launch_config,
    )

    runner = LocalRLRunner(config)
    runner.setup()
    sys.exit(runner.run())


if __name__ == "__main__":
    main()
