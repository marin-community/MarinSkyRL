"""K-actor rollout fan-out for terminal-bench RL orchestration.

This module implements the gated, default-OFF rollout fan-out described in
``notes/ot-agent/RL/architecture/skyrl_harbor_rollout_fanout_design.md``.

Motivation (one constraint, not two bugs): the whole rollout-orchestration
tier — the ``submit_batch`` create_task storm, the per-trial Harbor/Terminus2
coroutine bodies, litellm, the ``asyncio.gather`` reconverge, AND the
post-gather token/logprob/reward processing — runs on a SINGLE asyncio loop
inside ONE ``ray::skyrl_entrypoint`` task process, pinning one CPU core while
the rest sit idle. asyncio gives I/O concurrency but no parallelism for the
per-task Python work, and the entrypoint is a ``@ray.remote`` *task* (no
``max_concurrency`` knob).

The fix is more *processes*. We insert a pool of K ``RolloutCoordinator`` Ray
actors between the trainer and the trajectory runner:

  * Each actor builds its OWN ``HarborTrajectoryRunner`` scoped to the process
    with ``n_concurrent_trials // K`` and ``daytona connection_pool_maxsize // K``
    so the per-process load (and the Daytona control-plane load) is divided,
    not replicated.
  * The clean seam is ``HarborTrajectoryRunner.run(TrajectoryRequestBatch)
    -> TrajectoryBatch`` — already an awaited, serializable-in/serializable-out
    boundary. Because ``run()`` itself runs ``submit_batch`` + ``gather``
    + ALL post-gather token/logprob/reward shaping (see
    ``harbor/runner.py`` ``_run()`` body), wrapping ``run()``
    in ``run_shard`` moves *all* of that work off the dispatcher loop and into
    the actor. (This is the CRITICAL move the design calls out — the
    post-gather processing must not survive on the single dispatcher core.)
  * Inference is already a shared HTTP service on its own thread; actors only
    need the host:port string (carried in ``trajectory_runner_cfg.http_endpoint_*``),
    so weights propagate "for free" via the existing broadcast — actors never
    touch weights.

The ``RolloutDispatcher`` is a thin, trajectory-runner-compatible object
(NOT a Ray actor) that the trainer holds in place of ``self.trajectory_runner`` when
fan-out is enabled. It owns NO staleness state (that stays single-loop in
``FullyAsyncRayPPOTrainer`` — same code class that caused prior all_reduce
key-mismatch NCCL deadlocks; must not be distributed). It round-robins each
group-sized ``run()`` call to ONE coordinator (a group is the atomic
reward-shaping unit, so it is never split across actors) and ``ray.get``s the
compact ``TrajectoryBatch`` back.

Default OFF: when ``rollout.fanout.enabled`` is false, the trainer never
constructs any of this and the code path is byte-for-byte the current
behavior. ``enabled: true, num_coordinators: 1`` is behavior-identical modulo
one RPC hop (the K=1 parity check).
"""

from __future__ import annotations

import asyncio
import itertools
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Protocol

import ray
from omegaconf import DictConfig, OmegaConf

from skyrl_train.trajectory_runners.base import TrajectoryRequestBatch, TrajectoryBatch
from skyrl_train.utils.algorithm_registry import rollout_logprobs_enabled
from skyrl_train.utils.harbor_errors import AGENT_TIMEOUT_ERROR
from skyrl_train.utils.fd_monitor import start_fd_monitor
from skyrl_train.worker_setup import configure_worker_process


OUTER_AGENT_TIMEOUT_METRIC = "generate/outer_agent_timeouts"


class _RetryConfig(Protocol):
    max_retries: int
    include_exceptions: set[str] | None
    exclude_exceptions: set[str] | None
    min_wait_sec: float
    max_wait_sec: float
    wait_multiplier: float


class _ShardRunner(Protocol):
    async def run(self, input_batch: TrajectoryRequestBatch, disable_tqdm: bool = False) -> TrajectoryBatch: ...

    async def agent_timeout_output(self, input_batch: TrajectoryRequestBatch) -> TrajectoryBatch: ...


def _retry_backoff_seconds(retry_config: _RetryConfig, attempt: int) -> float:
    return min(
        retry_config.min_wait_sec * retry_config.wait_multiplier**attempt,
        retry_config.max_wait_sec,
    )


@dataclass(frozen=True)
class ShardTimeoutPolicy:
    """Apply the outer shard deadline using Harbor's AgentTimeoutError policy."""

    timeout_seconds: float
    retry_config: _RetryConfig

    @classmethod
    def from_config(
        cls,
        *,
        configured_timeout: float | None,
        agent_timeout: float,
        retry_config: _RetryConfig,
    ) -> "ShardTimeoutPolicy":
        if agent_timeout <= 0:
            raise ValueError("Harbor agent timeout must be greater than zero")
        if configured_timeout is not None:
            if configured_timeout <= 0:
                raise ValueError("rollout.fanout.shard_timeout_seconds must be greater than zero")
            return cls(timeout_seconds=float(configured_timeout), retry_config=retry_config)

        backoff_budget = sum(
            _retry_backoff_seconds(retry_config, attempt) for attempt in range(retry_config.max_retries)
        )
        attempt_budget = agent_timeout * (retry_config.max_retries + 1)
        return cls(timeout_seconds=attempt_budget + backoff_budget, retry_config=retry_config)

    def _should_retry_agent_timeout(self) -> bool:
        excluded = self.retry_config.exclude_exceptions
        if excluded and AGENT_TIMEOUT_ERROR in excluded:
            return False
        included = self.retry_config.include_exceptions
        return not included or AGENT_TIMEOUT_ERROR in included

    async def run(self, runner: _ShardRunner, input_batch: TrajectoryRequestBatch) -> TrajectoryBatch:
        """Generate one shard, converting the outer deadline to AgentTimeoutError semantics."""
        timeout_count = 0
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                output = await asyncio.wait_for(
                    runner.run(input_batch),
                    timeout=self.timeout_seconds,
                )
            except asyncio.TimeoutError:
                timeout_count += 1
                if attempt < self.retry_config.max_retries and self._should_retry_agent_timeout():
                    delay = _retry_backoff_seconds(self.retry_config, attempt)
                    _log().warning(
                        f"Shard exceeded its {self.timeout_seconds:g}s outer deadline; "
                        f"refiling as {AGENT_TIMEOUT_ERROR} and retrying after {delay:g}s "
                        f"({attempt + 1}/{self.retry_config.max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                _log().warning(
                    f"Shard exceeded its {self.timeout_seconds:g}s outer deadline; "
                    f"returning a terminal {AGENT_TIMEOUT_ERROR} after {timeout_count} timeout(s)"
                )
                output = await runner.agent_timeout_output(input_batch)

            metrics = output.setdefault("rollout_metrics", {})
            metrics[OUTER_AGENT_TIMEOUT_METRIC] = metrics.get(OUTER_AGENT_TIMEOUT_METRIC, 0) + timeout_count
            return output

        raise AssertionError("Shard timeout retry loop exhausted without returning an output")


def _log():
    """Lazily fetch the loguru logger INSIDE the calling function.

    CRITICAL (do not refactor back to a module-top ``from loguru import
    logger``): the ``RolloutCoordinator`` class below is a ``@ray.remote`` actor
    that Ray exports to workers via ``export_actor_class``, which cloudpickles
    the class *by value* (its module ``skyrl_train.trajectory_runners.harbor.rollout_dispatcher``
    is not importable on the workers). Cloudpickle's by-value class export walks
    every method's ``__globals__`` for the names the bytecode references
    (``co_names``) and pickles those objects too. Under the forced ``spawn`` start
    method (``main_base.py``), ``skyrl_train.utils.utils.configure_ray_worker_logging``
    has already called ``logger.add(sys.stderr, enqueue=True, ...)`` in this
    process, so the loguru singleton's handler holds a live
    ``multiprocessing.SimpleQueue``. If any method referenced a module-global
    ``logger``, cloudpickle would try to pickle that singleton -> its
    ``SimpleQueue`` -> ``assert_spawning`` -> ``RuntimeError: SimpleQueue objects
    should only be shared between processes through inheritance`` (the crash this
    fix targets). By importing inside the function, ``logger`` is a *local*, not a
    captured module-global, so it is never walked during class export. The actual
    log records are emitted at runtime inside the actor process, where the
    per-process loguru singleton is perfectly usable.
    """
    from loguru import logger

    return logger


def _scale_terminal_bench_cfg(terminal_bench_cfg: DictConfig, num_coordinators: int) -> DictConfig:
    """Return a deep copy of the terminal_bench config scaled for one coordinator.

    Divides the two per-process knobs the design identifies by K:
      * ``harbor.n_concurrent_trials`` — the QueueOrchestrator/TrialQueue
        semaphore size (and therefore concurrent Daytona sandboxes + in-flight
        LLM calls) per process.
      * ``environment.kwargs.connection_pool_maxsize`` (if present) — the
        Daytona httpx pool, which is first-config-wins per process. K processes
        each sized for the full N would be K× load on the Daytona control
        plane; divide by K to keep aggregate load flat.

    Other litellm caches / reap / FD-monitor are naturally per-process and need
    no rescaling. We never scale BELOW 1.
    """
    if num_coordinators <= 1:
        # K=1 parity: hand back an exact copy (no scaling) so the single
        # coordinator is behavior-identical to the non-fanout runner.
        return OmegaConf.create(OmegaConf.to_container(terminal_bench_cfg, resolve=False))

    scaled = OmegaConf.create(OmegaConf.to_container(terminal_bench_cfg, resolve=False))

    # n_concurrent_trials lives under harbor.* (see harbor_config schema).
    harbor = scaled.get("harbor", None)
    if harbor is not None and "n_concurrent_trials" in harbor:
        full = int(harbor["n_concurrent_trials"])
        per_actor = max(1, full // num_coordinators)
        harbor["n_concurrent_trials"] = per_actor
        _log().info(f"[RolloutCoordinator] scaled n_concurrent_trials {full} -> {per_actor} (// {num_coordinators})")

    # connection_pool_maxsize lives under environment.kwargs.* when configured.
    env = scaled.get("environment", None)
    if env is not None:
        env_kwargs = env.get("kwargs", None)
        if env_kwargs is not None and "connection_pool_maxsize" in env_kwargs:
            full_pool = int(env_kwargs["connection_pool_maxsize"])
            per_actor_pool = max(1, full_pool // num_coordinators)
            env_kwargs["connection_pool_maxsize"] = per_actor_pool
            _log().info(
                f"[RolloutCoordinator] scaled connection_pool_maxsize {full_pool} -> "
                f"{per_actor_pool} (// {num_coordinators})"
            )

    return scaled


@ray.remote
class RolloutCoordinator:
    """One rollout-orchestration worker process (own event loop, own Harbor).

    Holds its own ``HarborTrajectoryRunner`` scoped to ``n_concurrent_trials // K``
    and ``connection_pool_maxsize // K``. ``run_shard`` runs the full
    ``run()`` — submit/gather/post-process — locally, returning only the
    compact ``TrajectoryBatch`` over Ray.

    NOTE: the actor is created with ``num_cpus`` set at ``.options(...)`` time by
    the dispatcher (so the PlacementGroup bundle sizing is explicit and visible
    at the call site), not hard-coded here.
    """

    def __init__(
        self,
        cfg: DictConfig,
        trajectory_runner_cfg: DictConfig,
        terminal_bench_cfg: DictConfig,
        shard_idx: int,
        num_coordinators: int,
    ):
        configure_worker_process()
        from skyrl_train.trajectory_runners.harbor.runner import (
            HarborTrajectoryRunner,
        )
        from transformers import AutoTokenizer

        # Each actor process gets its own FD monitor (per-process daemon thread),
        # mirroring the entrypoint behavior.
        try:
            start_fd_monitor()
        except Exception as e:  # pragma: no cover - best-effort
            _log().warning(f"[RolloutCoordinator {shard_idx}] start_fd_monitor failed: {e}")

        self._shard_idx = shard_idx
        self._num_coordinators = num_coordinators

        scaled_tb_cfg = _scale_terminal_bench_cfg(terminal_bench_cfg, num_coordinators)

        # Build the tokenizer in-process (same construction as
        # BasePPOExp.get_tokenizer) — the runner uses it during
        # post-gather token/logprob extraction (apply_chat_template).
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.trainer.policy.model.path,
            trust_remote_code=True,
            use_fast=not cfg.trainer.disable_fast_tokenizer,
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # The runner never actually dereferences inference_engine_client — it
        # talks to vLLM over HTTP via trajectory_runner_cfg.http_endpoint_{host,port}.
        # So None is safe and avoids shipping a Ray actor handle into the worker.
        # NOTE: this is verified against the current HarborTrajectoryRunner,
        # which only stores the handle and never calls it.
        self._runner = HarborTrajectoryRunner(
            trajectory_runner_cfg=trajectory_runner_cfg,
            terminal_bench_cfg=scaled_tb_cfg,
            inference_engine_client=None,
            tokenizer=tokenizer,
            moe_router_replay=bool(cfg.trainer.policy.fsdp_config.get("moe_router_replay", False)),
            rollout_logprobs_required=rollout_logprobs_enabled(cfg.trainer.algorithm),
            tito_full=cfg.trainer.algorithm.get("tito_full", None),
        )

        configured_timeout = cfg.get("rollout", {}).get("fanout", {}).get("shard_timeout_seconds", None)
        self._shard_timeout_policy = ShardTimeoutPolicy.from_config(
            configured_timeout=configured_timeout,
            agent_timeout=self._runner.agent_timeout_seconds,
            retry_config=self._runner.retry_config,
        )

        _log().info(
            f"[RolloutCoordinator {shard_idx}/{num_coordinators}] constructed "
            f"(http={trajectory_runner_cfg.http_endpoint_host}:{trajectory_runner_cfg.http_endpoint_port}, "
            f"shard_timeout={self._shard_timeout_policy.timeout_seconds:g}s)"
        )

    async def startup(self) -> None:
        """Create the coordinator's QueueOrchestrator (mirrors runner.startup)."""
        # Widen THIS actor-loop's default ThreadPoolExecutor. litellm.acompletion runs
        # its whole SYNCHRONOUS preamble (param validation, get_optional_params, provider
        # config, header/body build) via loop.run_in_executor(None, ...) — the loop's
        # default ~min(32, cpu+4)-wide pool — before awaiting the async httpx POST. With
        # ~n_concurrent_trials/num_coordinators trials multiplexed on this one loop, that
        # 32-wide pool serializes the completion preambles and starves the vLLM engines
        # (bursty saturation, ~1/3 TDP — v0i round-3 py-spy). Widen it so all in-flight
        # completions dispatch concurrently. Tunable via SKYRL_COORDINATOR_EXECUTOR_WORKERS
        # (default 256). Best-effort: never fail startup on this.
        try:
            workers = int(os.environ.get("SKYRL_COORDINATOR_EXECUTOR_WORKERS", "256"))
            asyncio.get_running_loop().set_default_executor(
                ThreadPoolExecutor(max_workers=workers, thread_name_prefix="coord-exec")
            )
            _log().info(
                f"[RolloutCoordinator {self._shard_idx}] default executor widened to "
                f"max_workers={workers} (litellm acompletion preamble concurrency)"
            )
        except Exception as e:
            _log().warning(f"[RolloutCoordinator {self._shard_idx}] set_default_executor failed: {e}")
        await self._runner.startup()
        _log().info(f"[RolloutCoordinator {self._shard_idx}] startup complete")

    async def shutdown(self) -> None:
        await self._runner.shutdown()
        _log().info(f"[RolloutCoordinator {self._shard_idx}] shutdown complete")

    async def run_shard(self, sub_batch: TrajectoryRequestBatch, global_step: Optional[int]) -> TrajectoryBatch:
        """Run one group's generation locally and return the TrajectoryBatch.

        ``global_step`` is the dispatcher's current step at submission time. We
        pin the runner's ``global_step_fn`` to return it for the duration of
        the call so the in-actor staleness/step-time bookkeeping
        (``_record_step_time``/``actual_global_step``) behaves exactly as it
        would single-process. The dispatcher remains the authority on staleness
        accounting; this only affects the ``actual_global_step`` hint the actor
        returns in the TrajectoryBatch.
        """
        if global_step is not None:
            self._runner.global_step_fn = lambda: global_step

        return await self._shard_timeout_policy.run(self._runner, sub_batch)

    # ---- Eval session passthrough (single-coordinator delegation) ----
    async def start_eval_session(self, run_name: str, eval_step: int, val_set_name=None) -> None:
        await self._runner.start_eval_session(run_name, eval_step, val_set_name)

    async def stop_eval_session(self) -> None:
        await self._runner.stop_eval_session()


class RolloutDispatcher:
    """Trajectory-runner-compatible proxy that fans out across K coordinators.

    Drop-in for ``self.trajectory_runner`` in the trainer when ``rollout.fanout.enabled``.
    Owns NO staleness state. Each ``run()`` call (one group =
    n_samples_per_prompt trajectories) is routed round-robin to ONE coordinator;
    a group is never split (it is the atomic reward-shaping unit). With
    ``num_parallel_generation_workers`` concurrent ``run()`` calls in flight,
    the load spreads naturally across the K coordinators' event loops.

    Lifecycle mirrors ``TrajectoryRunner``: ``startup`` / ``run`` /
    ``shutdown`` (+ optional eval-session passthrough). ``global_step_fn`` is set
    by the trainer; we forward its current value into each ``run_shard`` so the
    actor's staleness hint is accurate.
    """

    def __init__(
        self,
        cfg: DictConfig,
        trajectory_runner_cfg: DictConfig,
        terminal_bench_cfg: DictConfig,
        num_coordinators: int,
        cpus_per_coordinator: int,
    ):
        # Detach each config to a parent-ref-free, object-free OmegaConf copy
        # BEFORE it can cross a `.remote()` boundary. The live `cfg` tree (under
        # the forced `spawn` start method) transitively reaches a wandb-class
        # `multiprocessing.SimpleQueue`, which cannot be pickled into a Ray actor
        # ("SimpleQueue objects should only be shared between processes through
        # inheritance"). The `to_container(resolve=True)` round-trip severs
        # OmegaConf parent back-references and drops any attached live object;
        # it changes only HOW configs are shipped, not their values. Mirrors the
        # pattern already used by `_scale_terminal_bench_cfg` in this file.
        self.cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        self._trajectory_runner_cfg = OmegaConf.create(OmegaConf.to_container(trajectory_runner_cfg, resolve=True))
        self._terminal_bench_cfg = OmegaConf.create(OmegaConf.to_container(terminal_bench_cfg, resolve=True))

        # --- Fan-out connectivity fix (head-IP injection) ---
        # The vLLM HTTP inference endpoint (InferenceEngineClient) is bound on the
        # HEAD node — the same process that constructs this dispatcher. Its
        # configured `http_endpoint_host` is 127.0.0.1, which only resolves to the
        # endpoint ON the head. The RolloutCoordinator actors below run on WORKER
        # nodes (SPREAD PlacementGroup), where 127.0.0.1:8000 has nothing
        # listening -> every litellm request fails "All connection attempts
        # failed". This dispatcher runs on the head where the endpoint is bound, so
        # `ray.util.get_node_ip_address()` here yields the head's ROUTABLE compute
        # IP. We substitute it for the loopback host in the per-coordinator
        # runner config so each coordinator builds its litellm base_url against
        # a reachable address. The server is bound to 0.0.0.0 (see
        # InferenceEngineClient._spin_up_http_endpoint), so this routable host is
        # reachable from every node.
        #
        # GATING: this only happens on the fan-out path — the RolloutDispatcher is
        # constructed ONLY when rollout.fanout.enabled (see
        # fully_async_trainer._maybe_enable_rollout_fanout). When fan-out is OFF,
        # the dispatcher never exists and the runner runs in-process on the head
        # using the unchanged 127.0.0.1 host. We only override the loopback host so
        # an explicitly-configured non-loopback host (e.g. a manual remote setup)
        # is respected.
        configured_host = self._trajectory_runner_cfg.get("http_endpoint_host", None)
        if configured_host in ("127.0.0.1", "localhost", None):
            head_ip = ray.util.get_node_ip_address()
            self._trajectory_runner_cfg["http_endpoint_host"] = head_ip
            _log().info(
                f"[RolloutDispatcher] fan-out path: overriding inference host "
                f"{configured_host} -> {head_ip} (routable head IP) for "
                f"coordinator litellm base_url connectivity"
            )
        self._num_coordinators = num_coordinators
        self._cpus_per_coordinator = cpus_per_coordinator

        # Trainer sets this; default returns None until then.
        self.global_step_fn = None

        self._actors: List = []
        self._rr = itertools.cycle(range(num_coordinators))
        self._pg = None
        # When an eval session is active, run() is pinned to shard 0 (the
        # only coordinator with the eval orchestrator). See start_eval_session.
        self._eval_session_active = False

        _log().info(
            f"[RolloutDispatcher] configured num_coordinators={num_coordinators}, "
            f"cpus_per_coordinator={cpus_per_coordinator}"
        )

    def _current_global_step(self) -> Optional[int]:
        if self.global_step_fn is None:
            return None
        try:
            return self.global_step_fn()
        except Exception:
            return None

    async def startup(self) -> None:
        """Create the K coordinators (pinned to the proxy's node) and start each runner.

        All coordinators are pinned via NodeAffinity to THIS (rank-0/head) node — the
        node where the RecordProxy writes the node-local opencode literal log that
        ``LiteralLogStore`` reads with a local ``open()``. A SPREAD placement would scatter
        them and break that read on every off-node coordinator (keep1-v25). Each actor
        requests ``cpus_per_coordinator`` CPUs on the head node.
        """
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        # The RecordProxy writes the opencode literal log to a NODE-LOCAL path on THIS
        # (rank-0/head) node, and LiteralLogStore reads it with a bare local open(). A
        # SPREAD placement group scattered the K coordinators across nodes, so ~(K-1)/K of
        # them could not open the log -> _maybe_build_opencode_chat_history returned None ->
        # 100% 'all_messages' drops -> empty training batch (keep1-v25; v24 only worked
        # because its lone reader happened to co-locate with the proxy). Pin every
        # coordinator to the proxy's node so the local read always resolves. The K-pool's
        # parallelism is per-PROCESS (K GILs) and is preserved: K actor processes on one
        # node still run K independent event loops. (Placement-independent alternative for
        # later: read the proxy's periodic remote mirror via upath -- literal_proxy_utils.)
        self._proxy_node_id = ray.get_runtime_context().get_node_id()
        _log().info(
            f"[RolloutDispatcher] pinning {self._num_coordinators} coordinators to the "
            f"proxy/head node {self._proxy_node_id} for the node-local literal log "
            f"({self._cpus_per_coordinator} CPU each)"
        )

        # SEQUENTIAL bring-up: create one coordinator, await its startup to
        # completion, THEN create + start the next. Each coordinator imports the
        # full Harbor/transformers stack and loads a tokenizer off the shared FS
        # (GPFS) at startup; doing all K concurrently produced a thundering-herd
        # page-in burst that — coincident with the vLLM engines loading weights —
        # tipped GPFS into a SIGBUS / errno=116 (ESTALE) mmap fault that killed
        # raylets and cascaded to ActorUnavailableError at weight-sync-state
        # init. Serializing construction+startup keeps only one coordinator
        # paging the heavy stack in at a time. The SPREAD PlacementGroup is
        # retained (it is not the cause).
        self._actors = []
        for shard_idx in range(self._num_coordinators):
            actor = RolloutCoordinator.options(
                num_cpus=self._cpus_per_coordinator,
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=self._proxy_node_id, soft=False),
            ).remote(
                cfg=self.cfg,
                trajectory_runner_cfg=self._trajectory_runner_cfg,
                terminal_bench_cfg=self._terminal_bench_cfg,
                shard_idx=shard_idx,
                num_coordinators=self._num_coordinators,
            )
            # Await THIS coordinator's startup/readiness to completion before
            # constructing the next one, so its heavy GPFS import + tokenizer
            # load finishes (and pages settle) before the next begins.
            await actor.startup.remote()
            self._actors.append(actor)
            _log().info(f"[RolloutDispatcher] coordinator {shard_idx + 1}/{self._num_coordinators} started")
            # Spread the page-in further: brief pause between coordinators.
            if shard_idx + 1 < self._num_coordinators:
                await asyncio.sleep(2)

        _log().info(f"[RolloutDispatcher] {self._num_coordinators} coordinators started")

    async def run(self, input_batch: TrajectoryRequestBatch, disable_tqdm: bool = False) -> TrajectoryBatch:
        """Route one group to one coordinator and await its TrajectoryBatch.

        Training: round-robin across all coordinators. Eval: pinned to shard 0
        (the only coordinator with an active eval orchestrator).
        """
        del disable_tqdm
        if self._eval_session_active:
            actor = self._actors[0]
        else:
            actor = self._actors[next(self._rr)]
        global_step = self._current_global_step()
        return await actor.run_shard.remote(input_batch, global_step)

    async def shutdown(self) -> None:
        if self._actors:
            try:
                await asyncio.gather(*[a.shutdown.remote() for a in self._actors], return_exceptions=True)
            except Exception as e:  # pragma: no cover - best-effort
                _log().warning(f"[RolloutDispatcher] coordinator shutdown error: {e}")
        if self._pg is not None:
            try:
                from ray.util.placement_group import remove_placement_group

                remove_placement_group(self._pg)
            except Exception as e:  # pragma: no cover - best-effort
                _log().warning(f"[RolloutDispatcher] remove_placement_group error: {e}")
            self._pg = None
        self._actors = []

    # ---- Eval session passthrough ----
    # Eval routes through a SINGLE coordinator (shard 0) to keep eval-session
    # orchestrator lifecycle simple and correct. Eval is gated off in production
    # (eval_interval is effectively infinite), so this path is rarely exercised
    # under fan-out; routing to one coordinator avoids fanning eval-session
    # state across K orchestrators.
    async def start_eval_session(self, run_name: str, eval_step: int, val_set_name=None) -> None:
        if self._actors:
            await self._actors[0].start_eval_session.remote(run_name, eval_step, val_set_name)
            self._eval_session_active = True

    async def stop_eval_session(self) -> None:
        if self._actors:
            await self._actors[0].stop_eval_session.remote()
            self._eval_session_active = False
