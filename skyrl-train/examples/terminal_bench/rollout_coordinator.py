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
actors between the trainer and the generator:

  * Each actor builds its OWN ``TerminalBenchGenerator`` scoped to the process
    with ``n_concurrent_trials // K`` and ``daytona connection_pool_maxsize // K``
    so the per-process load (and the Daytona control-plane load) is divided,
    not replicated.
  * The clean seam is ``TerminalBenchGenerator.generate(GeneratorInput)
    -> GeneratorOutput`` — already an awaited, serializable-in/serializable-out
    boundary. Because ``generate()`` itself runs ``submit_batch`` + ``gather``
    + ALL post-gather token/logprob/reward shaping (see
    ``terminal_bench_generator.py`` ``generate()`` body), wrapping ``generate()``
    in ``run_shard`` moves *all* of that work off the dispatcher loop and into
    the actor. (This is the CRITICAL move the design calls out — the
    post-gather processing must not survive on the single dispatcher core.)
  * Inference is already a shared HTTP service on its own thread; actors only
    need the host:port string (carried in ``generator_cfg.http_endpoint_*``),
    so weights propagate "for free" via the existing broadcast — actors never
    touch weights.

The ``RolloutDispatcher`` is a thin, generator-interface-compatible object
(NOT a Ray actor) that the trainer holds in place of ``self.generator`` when
fan-out is enabled. It owns NO staleness state (that stays single-loop in
``FullyAsyncRayPPOTrainer`` — same code class that caused prior all_reduce
key-mismatch NCCL deadlocks; must not be distributed). It round-robins each
group-sized ``generate()`` call to ONE coordinator (a group is the atomic
reward-shaping unit, so it is never split across actors) and ``ray.get``s the
compact ``GeneratorOutput`` back.

Default OFF: when ``rollout.fanout.enabled`` is false, the trainer never
constructs any of this and the code path is byte-for-byte the current
behavior. ``enabled: true, num_coordinators: 1`` is behavior-identical modulo
one RPC hop (the K=1 parity check).
"""

from __future__ import annotations

import asyncio
import itertools
from typing import List, Optional

import ray
from omegaconf import DictConfig, OmegaConf

from skyrl_train.generators.base import GeneratorInput, GeneratorOutput


def _log():
    """Lazily fetch the loguru logger INSIDE the calling function.

    CRITICAL (do not refactor back to a module-top ``from loguru import
    logger``): the ``RolloutCoordinator`` class below is a ``@ray.remote`` actor
    that Ray exports to workers via ``export_actor_class``, which cloudpickles
    the class *by value* (its module ``examples.terminal_bench.rollout_coordinator``
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


def _scale_terminal_bench_cfg(
    terminal_bench_cfg: DictConfig, num_coordinators: int
) -> DictConfig:
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
        # coordinator is behavior-identical to the non-fanout generator.
        return OmegaConf.create(
            OmegaConf.to_container(terminal_bench_cfg, resolve=False)
        )

    scaled = OmegaConf.create(OmegaConf.to_container(terminal_bench_cfg, resolve=False))

    # n_concurrent_trials lives under harbor.* (see harbor_config schema).
    harbor = scaled.get("harbor", None)
    if harbor is not None and "n_concurrent_trials" in harbor:
        full = int(harbor["n_concurrent_trials"])
        per_actor = max(1, full // num_coordinators)
        harbor["n_concurrent_trials"] = per_actor
        _log().info(
            f"[RolloutCoordinator] scaled n_concurrent_trials {full} -> {per_actor} "
            f"(// {num_coordinators})"
        )

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

    Holds its own ``TerminalBenchGenerator`` scoped to ``n_concurrent_trials // K``
    and ``connection_pool_maxsize // K``. ``run_shard`` runs the full
    ``generate()`` — submit/gather/post-process — locally, returning only the
    compact ``GeneratorOutput`` over Ray.

    NOTE: the actor is created with ``num_cpus`` set at ``.options(...)`` time by
    the dispatcher (so the PlacementGroup bundle sizing is explicit and visible
    at the call site), not hard-coded here.
    """

    def __init__(
        self,
        cfg: DictConfig,
        generator_cfg: DictConfig,
        terminal_bench_cfg: DictConfig,
        shard_idx: int,
        num_coordinators: int,
    ):
        # Import here so the heavy Harbor/terminal-bench import only happens in
        # the actor process, not on the dispatcher when fan-out is off.
        from examples.terminal_bench.terminal_bench_generator import (
            TerminalBenchGenerator,
        )
        from examples.terminal_bench.fd_monitor import start_fd_monitor
        from transformers import AutoTokenizer

        # Each actor process gets its own FD monitor (per-process daemon thread),
        # mirroring the entrypoint behavior.
        try:
            start_fd_monitor()
        except Exception as e:  # pragma: no cover - best-effort
            _log().warning(
                f"[RolloutCoordinator {shard_idx}] start_fd_monitor failed: {e}"
            )

        self._shard_idx = shard_idx
        self._num_coordinators = num_coordinators

        scaled_tb_cfg = _scale_terminal_bench_cfg(terminal_bench_cfg, num_coordinators)

        # Build the tokenizer in-process (same construction as
        # BasePPOExp.get_tokenizer) — the generator uses it during
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

        # The generator never actually dereferences inference_engine_client — it
        # talks to vLLM over HTTP via generator_cfg.http_endpoint_{host,port}.
        # So None is safe and avoids shipping a Ray actor handle into the worker.
        # NOTE: this is verified against the current TerminalBenchGenerator,
        # which only stores the handle and never calls it.
        self._generator = TerminalBenchGenerator(
            generator_cfg=generator_cfg,
            terminal_bench_cfg=scaled_tb_cfg,
            inference_engine_client=None,
            tokenizer=tokenizer,
        )

        # Pause gate. When set (paused), run_shard refuses to admit new shards.
        # We use an asyncio.Event toggled true=running / false=paused.
        self._running_event = asyncio.Event()
        self._running_event.set()
        # Count of shards currently executing inside this actor (for drain).
        self._inflight = 0
        self._inflight_zero = asyncio.Event()
        self._inflight_zero.set()

        _log().info(
            f"[RolloutCoordinator {shard_idx}/{num_coordinators}] constructed "
            f"(http={generator_cfg.http_endpoint_host}:{generator_cfg.http_endpoint_port})"
        )

    async def startup(self) -> None:
        """Create the coordinator's QueueOrchestrator (mirrors generator.startup)."""
        await self._generator.startup()
        _log().info(f"[RolloutCoordinator {self._shard_idx}] startup complete")

    async def shutdown(self) -> None:
        await self._generator.shutdown()
        _log().info(f"[RolloutCoordinator {self._shard_idx}] shutdown complete")

    async def run_shard(
        self, sub_batch: GeneratorInput, global_step: Optional[int]
    ) -> GeneratorOutput:
        """Run one group's generation locally and return the GeneratorOutput.

        ``global_step`` is the dispatcher's current step at submission time. We
        pin the generator's ``global_step_fn`` to return it for the duration of
        the call so the in-actor staleness/step-time bookkeeping
        (``_record_step_time``/``actual_global_step``) behaves exactly as it
        would single-process. The dispatcher remains the authority on staleness
        accounting; this only affects the ``actual_global_step`` hint the actor
        returns in the GeneratorOutput.
        """
        # Block until resumed if we're paused (weight-sync quiescing).
        await self._running_event.wait()

        if global_step is not None:
            self._generator.global_step_fn = lambda: global_step

        self._inflight += 1
        self._inflight_zero.clear()
        try:
            return await self._generator.generate(sub_batch)
        finally:
            self._inflight -= 1
            if self._inflight == 0:
                self._inflight_zero.set()

    async def pause(self) -> None:
        """Stop admitting new shards and wait for in-flight ones to drain.

        Weight-sync quiescing: the dispatcher barriers on
        ``ray.get([a.pause.remote() for a in actors])`` before broadcasting
        fresh weights. ``run_shard`` awaits ``_running_event`` at the top, so
        once we clear it no new generate() begins; we then wait for currently
        executing shards to finish so no rollout straddles the weight swap.

        Because the dispatcher's staleness manager already bounds the number of
        groups in flight (``num_parallel_generation_workers``), the in-flight
        set here is small and drains quickly. This is a bounded-drain, matching
        the single-loop ``pause_generation`` semantics.
        """
        self._running_event.clear()
        await self._inflight_zero.wait()
        _log().debug(f"[RolloutCoordinator {self._shard_idx}] paused (drained)")

    async def resume(self) -> None:
        """Resume admitting new shards (symmetric to pause)."""
        self._running_event.set()
        _log().debug(f"[RolloutCoordinator {self._shard_idx}] resumed")

    # ---- Eval session passthrough (single-coordinator delegation) ----
    async def start_eval_session(
        self, run_name: str, eval_step: int, val_set_name=None
    ) -> None:
        if hasattr(self._generator, "start_eval_session"):
            await self._generator.start_eval_session(run_name, eval_step, val_set_name)

    async def stop_eval_session(self) -> None:
        if hasattr(self._generator, "stop_eval_session"):
            await self._generator.stop_eval_session()


class RolloutDispatcher:
    """Generator-interface-compatible proxy that fans out across K coordinators.

    Drop-in for ``self.generator`` in the trainer when ``rollout.fanout.enabled``.
    Owns NO staleness state. Each ``generate()`` call (one group =
    n_samples_per_prompt trajectories) is routed round-robin to ONE coordinator;
    a group is never split (it is the atomic reward-shaping unit). With
    ``num_parallel_generation_workers`` concurrent ``generate()`` calls in flight,
    the load spreads naturally across the K coordinators' event loops.

    Lifecycle mirrors ``GeneratorInterface``: ``startup`` / ``generate`` /
    ``shutdown`` (+ optional eval-session passthrough). ``global_step_fn`` is set
    by the trainer; we forward its current value into each ``run_shard`` so the
    actor's staleness hint is accurate.
    """

    def __init__(
        self,
        cfg: DictConfig,
        generator_cfg: DictConfig,
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
        self._generator_cfg = OmegaConf.create(
            OmegaConf.to_container(generator_cfg, resolve=True)
        )
        self._terminal_bench_cfg = OmegaConf.create(
            OmegaConf.to_container(terminal_bench_cfg, resolve=True)
        )

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
        # generator config so each coordinator builds its litellm base_url against
        # a reachable address. The server is bound to 0.0.0.0 (see
        # InferenceEngineClient._spin_up_http_endpoint), so this routable host is
        # reachable from every node.
        #
        # GATING: this only happens on the fan-out path — the RolloutDispatcher is
        # constructed ONLY when rollout.fanout.enabled (see
        # fully_async_trainer._maybe_enable_rollout_fanout). When fan-out is OFF,
        # the dispatcher never exists and the generator runs in-process on the head
        # using the unchanged 127.0.0.1 host. We only override the loopback host so
        # an explicitly-configured non-loopback host (e.g. a manual remote setup)
        # is respected.
        configured_host = self._generator_cfg.get("http_endpoint_host", None)
        if configured_host in ("127.0.0.1", "localhost", None):
            head_ip = ray.util.get_node_ip_address()
            self._generator_cfg["http_endpoint_host"] = head_ip
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
        # When an eval session is active, generate() is pinned to shard 0 (the
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
        """Create the PlacementGroup + K coordinators and start each generator.

        Uses a SPREAD PlacementGroup so coordinators land on idle CPUs across
        all allocation nodes (engine nodes have idle cores too). Each bundle
        requests ``cpus_per_coordinator`` CPUs.
        """
        from ray.util.placement_group import placement_group
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        bundles = [
            {"CPU": float(self._cpus_per_coordinator)}
            for _ in range(self._num_coordinators)
        ]
        self._pg = placement_group(bundles, strategy="SPREAD")
        await self._pg.ready()
        _log().info(
            f"[RolloutDispatcher] PlacementGroup ready: {self._num_coordinators} bundles "
            f"x {self._cpus_per_coordinator} CPU (SPREAD)"
        )

        self._actors = []
        for shard_idx in range(self._num_coordinators):
            actor = RolloutCoordinator.options(
                num_cpus=self._cpus_per_coordinator,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=self._pg,
                    placement_group_bundle_index=shard_idx,
                ),
            ).remote(
                cfg=self.cfg,
                generator_cfg=self._generator_cfg,
                terminal_bench_cfg=self._terminal_bench_cfg,
                shard_idx=shard_idx,
                num_coordinators=self._num_coordinators,
            )
            self._actors.append(actor)

        # Barrier on every coordinator's startup so the orchestrators exist
        # before the first generate() RPC.
        await asyncio.gather(*[a.startup.remote() for a in self._actors])
        _log().info(
            f"[RolloutDispatcher] {self._num_coordinators} coordinators started"
        )

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """Route one group to one coordinator and await its GeneratorOutput.

        Training: round-robin across all coordinators. Eval: pinned to shard 0
        (the only coordinator with an active eval orchestrator).
        """
        if self._eval_session_active:
            actor = self._actors[0]
        else:
            actor = self._actors[next(self._rr)]
        global_step = self._current_global_step()
        return await actor.run_shard.remote(input_batch, global_step)

    async def pause(self) -> None:
        """Barrier-pause every coordinator (weight-sync quiescing)."""
        await asyncio.gather(*[a.pause.remote() for a in self._actors])

    async def resume(self) -> None:
        """Barrier-resume every coordinator."""
        await asyncio.gather(*[a.resume.remote() for a in self._actors])

    async def shutdown(self) -> None:
        if self._actors:
            try:
                await asyncio.gather(
                    *[a.shutdown.remote() for a in self._actors], return_exceptions=True
                )
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
    async def start_eval_session(
        self, run_name: str, eval_step: int, val_set_name=None
    ) -> None:
        if self._actors:
            await self._actors[0].start_eval_session.remote(
                run_name, eval_step, val_set_name
            )
            self._eval_session_active = True

    async def stop_eval_session(self) -> None:
        if self._actors:
            await self._actors[0].stop_eval_session.remote()
            self._eval_session_active = False
