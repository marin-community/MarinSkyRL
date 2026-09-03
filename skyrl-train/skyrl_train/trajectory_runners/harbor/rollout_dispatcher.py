"""Process-isolated Harbor trajectory execution.

Production Harbor workloads use this runner regardless of trainer type. A request can
contain arbitrary reward groups. The dispatcher keeps each group on one coordinator,
runs groups concurrently, and restores the input row order before returning.
"""

from __future__ import annotations

import asyncio
import itertools
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from typing import Any, List, Optional

import ray
from omegaconf import DictConfig, OmegaConf

from skyrl_train.timing_observability import RolloutTimings
from skyrl_train.trajectory_runners.base import TrajectoryID, TrajectoryRequestBatch, TrajectoryBatch
from skyrl_train.trajectory_runners.harbor.execution import HarborRunnerSpec, ProcessPoolResources
from skyrl_train.trajectory_runners.trajectory_processing import concatenate_trajectory_batches
from skyrl_train.trajectory_runners.trajectory_retention import TrajectorySink, retain_trajectories
from skyrl_train.utils.algorithm_registry import rollout_logprobs_enabled
from skyrl_train.utils.fd_monitor import start_fd_monitor
from skyrl_train.worker_setup import configure_worker_process

# A literal because the harbor package does not import off Linux and this runs in the driver.
# Nothing catches a rename of the class: fan-out would fail at startup on `bind_runner`.
RETAINED_RUNNER_NAME = "HarborTrajectoryRunner"


class RolloutCoordinatorRPCTimeoutError(TimeoutError):
    """The rollout coordinator did not return before its RPC watchdog expired."""


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
        spec: HarborRunnerSpec,
        shard_idx: int,
        num_coordinators: int,
        executor_workers: int,
    ):
        configure_worker_process()
        from transformers import AutoTokenizer

        # Each actor process gets its own FD monitor (per-process daemon thread),
        # mirroring the entrypoint behavior.
        try:
            start_fd_monitor()
        except Exception as e:  # pragma: no cover - best-effort
            _log().warning(f"[RolloutCoordinator {shard_idx}] start_fd_monitor failed: {e}")

        self._shard_idx = shard_idx
        self._num_coordinators = num_coordinators
        self._executor_workers = executor_workers

        scaled_tb_cfg = _scale_terminal_bench_cfg(spec.terminal_bench_config, num_coordinators)
        spec = spec.with_terminal_bench_config(scaled_tb_cfg)

        # Build the tokenizer in-process (same construction as
        # BasePPOExp.get_tokenizer) — the runner uses it during
        # post-gather token/logprob extraction (apply_chat_template).
        tokenizer = AutoTokenizer.from_pretrained(
            spec.config.trainer.policy.model.path,
            trust_remote_code=True,
            use_fast=not spec.config.trainer.disable_fast_tokenizer,
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        self._runner = spec.build(tokenizer)

        _log().info(
            f"[RolloutCoordinator {shard_idx}/{num_coordinators}] constructed "
            f"(http={spec.runner_config.http_endpoint_host}:{spec.runner_config.http_endpoint_port})"
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
        # completions dispatch concurrently. Best-effort: never fail startup on this.
        try:
            workers = self._executor_workers
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

        return await self._runner.run(sub_batch)

    # ---- Eval session passthrough (single-coordinator delegation) ----
    async def start_eval_session(self, *, run_name: str, eval_step: int, val_set_name: str | None = None) -> None:
        await self._runner.start_eval_session(run_name=run_name, eval_step=eval_step, val_set_name=val_set_name)

    async def stop_eval_session(self) -> None:
        await self._runner.stop_eval_session()


class RolloutDispatcher:
    """Trajectory-runner proxy that runs atomic reward groups across K coordinators.

    One call can contain any number of groups. The dispatcher partitions them by
    instance identity, sends each complete group to one coordinator, and restores
    the request order after concurrent execution. It owns no trainer staleness state.

    Lifecycle mirrors ``TrajectoryRunner``: ``startup`` / ``run`` /
    ``shutdown`` (+ optional eval-session passthrough). ``global_step_fn`` is set
    by the trainer; we forward its current value into each ``run_shard`` so the
    actor's staleness hint is accurate.
    """

    def __init__(
        self,
        spec: HarborRunnerSpec,
        resources: ProcessPoolResources,
    ):
        self._spec = spec

        self._num_coordinators = resources.num_coordinators
        self._cpus_per_coordinator = resources.cpus_per_coordinator
        self._executor_workers = resources.executor_workers
        self._coordinator_rpc_timeout = resources.rpc_timeout_seconds

        # Trainer sets this; default returns None until then.
        self.global_step_fn = None

        # The coordinators' runners never receive it: the sink is trainer-owned, lives in this
        # process, and run() gets the finished batch back here.
        self._trajectory_sink: Optional[TrajectorySink] = None

        self._actors: List = []
        self._rr = itertools.cycle(range(self._num_coordinators))
        self._pg = None
        # When an eval session is active, run() is pinned to shard 0 (the
        # only coordinator with the eval orchestrator). See start_eval_session.
        self._eval_session_active = False

        _log().info(
            f"[RolloutDispatcher] configured num_coordinators={self._num_coordinators}, "
            f"cpus_per_coordinator={self._cpus_per_coordinator}, "
            f"coordinator_rpc_timeout={self._coordinator_rpc_timeout:g}s"
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

        runner_config = self._spec.runner_config.copy()
        configured_host = runner_config.get("http_endpoint_host", None)
        if configured_host in ("127.0.0.1", "localhost", None):
            runner_config.http_endpoint_host = ray.util.get_node_ip_address()
        actor_spec = self._spec.with_runner_config(runner_config)

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
        # paging the heavy stack in at a time.
        self._actors = []
        for shard_idx in range(self._num_coordinators):
            actor = RolloutCoordinator.options(
                num_cpus=self._cpus_per_coordinator,
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=self._proxy_node_id, soft=False),
            ).remote(
                spec=actor_spec,
                shard_idx=shard_idx,
                num_coordinators=self._num_coordinators,
                executor_workers=self._executor_workers,
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

    def set_trajectory_sink(self, sink: TrajectorySink) -> None:
        """Attach the trainer-owned sink used to retain each returned batch.

        Binds it to the Harbor runner rather than to this proxy. ``runner_name`` is
        recorded on each retained trajectory, so process placement must not change provenance.
        """
        sink.bind_runner(RETAINED_RUNNER_NAME)
        self._trajectory_sink = sink

    async def run(
        self,
        input_batch: TrajectoryRequestBatch,
        disable_tqdm: bool = False,
        *,
        phase_timings: RolloutTimings | None = None,
    ) -> TrajectoryBatch:
        """Run complete reward groups concurrently and restore input row order.

        ``phase_timings`` is accepted and dropped: this path runs K coordinators concurrently, and
        the generate span tree decomposes a single wall. Forwarding it to every shard would sum
        overlapping walls into one dict and produce a residual with no meaning. The dispatcher
        therefore publishes no generate spans at all -- absence, rather than a residual equal to the
        whole parent, which would read as "generate is entirely unaccounted for".
        """
        del disable_tqdm, phase_timings
        trajectory_ids = input_batch.get("trajectory_ids")
        if not trajectory_ids or len(trajectory_ids) != len(input_batch["prompts"]):
            raise ValueError("process-isolated trajectory execution requires one trajectory ID per request row")
        request_keys = [trajectory_id.to_string() for trajectory_id in trajectory_ids]
        if len(request_keys) != len(set(request_keys)):
            raise ValueError("process-isolated trajectory execution requires unique trajectory IDs")

        groups: dict[str, list[int]] = defaultdict(list)
        for index, trajectory_id in enumerate(trajectory_ids):
            groups[trajectory_id.instance_id].append(index)

        sub_batches = [self._select_request_rows(input_batch, indices) for indices in groups.values()]
        outputs = await asyncio.gather(*(self._run_group(sub_batch) for sub_batch in sub_batches))
        for sub_batch, output in zip(sub_batches, outputs, strict=True):
            self._validate_group_identity(sub_batch, output)

        if len(outputs) == 1:
            result = outputs[0]
        else:
            result = concatenate_trajectory_batches(
                outputs,
                require_rollout_logprobs=rollout_logprobs_enabled(self._spec.config.trainer.algorithm),
                tis_lcs_alert_threshold=float(self._spec.config.trainer.algorithm.tis_lcs_alert_threshold),
            )
            actual_steps = [output.get("actual_global_step") for output in outputs]
            observed_steps = [step for step in actual_steps if step is not None]
            if observed_steps:
                result["actual_global_step"] = min(observed_steps)
            self._restore_request_order(result, trajectory_ids)

        # Outside the deadline: a slow sink write is not an unresponsive coordinator.
        if self._trajectory_sink is not None:
            await retain_trajectories(self._trajectory_sink, input_batch, result)
        return result

    async def _run_group(self, input_batch: TrajectoryRequestBatch) -> TrajectoryBatch:
        if self._eval_session_active:
            coordinator_index = 0
            actor = self._actors[0]
        else:
            coordinator_index = next(self._rr)
            actor = self._actors[coordinator_index]
        global_step = self._current_global_step()
        rpc = actor.run_shard.remote(input_batch, global_step)
        rpc_deadline = asyncio.timeout(self._coordinator_rpc_timeout)
        try:
            async with rpc_deadline:
                output = await asyncio.shield(rpc)
        except TimeoutError as error:
            if not rpc_deadline.expired():
                raise
            raise RolloutCoordinatorRPCTimeoutError(
                f"Rollout coordinator {coordinator_index} RPC did not return within "
                f"{self._coordinator_rpc_timeout:g} seconds"
            ) from error
        return output

    @staticmethod
    def _select_request_rows(input_batch: TrajectoryRequestBatch, indices: list[int]) -> TrajectoryRequestBatch:
        row_keys = ("prompts", "env_classes", "env_extras", "trajectory_ids")
        selected: dict[str, Any] = dict(input_batch)
        for key in row_keys:
            values = input_batch.get(key)
            if values is not None:
                selected[key] = [values[index] for index in indices]
        return selected  # type: ignore[return-value]

    @staticmethod
    def _validate_group_identity(input_batch: TrajectoryRequestBatch, output: TrajectoryBatch) -> None:
        expected = [trajectory_id.to_string() for trajectory_id in input_batch["trajectory_ids"] or []]
        returned_ids = output.get("trajectory_ids")
        if returned_ids is None:
            raise ValueError("trajectory runner output omitted trajectory IDs")
        returned = [trajectory_id.to_string() for trajectory_id in returned_ids]
        if len(returned) != len(set(returned)) or set(returned) != set(expected):
            raise ValueError(f"trajectory runner output identity mismatch: expected {expected}, got {returned}")

    @staticmethod
    def _restore_request_order(output: TrajectoryBatch, requested_ids: list[TrajectoryID]) -> None:
        returned_ids = output.get("trajectory_ids")
        assert returned_ids is not None
        returned_positions = {trajectory_id.to_string(): index for index, trajectory_id in enumerate(returned_ids)}
        order = [returned_positions[trajectory_id.to_string()] for trajectory_id in requested_ids]
        row_count = len(returned_ids)
        for key, values in list(output.items()):
            if isinstance(values, list) and len(values) == row_count:
                output[key] = [values[index] for index in order]  # type: ignore[literal-required]

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
    async def start_eval_session(self, *, run_name: str, eval_step: int, val_set_name: str | None = None) -> None:
        if self._actors:
            await self._actors[0].start_eval_session.remote(
                run_name=run_name, eval_step=eval_step, val_set_name=val_set_name
            )
            self._eval_session_active = True

    async def stop_eval_session(self) -> None:
        if self._actors:
            await self._actors[0].stop_eval_session.remote()
            self._eval_session_active = False
