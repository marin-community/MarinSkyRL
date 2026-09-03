from abc import ABC, abstractmethod
from types import MappingProxyType
from skyrl_train.metric_names import (
    TIS_ALIGNED_TOKENS_METRIC,
    TIS_ALIGNMENT_ALERT_METRIC,
    TIS_ALIGNMENT_FAIL_COUNT_METRIC,
    TIS_EXACT_MATCH_FRACTION_METRIC,
    TIS_LCS_FALLBACK_ALERT_METRIC,
    TIS_LCS_FALLBACK_FRACTION_METRIC,
    TIS_LCS_FALLBACK_MESSAGES_METRIC,
    TIS_UNALIGNED_FRACTION_METRIC,
)
from skyrl_train.timing_observability import RolloutTimings, rollout_span, rollout_timings_scope
from skyrl_train.trajectory_runners.types import (
    BatchMetadata as BatchMetadata,
    ConversationType as ConversationType,
    TrajectoryBatch as TrajectoryBatch,
    TrajectoryID as TrajectoryID,
    TrajectoryRequestBatch as TrajectoryRequestBatch,
    TrainingPhase as TrainingPhase,
)
from skyrl_train.trajectory_runners.trajectory_reward_shaping import shape_trajectory_rewards
from skyrl_train.trajectory_runners.trajectory_retention import TrajectorySink, retain_trajectories


class TrajectoryRunner(ABC):
    """Abstract base class for acquiring trainer-ready trajectories.

    Lifecycle:
        1. __init__() - Synchronous initialization (no async resources)
        2. startup() - Async initialization of resources (e.g., orchestrators, connections)
        3. run() - Called repeatedly during training
        4. shutdown() - Async cleanup of resources

    Implementations should handle errors gracefully in run() to avoid killing the
    training job. Use restart logic for recoverable failures.
    """

    trajectory_runner_cfg = MappingProxyType({})
    trajectory_sink: TrajectorySink | None = None

    #: Whether every call site inside this runner's rollout is bracketed for the generate span tree.
    #: False is the safe default and means this runner publishes NO generate spans -- not even a
    #: residual. A residual equal to the whole parent from an unbracketed runner would read as
    #: "generate is entirely unaccounted for", which is a claim about the rollout rather than about
    #: the instrument. A new runner opts in only after bracketing its own engine and environment
    #: waits; until then absence is the honest signal.
    generate_spans_instrumented: bool = False

    #: Overriding any of these replaces bracketed code, so a subclass that does so without
    #: re-declaring the certificate loses it. ``_run`` drives the rollout; ``agent_loop`` and
    #: ``collect_batched`` ARE the bracketed loops -- they hold the engine awaits, the environment
    #: calls and the trajectory scopes, and a subclass overriding one of them keeps every seeded
    #: zero while measuring none of it. An earlier version listed only ``_run``, which left the two
    #: methods the certificate is actually about uncovered.
    BRACKETED_METHODS = ("_run", "agent_loop", "collect_batched")

    def __init_subclass__(cls, **kwargs) -> None:
        """Revoke an inherited instrumented flag from a subclass that replaces bracketed code.

        A subclass that overrides one of BRACKETED_METHODS without re-declaring the flag would
        inherit True and publish a seeded all-zero decomposition -- the "measured zero" lie the flag
        exists to prevent, made indistinguishable from truth by the explicit zeros.
        """
        super().__init_subclass__(**kwargs)
        if "generate_spans_instrumented" in cls.__dict__:
            return
        if any(method in cls.__dict__ for method in TrajectoryRunner.BRACKETED_METHODS):
            cls.generate_spans_instrumented = False

    async def run(
        self,
        input_batch: TrajectoryRequestBatch,
        disable_tqdm: bool = False,
        *,
        phase_timings: RolloutTimings | None = None,
    ) -> TrajectoryBatch:
        """Acquire trajectories and apply runner-independent output finalization.

        Returns outputs in the same order as the input batch.

        Args:
            input_batch (TrajectoryRequestBatch): Input batch
            phase_timings: accumulator for the generate span tree, or None to measure nothing. A
                caller that runs several run() calls concurrently must pass None: overlapping walls
                accumulated into one dict are not a decomposition of anything.
        Returns:
            TrajectoryBatch: Generated trajectories
        """
        if phase_timings is not None and self.generate_spans_instrumented:
            phase_timings.mark_supported()
        else:
            phase_timings = None
        with rollout_timings_scope(phase_timings):
            # rollout_collect / rollout_assemble are opened by the runner's own _run, which knows
            # where its fan-out ends and its projection begins. Only the shared epilogue is bracketed
            # here, because only it is shared.
            output = await self._run(input_batch, disable_tqdm=disable_tqdm)
            trajectory_ids = input_batch.get("trajectory_ids")
            if trajectory_ids is not None and output.get("trajectory_ids") is None:
                if len(trajectory_ids) != len(output["response_ids"]):
                    raise ValueError("trajectory runner output rows must align with request trajectory IDs")
                output["trajectory_ids"] = list(trajectory_ids)
            with rollout_span("rollout_finalize"):
                return await self._finalize_output(input_batch, output)

    async def _finalize_output(self, input_batch: TrajectoryRequestBatch, output: TrajectoryBatch) -> TrajectoryBatch:
        """Apply runner-independent shaping, metrics, and retention."""
        shape_trajectory_rewards(output, self.trajectory_runner_cfg.get("trajectory_reward_shaping"))
        self._add_alignment_metrics(output)
        if self.trajectory_sink is not None:
            # On by default, and a blocking write of the whole batch -- ~8 MiB at E6 geometry -- on
            # the event-loop thread. Without its own leaf it lands in generate_span_residual, which
            # already has several known occupants and so explains nothing.
            with rollout_span("rollout_retain"):
                await retain_trajectories(self.trajectory_sink, input_batch, output)
        return output

    def set_trajectory_sink(self, sink: TrajectorySink) -> None:
        """Attach the trainer-owned sink used by shared output finalization."""
        sink.bind_runner(type(self).__name__)
        self.trajectory_sink = sink

    async def start_eval_session(
        self,
        *,
        run_name: str,
        eval_step: int,
        val_set_name: str | None = None,
        n_concurrent_trials: int | None = None,
    ) -> None:
        """Start an evaluation-scoped resource session when a runner needs one."""

    async def stop_eval_session(self) -> None:
        """Stop resources created for the current evaluation session."""

    @abstractmethod
    async def _run(self, input_batch: TrajectoryRequestBatch, disable_tqdm: bool = False) -> TrajectoryBatch:
        """Produce trajectories before shared output finalization."""
        raise NotImplementedError()

    @staticmethod
    def _add_alignment_metrics(output: TrajectoryBatch) -> None:
        """Expose alignment health implied by the ``TrajectoryBatch`` contract.

        A runner that returns rollout logprobs promises they are position-aligned
        with its response IDs. That direct token-in/token-out path is exact by
        construction. Runners that reconstruct token streams can publish richer
        exact/LCS/failure metrics themselves; those observations take precedence.
        """
        rollout_logprobs = output.get("rollout_logprobs")
        if rollout_logprobs is None:
            return

        rollout_metrics = output.get("rollout_metrics") or {}
        if TIS_ALIGNED_TOKENS_METRIC in rollout_metrics:
            return

        response_ids = output["response_ids"]
        loss_masks = output["loss_masks"]
        if not (len(response_ids) == len(loss_masks) == len(rollout_logprobs)):
            raise ValueError("response IDs, loss masks, and rollout logprobs must have the same batch size")

        aligned_tokens = 0
        for sample_response_ids, sample_loss_mask, sample_logprobs in zip(response_ids, loss_masks, rollout_logprobs):
            if not (len(sample_response_ids) == len(sample_loss_mask) == len(sample_logprobs)):
                raise ValueError("rollout logprobs must align one-for-one with response IDs and loss masks")
            aligned_tokens += sum(bool(value) for value in sample_loss_mask)

        rollout_metrics.update(
            {
                TIS_ALIGNED_TOKENS_METRIC: float(aligned_tokens),
                TIS_EXACT_MATCH_FRACTION_METRIC: 1.0 if aligned_tokens else 0.0,
                TIS_LCS_FALLBACK_FRACTION_METRIC: 0.0,
                TIS_UNALIGNED_FRACTION_METRIC: 0.0,
                TIS_ALIGNMENT_FAIL_COUNT_METRIC: 0.0,
                TIS_LCS_FALLBACK_MESSAGES_METRIC: 0.0,
                TIS_LCS_FALLBACK_ALERT_METRIC: 0.0,
                TIS_ALIGNMENT_ALERT_METRIC: 0.0,
            }
        )
        output["rollout_metrics"] = rollout_metrics

    async def startup(self) -> None:
        """Initialize runner resources before the first call to :meth:`run`."""
        pass

    async def shutdown(self) -> None:
        """Release runner resources after use; repeated calls must be safe."""
        pass
