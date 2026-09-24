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


def propagate_teacher_routes(input_batch: TrajectoryRequestBatch, output: TrajectoryBatch) -> None:
    """Carry explicit dataset routing through a trajectory-runner boundary."""
    env_extras = input_batch.get("env_extras")
    if not env_extras:
        return
    supplied = ["teacher_route" in extras for extras in env_extras]
    if not any(supplied):
        return
    if not all(supplied):
        raise ValueError("teacher_route must be present on every request row when teacher routing is used")

    route_keys = [extras["teacher_route"] for extras in env_extras]
    if any(not isinstance(route_key, str) or not route_key.strip() for route_key in route_keys):
        raise ValueError("teacher_route must be a non-empty string on every request row")
    if len(route_keys) != len(output["response_ids"]):
        raise ValueError("teacher_route rows must align with trajectory runner output rows")

    output_route_keys = output.get("teacher_route_keys")
    if output_route_keys is not None and output_route_keys != route_keys:
        raise ValueError("trajectory runner output teacher_route_keys do not match request metadata")
    output["teacher_route_keys"] = route_keys


def propagate_data_sources(input_batch: TrajectoryRequestBatch, output: TrajectoryBatch) -> None:
    """Keep request source labels aligned with whole or step-wise rollout rows."""
    env_extras = input_batch.get("env_extras")
    if env_extras is None:
        return
    sources = []
    for extras in env_extras:
        extra_info = extras.get("extra_info")
        source = extra_info.get("data_source") if isinstance(extra_info, dict) else None
        if source is None:
            source = extras.get("data_source")
        sources.append(source if isinstance(source, str) else None)

    if len(sources) == len(output["response_ids"]):
        output["data_sources"] = sources
        return

    request_ids = input_batch.get("trajectory_ids")
    output_ids = output.get("trajectory_ids")
    if request_ids is None or output_ids is None:
        return
    if len(request_ids) != len(sources) or len(output_ids) != len(output["response_ids"]):
        raise ValueError("trajectory IDs and source labels must align with their rows")
    sources_by_id = {(item.instance_id, item.repetition_id): source for item, source in zip(request_ids, sources)}
    if len(sources_by_id) != len(request_ids):
        raise ValueError("request trajectory IDs must be unique to map source labels")
    try:
        output["data_sources"] = [sources_by_id[(item.instance_id, item.repetition_id)] for item in output_ids]
    except KeyError as error:
        raise ValueError("output trajectory ID has no matching request source label") from error


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

    async def run(self, input_batch: TrajectoryRequestBatch, disable_tqdm: bool = False) -> TrajectoryBatch:
        """Acquire trajectories and apply runner-independent output finalization.

        Returns outputs in the same order as the input batch.

        Args:
            input_batch (TrajectoryRequestBatch): Input batch
        Returns:
            TrajectoryBatch: Generated trajectories
        """
        output = await self._run(input_batch, disable_tqdm=disable_tqdm)
        trajectory_ids = input_batch.get("trajectory_ids")
        if trajectory_ids is not None and output.get("trajectory_ids") is None:
            if len(trajectory_ids) != len(output["response_ids"]):
                raise ValueError("trajectory runner output rows must align with request trajectory IDs")
            output["trajectory_ids"] = list(trajectory_ids)
        propagate_teacher_routes(input_batch, output)
        propagate_data_sources(input_batch, output)
        return await self._finalize_output(input_batch, output)

    async def _finalize_output(self, input_batch: TrajectoryRequestBatch, output: TrajectoryBatch) -> TrajectoryBatch:
        """Apply runner-independent shaping, metrics, and retention."""
        shape_trajectory_rewards(output, self.trajectory_runner_cfg.get("trajectory_reward_shaping"))
        self._add_alignment_metrics(output)
        if self.trajectory_sink is not None:
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
