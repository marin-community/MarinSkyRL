"""Training metadata, reward shaping, token checks, and rollout retention."""

from collections.abc import Mapping
from typing import Any

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
from skyrl_train.trajectory_runners.types import TrajectoryBatch, TrajectoryRequestBatch
from skyrl_train.trajectory_runners.trajectory_reward_shaping import shape_trajectory_rewards
from skyrl_train.trajectory_runners.trajectory_retention import RetentionSink, retain_trajectories
from skyrl_train.rollout_observability import rollout_phase


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
    request_ids = input_batch.get("trajectory_ids")
    output_ids = output.get("trajectory_ids")
    if request_ids is not None and output_ids is not None:
        if len(request_ids) != len(route_keys):
            raise ValueError("request trajectory IDs must align with teacher routes")
        routes_by_id = {(item.instance_id, item.repetition_id): key for item, key in zip(request_ids, route_keys)}
        if len(routes_by_id) != len(request_ids):
            raise ValueError("request trajectory IDs must be unique to map teacher routes")
        route_keys = [routes_by_id[(item.instance_id, item.repetition_id)] for item in output_ids]
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

    request_ids = input_batch.get("trajectory_ids")
    output_ids = output.get("trajectory_ids")
    if request_ids is None or output_ids is None:
        if len(sources) == len(output["response_ids"]):
            output["data_sources"] = sources
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


async def finalize_trajectory_batch(
    input_batch: TrajectoryRequestBatch,
    output: TrajectoryBatch,
    config: Mapping[str, Any],
    sink: RetentionSink | None,
) -> TrajectoryBatch:
    """Attach request identity, apply rewards, and retain the completed output."""
    trajectory_ids = input_batch.get("trajectory_ids")
    if trajectory_ids is not None and output.get("trajectory_ids") is None:
        if len(trajectory_ids) != len(output["response_ids"]):
            raise ValueError("trajectory runner output rows must align with request trajectory IDs")
        output["trajectory_ids"] = list(trajectory_ids)
    propagate_teacher_routes(input_batch, output)
    metrics = output.get("rollout_metrics") or {}
    for extras in input_batch.get("env_extras") or []:
        ultra = (extras.get("extra_info") or {}).get("nemotron_ultra")
        if ultra is not None and "blend" in ultra and "agent" in ultra:
            key = f"nemotron_ultra/coverage/{ultra['blend']}/{ultra['agent']}"
            metrics[key] = metrics.get(key, 0) + 1
    output["rollout_metrics"] = metrics
    with rollout_phase("finalize"):
        shape_trajectory_rewards(output, config.get("trajectory_reward_shaping"))
        add_alignment_metrics(output)
        if sink is not None:
            with rollout_phase("retain"):
                await retain_trajectories(sink, input_batch, output)
    return output


def add_alignment_metrics(output: TrajectoryBatch) -> None:
    """Expose alignment health implied by the ``TrajectoryBatch`` contract.

    A runner that returns rollout logprobs promises they are position-aligned
    with its response IDs. That direct token-in/token-out path is exact by
    construction. Preserve metrics that a runner supplied.
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
