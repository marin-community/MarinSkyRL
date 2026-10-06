"""Shared scalar observations and metrics for admitted rollout groups."""

from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Tuple, Union
import re

import numpy as np
import torch
from skyrl_gym.metrics import aggregate_for_environment
from skyrl_gym.verification import VerificationResult, VerificationStatus, normalized_verifier_score
from skyrl_train.trajectory_runners.types import (
    BatchFields,
    RolloutObservations,
    TitoFullDeclineReason,
    TrajectoryBatch,
)
from skyrl_train.trajectory_runners.trajectory_reward_shaping import NormalizedReward, observe_shaping, shaping_metrics
from skyrl_train.trajectory_runners.trajectory_retention import RETENTION_METRIC_PREFIX
from skyrl_train.metric_names import (
    ENVIRONMENT_METRIC_PREFIX,
    IDENTITY_AWARE_REWARD_METRIC_PREFIX,
    LITERAL_BRIDGE_CORRELATED_TRIALS_METRIC,
    LITERAL_BRIDGE_CORRELATED_TURNS_METRIC,
    TIS_ALIGNED_TOKENS_METRIC,
    TIS_ALIGNMENT_ALERT_METRIC,
    TIS_EXACT_MATCH_FRACTION_METRIC,
    TIS_LCS_FALLBACK_FRACTION_METRIC,
    TIS_UNALIGNED_FRACTION_METRIC,
    TIS_ALIGNMENT_FAIL_COUNT_METRIC,
    TIS_LCS_FALLBACK_MESSAGES_METRIC,
    TIS_LCS_FALLBACK_ALERT_METRIC,
    TIS_TITO_FULL_ATTEMPTS_METRIC,
    TIS_TITO_FULL_SUCCESS_FRACTION_METRIC,
    TIS_TITO_FULL_DECLINE_COUNT_METRIC,
    TIS_TITO_FULL_DECLINE_METRIC_PREFIX,
)
from skyrl_train.metric_names import ROLLOUT_FAILURE_FRACTION_METRIC

BATCH_ERROR_METRIC_PREFIX = "generate/errors/"
_NUM_TRIALS_METRIC = "generate/num_trials"
_NUM_FAILED_INSTANCES_METRIC = "generate/num_failed_instances"
_NUM_FAILED_TRAJECTORIES_METRIC = "generate/num_failed_trajectories"
_NUM_MASKED_TRAJECTORIES_METRIC = "generate/num_masked_trajectories"

_BATCH_FAILURE_COUNT_KEYS = (
    _NUM_TRIALS_METRIC,
    _NUM_FAILED_INSTANCES_METRIC,
    _NUM_FAILED_TRAJECTORIES_METRIC,
    _NUM_MASKED_TRAJECTORIES_METRIC,
)

MAX_DOMAIN_REWARD_METRICS = 32


@torch.no_grad()
def get_metrics_from_trajectory_batch(trajectory_batch: TrajectoryBatch, uids: List[str]) -> Tuple[float, float]:
    """Return mean optimization reward and prompt-group pass rate using raw outcomes and verifier verdicts."""
    rewards: Union[List[float], List[List[float]]] = trajectory_batch["rewards"]
    if not len(rewards):
        raise ValueError(f"`rewards` must be a non-empty list, got {rewards}")

    trajectory_passes = get_trajectory_passes(trajectory_batch)

    if isinstance(rewards[0], list):
        # Token-level rewards: rewards is List[List[float]]
        # For each trajectory, sum token rewards before computing the batch mean.
        mean_reward = float(np.mean([sum(trajectory_rewards) for trajectory_rewards in rewards]))
    else:
        mean_reward = float(np.mean(rewards))

    uid_to_trajectory_passes = defaultdict(list)
    for uid, passed in zip(uids, trajectory_passes, strict=True):
        uid_to_trajectory_passes[uid].append(passed)

    pass_at_n = sum(any(passes) for passes in uid_to_trajectory_passes.values()) / len(uid_to_trajectory_passes)

    return mean_reward, pass_at_n


def get_outcome_rewards(trajectory_batch: TrajectoryBatch) -> List[float]:
    """Return the unshaped task outcome associated with each trajectory."""
    rewards = trajectory_batch["rewards"]
    unshaped_rewards = trajectory_batch.get("unshaped_rewards")
    if unshaped_rewards is not None:
        if len(unshaped_rewards) != len(rewards):
            raise ValueError(
                "`unshaped_rewards` must have one entry per trajectory: "
                f"got {len(unshaped_rewards)} unshaped rewards and {len(rewards)} optimization rewards"
            )
        return [float(reward) for reward in unshaped_rewards]
    return [NormalizedReward.from_output(reward).outcome for reward in rewards]


def normalized_verifier_scores(trajectory_batch: TrajectoryBatch) -> List[float | None] | None:
    """Return bounded task scores, excluding skipped or missing verdicts and counting failures as zero."""
    results = trajectory_batch.get("verification_results")
    if results is None:
        return None
    if len(results) != len(trajectory_batch["rewards"]):
        raise ValueError("verification_results must have one entry per reward")
    scores: List[float | None] = []
    for result in results:
        if result is None or result.status is VerificationStatus.SKIPPED:
            scores.append(None)
        elif result.status is not VerificationStatus.VERIFIED:
            scores.append(0.0)
        else:
            scores.append(normalized_verifier_score(result))
    return scores


def verifier_score_summary(scores: List[float | None]) -> tuple[float, float | None]:
    """Return included-row coverage and mean bounded score."""
    included = [score for score in scores if score is not None]
    coverage = len(included) / len(scores) if scores else 0.0
    return coverage, float(np.mean(included)) if included else None


def graded_row_indices(trajectory_batch: TrajectoryBatch) -> List[int]:
    """Return rows whose environment ran grading; skipped rows carry no reward to report."""
    results = trajectory_batch.get("verification_results")
    if results is None:
        return list(range(len(trajectory_batch["rewards"])))
    return [
        index
        for index, result in enumerate(results)
        if result is None or result.status is not VerificationStatus.SKIPPED
    ]


def get_trajectory_passes(trajectory_batch: TrajectoryBatch) -> List[bool]:
    """Return task success, honoring explicit verifier verdicts when available."""
    outcomes = get_outcome_rewards(trajectory_batch)
    results = trajectory_batch.get("verification_results")
    if results is None:
        return [outcome > 0.0 for outcome in outcomes]
    passes = []
    for outcome, result in zip(outcomes, results, strict=True):
        if result is None:
            passes.append(outcome > 0.0)
        elif result.status is not VerificationStatus.VERIFIED:
            passes.append(False)
        else:
            passes.append(result.passed if result.passed is not None else outcome > 0.0)
    return passes


def rollout_metrics(observations: RolloutObservations, *, tis_lcs_alert_threshold: float) -> dict[str, float]:
    """Compute rollout, environment and group counters from ordered scalar observations."""
    verification_results = observations.verification_results
    env_metrics = observations.env_metrics
    env_classes = observations.env_classes
    num_tokens_arr = np.array(observations.response_lengths)
    successes = [value > 0.0 for value in observations.reward_sign_totals]
    if verification_results is not None:
        for index, result in enumerate(verification_results):
            if result is None:
                continue
            if result.status is not VerificationStatus.VERIFIED:
                successes[index] = False
            else:
                successes[index] = result.passed if result.passed is not None else float(result.score) > 0.0
    non_zero_rewards_arr = np.array(successes, dtype=bool)
    zero_rewards_arr = ~non_zero_rewards_arr
    # average tokens for non zero rewards
    avg_tokens_non_zero_rewards = (
        np.mean(num_tokens_arr[non_zero_rewards_arr]) if non_zero_rewards_arr.sum() > 0 else np.zeros(1)
    )
    # average tokens for zero rewards
    avg_tokens_zero_rewards = np.mean(num_tokens_arr[zero_rewards_arr]) if zero_rewards_arr.sum() > 0 else np.zeros(1)

    rollout_metrics = {
        "generate/min_num_tokens": np.min(num_tokens_arr).item(),
        "generate/max_num_tokens": np.max(num_tokens_arr).item(),
        "generate/avg_num_tokens": np.mean(num_tokens_arr).item(),
        "generate/std_num_tokens": np.std(num_tokens_arr).item(),
        "generate/avg_tokens_non_zero_rewards": avg_tokens_non_zero_rewards.item(),
        "generate/avg_tokens_zero_rewards": avg_tokens_zero_rewards.item(),
    }

    if env_metrics is not None and env_classes is not None:
        env_to_metrics = defaultdict(list)
        for i, metrics in enumerate(env_metrics):
            # Skipped episodes (e.g. over-length prompts) never step the environment and
            # report an empty dict; per-environment aggregators only see stepped episodes.
            if metrics:
                env_to_metrics[env_classes[i]].append(metrics)
        for env_name, metrics in env_to_metrics.items():
            # Aggregate metrics across all trajectories for the same environment
            agg = aggregate_for_environment(env_name, metrics)
            for key, value in agg.items():
                rollout_metrics[f"{ENVIRONMENT_METRIC_PREFIX}{key}"] = value

    if observations.group_metrics:
        # TIS alignment metrics use token-weighted fractions across batches.
        total_aligned = 0.0
        sum_exact = sum_lcs = sum_unaligned = 0.0
        sum_fail = sum_lcs_msgs = 0.0
        saw_tis = False
        for group_metrics in observations.group_metrics:
            rm = group_metrics
            n = rm.get(TIS_ALIGNED_TOKENS_METRIC)
            if n is None:
                continue
            saw_tis = True
            total_aligned += n
            sum_exact += rm.get(TIS_EXACT_MATCH_FRACTION_METRIC, 0.0) * n
            sum_lcs += rm.get(TIS_LCS_FALLBACK_FRACTION_METRIC, 0.0) * n
            sum_unaligned += rm.get(TIS_UNALIGNED_FRACTION_METRIC, 0.0) * n
            sum_fail += rm.get(TIS_ALIGNMENT_FAIL_COUNT_METRIC, 0.0)
            sum_lcs_msgs += rm.get(TIS_LCS_FALLBACK_MESSAGES_METRIC, 0.0)
        if saw_tis:
            denom = max(total_aligned, 1.0)
            rollout_metrics[TIS_ALIGNED_TOKENS_METRIC] = total_aligned
            rollout_metrics[TIS_EXACT_MATCH_FRACTION_METRIC] = sum_exact / denom
            rollout_metrics[TIS_LCS_FALLBACK_FRACTION_METRIC] = sum_lcs / denom
            rollout_metrics[TIS_UNALIGNED_FRACTION_METRIC] = sum_unaligned / denom
            rollout_metrics[TIS_ALIGNMENT_FAIL_COUNT_METRIC] = sum_fail
            rollout_metrics[TIS_LCS_FALLBACK_MESSAGES_METRIC] = sum_lcs_msgs
            lcs_alert = 1.0 if (sum_lcs / denom) > tis_lcs_alert_threshold else 0.0
            rollout_metrics[TIS_LCS_FALLBACK_ALERT_METRIC] = lcs_alert
            total_tito_attempts = sum(
                group_metrics.get(TIS_TITO_FULL_ATTEMPTS_METRIC, 0.0) for group_metrics in observations.group_metrics
            )
            total_tito_successes = sum(
                group_metrics.get(TIS_TITO_FULL_SUCCESS_FRACTION_METRIC, 0.0)
                * group_metrics.get(TIS_TITO_FULL_ATTEMPTS_METRIC, 0.0)
                for group_metrics in observations.group_metrics
            )
            rollout_metrics[TIS_TITO_FULL_ATTEMPTS_METRIC] = total_tito_attempts
            rollout_metrics[TIS_TITO_FULL_SUCCESS_FRACTION_METRIC] = (
                total_tito_successes / total_tito_attempts if total_tito_attempts else 0.0
            )
            total_tito_declines = sum(
                group_metrics.get(TIS_TITO_FULL_DECLINE_COUNT_METRIC, 0.0)
                for group_metrics in observations.group_metrics
            )
            rollout_metrics[TIS_TITO_FULL_DECLINE_COUNT_METRIC] = total_tito_declines
            rollout_metrics[TIS_ALIGNMENT_ALERT_METRIC] = (
                1.0 if sum_unaligned > 0 or lcs_alert or total_tito_declines > 0 else 0.0
            )
            for reason in TitoFullDeclineReason:
                name = f"{TIS_TITO_FULL_DECLINE_METRIC_PREFIX}{reason.value}"
                rollout_metrics[name] = sum(
                    group_metrics.get(name, 0.0) for group_metrics in observations.group_metrics
                )

        rollout_metrics.update(_merge_batch_failure_metrics(observations.group_metrics))

        # Retention and identity counters count events within each group, so sum them across groups.
        for group_metrics in observations.group_metrics:
            for name, value in group_metrics.items():
                if name.startswith((RETENTION_METRIC_PREFIX, IDENTITY_AWARE_REWARD_METRIC_PREFIX)) or name in {
                    LITERAL_BRIDGE_CORRELATED_TRIALS_METRIC,
                    LITERAL_BRIDGE_CORRELATED_TURNS_METRIC,
                }:
                    rollout_metrics[name] = rollout_metrics.get(name, 0.0) + value

    if observations.shaping is not None:
        rollout_metrics.update(shaping_metrics(observations.shaping))
    return rollout_metrics


def _merge_batch_failure_metrics(group_observations: Sequence[dict[str, float]]) -> dict[str, float]:
    """Sum failure counts across rollout groups and recompute their batch fraction."""
    group_metrics = [metrics for metrics in group_observations if _NUM_TRIALS_METRIC in metrics]
    if not group_metrics:
        return {}
    for metrics in group_metrics:
        missing = [key for key in _BATCH_FAILURE_COUNT_KEYS if key not in metrics]
        if missing:
            raise ValueError(f"Incomplete rollout failure metrics: missing {', '.join(missing)}")

    merged = get_batch_failure_metrics(
        sum(metrics[_NUM_TRIALS_METRIC] for metrics in group_metrics),
        num_failed_trajectories=sum(metrics[_NUM_FAILED_TRAJECTORIES_METRIC] for metrics in group_metrics),
        num_failed_instances=sum(metrics[_NUM_FAILED_INSTANCES_METRIC] for metrics in group_metrics),
        num_masked_trajectories=sum(metrics[_NUM_MASKED_TRAJECTORIES_METRIC] for metrics in group_metrics),
    )
    error_keys = {key for metrics in group_metrics for key in metrics if key.startswith(BATCH_ERROR_METRIC_PREFIX)}
    merged.update({key: sum(metrics.get(key, 0) for metrics in group_metrics) for key in error_keys})
    return merged


def get_batch_failure_metrics(
    num_trials: int,
    num_failed_trajectories: int,
    num_failed_instances: int,
    num_masked_trajectories: int,
) -> Dict[str, float]:
    """Describe failed and masked trajectories within a requested rollout group."""
    return {
        _NUM_TRIALS_METRIC: num_trials,
        _NUM_FAILED_INSTANCES_METRIC: num_failed_instances,
        _NUM_FAILED_TRAJECTORIES_METRIC: num_failed_trajectories,
        _NUM_MASKED_TRAJECTORIES_METRIC: num_masked_trajectories,
        ROLLOUT_FAILURE_FRACTION_METRIC: num_failed_trajectories / num_trials if num_trials else 0.0,
    }


def _domain_metric_source_key(source: str | None) -> str:
    """Encode a source as a distinct tracker-safe path segment, reserving _missing for absent metadata."""
    if source is None:
        return "_missing"
    if re.fullmatch(r"[a-z_][a-z0-9_]*", source) and source != "_missing" and not source.startswith("_source_"):
        return source
    encoded = "".join(
        chr(byte) if byte in b"abcdefghijklmnopqrstuvwxyz0123456789" else f"_{byte:02x}"
        for byte in source.encode("utf-8")
    )
    return f"_source_{encoded}"


def _domain_reward_metrics(data_sources: List[str | None], rewards: List[float]) -> Dict[str, float]:
    """Return bounded per-source means with distinct, stable metric names."""
    if len(data_sources) != len(rewards):
        raise ValueError(
            f"Expected one data source per reward, got {len(data_sources)} sources and {len(rewards)} rewards"
        )

    rewards_by_source: Dict[str, List[float]] = defaultdict(list)
    for source, reward in zip(data_sources, rewards, strict=True):
        rewards_by_source[_domain_metric_source_key(source)].append(reward)

    sources = sorted(rewards_by_source)
    metrics = {
        f"reward/domain/{source}/avg_raw_reward": float(np.mean(rewards_by_source[source]))
        for source in sources[:MAX_DOMAIN_REWARD_METRICS]
    }
    if len(sources) > MAX_DOMAIN_REWARD_METRICS:
        overflow = [reward for source in sources[MAX_DOMAIN_REWARD_METRICS:] for reward in rewards_by_source[source]]
        metrics["reward/domain_overflow/avg_raw_reward"] = float(np.mean(overflow))
    return metrics


def observe_rollout(batch: TrajectoryBatch, *, fields: BatchFields) -> RolloutObservations:
    """Capture scalar metrics from rows normalized under the supplied whole-batch field policy."""
    rewards = batch["rewards"]
    token_rewards = any(isinstance(reward, list) for reward in rewards)
    return RolloutObservations(
        response_lengths=tuple(len(response) for response in batch["response_ids"]),
        optimization_totals=tuple(sum(reward) if isinstance(reward, list) else reward for reward in rewards),
        reward_sign_totals=tuple(float(np.sum(reward)) for reward in rewards),
        outcomes=tuple(NormalizedReward.from_output(reward).outcome for reward in rewards),
        unshaped_outcomes=(tuple(get_outcome_rewards(batch)) if batch.get("unshaped_rewards") is not None else None),
        scalar_rewards=None if token_rewards else tuple(float(reward) for reward in rewards),
        is_last_step=tuple(batch["is_last_step"]) if batch.get("is_last_step") is not None else None,
        verification_results=(
            tuple(batch["verification_results"]) if batch.get("verification_results") is not None else None
        ),
        data_sources=tuple(batch["data_sources"]) if batch.get("data_sources") is not None else None,
        env_metrics=tuple(batch["env_metrics"]) if batch.get("env_metrics") is not None else None,
        env_classes=tuple(batch["env_classes"]) if batch.get("env_classes") is not None else None,
        group_metrics=(batch.get("rollout_metrics") or {},),
        shaping=observe_shaping(batch, fields=fields),
    )


def reward_metrics(
    observations: RolloutObservations,
    uids: Sequence[str],
    *,
    n_samples_per_prompt: int,
    step_wise: bool,
) -> dict[str, float]:
    """Compute reward and verifier metrics over graded rows and complete prompt groups."""
    batch = {
        "rewards": list(observations.optimization_totals),
        "unshaped_rewards": list(
            observations.unshaped_outcomes if observations.unshaped_outcomes is not None else observations.outcomes
        ),
    }
    if observations.verification_results is not None:
        batch["verification_results"] = list(observations.verification_results)
    indices = graded_row_indices(batch)
    if step_wise:
        if observations.is_last_step is None:
            raise ValueError("step-wise reward metrics require is_last_step")
        indices = [index for index in indices if observations.is_last_step[index]]
    if not indices:
        return {}
    batch = {key: [values[index] for index in indices] for key, values in batch.items()}
    selected_uids = [uids[index] for index in indices]
    mean_reward, pass_at_n = get_metrics_from_trajectory_batch(batch, selected_uids)
    metrics = {
        f"reward/avg_pass_at_{n_samples_per_prompt}": pass_at_n,
        "reward/avg_raw_reward": mean_reward,
    }
    grouped_rewards = defaultdict(list)
    for uid, index in zip(selected_uids, indices, strict=True):
        grouped_rewards[uid].append(observations.reward_sign_totals[index])
    metrics["reward/informative_group_fraction"] = sum(
        max(values) > min(values) for values in grouped_rewards.values()
    ) / len(grouped_rewards)
    verifier_scores = normalized_verifier_scores(batch)
    if verifier_scores is not None:
        coverage, average = verifier_score_summary(verifier_scores)
        metrics["reward/verifier_score_coverage"] = coverage
        if average is not None:
            metrics["reward/avg_verifier_score"] = average
        scores_by_agent = defaultdict(list)
        for result, score in zip(batch["verification_results"], verifier_scores, strict=True):
            if result is not None and score is not None and isinstance(result.diagnostics.get("agent"), str):
                scores_by_agent[_domain_metric_source_key(result.diagnostics["agent"])].append(score)
        for agent in sorted(scores_by_agent)[:MAX_DOMAIN_REWARD_METRICS]:
            metrics[f"reward/agent/{agent}/avg_verifier_score"] = float(np.mean(scores_by_agent[agent]))
    if observations.data_sources is not None:
        metrics.update(
            _domain_reward_metrics(
                [observations.data_sources[index] for index in indices], [float(reward) for reward in batch["rewards"]]
            )
        )
    return metrics


def staleness_metrics(policy_steps: Sequence[int], *, global_step: int, max_staleness_steps: int) -> dict[str, float]:
    """Validate admitted staleness and summarize it with equal weight for each group."""
    stalenesses = [global_step - policy_step for policy_step in policy_steps]
    if max(stalenesses) > max_staleness_steps:
        raise ValueError("batch staleness exceeds the configured maximum")
    return {
        "async/staleness_mean": sum(stalenesses) / len(stalenesses),
        "async/staleness_max": max(stalenesses),
        "async/staleness_min": min(stalenesses),
        "async/staleness_ratio": sum(value > 0 for value in stalenesses) / len(stalenesses),
    }


def get_rollout_metrics(
    responses: List[List[int]],
    rewards: Union[List[float], List[List[float]]],
    env_metrics: Optional[List[Dict[str, Any]]] = None,
    env_classes: Optional[List[str]] = None,
    verification_results: Optional[List[Optional[VerificationResult]]] = None,
) -> dict[str, float]:
    """Compute rollout metrics from response rows and environment results."""
    batch = {"response_ids": responses, "rewards": rewards}
    if env_metrics is not None:
        batch["env_metrics"] = env_metrics
    if env_classes is not None:
        batch["env_classes"] = env_classes
    if verification_results is not None:
        batch["verification_results"] = verification_results
    observations = observe_rollout(batch, fields=BatchFields.from_batch(batch))
    return rollout_metrics(observations, tis_lcs_alert_threshold=0.0)
