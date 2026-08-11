from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re
from typing import Any

import numpy as np

from skyrl_train.generators.generator_types import GeneratorOutput, RewardShapingComponents, RewardShapingLoopSpan


SHAPING_METRIC_PREFIX = "generate/reward_shaping"
SHAPING_SCHEMA_VERSION = 1
REWARD_SHAPING_ROW_KEYS = (
    "reward_shaping_components",
    "reward_shaping_loop_spans",
    "reward_shaping_versions",
)
_HASH_BASE = 1_000_003
_HASH_MASK = (1 << 64) - 1
_DEFAULT_ACCEPTED_STOP_REASONS = ("complete", "end_turn", "eos", "stop")


@dataclass(frozen=True)
class LoopPenaltyConfig:
    window_tokens: int = 16
    minimum_occurrences: int = 3
    penalty_per_occurrence: float = 0.0
    max_penalty: float = 0.2


@dataclass(frozen=True)
class NonTerminationPenaltyConfig:
    penalty: float = 0.0
    accepted_stop_reasons: tuple[str, ...] = _DEFAULT_ACCEPTED_STOP_REASONS


@dataclass(frozen=True)
class SuccessfulLengthPenaltyConfig:
    free_tokens: int = 0
    penalty_per_token: float = 0.0
    max_penalty: float = 0.2


@dataclass(frozen=True)
class TrajectoryRewardShapingConfig:
    schema_version: int = SHAPING_SCHEMA_VERSION
    enabled: bool = False
    loop: LoopPenaltyConfig = LoopPenaltyConfig()
    non_termination: NonTerminationPenaltyConfig = NonTerminationPenaltyConfig()
    successful_length: SuccessfulLengthPenaltyConfig = SuccessfulLengthPenaltyConfig()


@dataclass
class _RepeatedWindow:
    representative_start: int
    occurrences: int = 1


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"generator.trajectory_reward_shaping.{name} must be a mapping")
    return value


def parse_trajectory_reward_shaping_config(config: Mapping[str, Any] | None) -> TrajectoryRewardShapingConfig:
    """Parse and validate the generator-independent reward-shaping configuration."""
    if config is None:
        return TrajectoryRewardShapingConfig()
    if not isinstance(config, Mapping):
        raise ValueError("generator.trajectory_reward_shaping must be a mapping")

    loop = _section(config, "loop")
    non_termination = _section(config, "non_termination")
    successful_length = _section(config, "successful_length")
    defaults = TrajectoryRewardShapingConfig()
    raw_stop_reasons = non_termination.get("accepted_stop_reasons", defaults.non_termination.accepted_stop_reasons)
    if not isinstance(raw_stop_reasons, Sequence) or isinstance(raw_stop_reasons, (str, bytes)):
        raise ValueError("non_termination.accepted_stop_reasons must be a sequence of strings")
    accepted_stop_reasons = tuple(str(reason).strip().lower() for reason in raw_stop_reasons)
    parsed = TrajectoryRewardShapingConfig(
        schema_version=int(config.get("schema_version", defaults.schema_version)),
        enabled=bool(config.get("enabled", defaults.enabled)),
        loop=LoopPenaltyConfig(
            window_tokens=int(loop.get("window_tokens", defaults.loop.window_tokens)),
            minimum_occurrences=int(loop.get("minimum_occurrences", defaults.loop.minimum_occurrences)),
            penalty_per_occurrence=float(loop.get("penalty_per_occurrence", defaults.loop.penalty_per_occurrence)),
            max_penalty=float(loop.get("max_penalty", defaults.loop.max_penalty)),
        ),
        non_termination=NonTerminationPenaltyConfig(
            penalty=float(non_termination.get("penalty", defaults.non_termination.penalty)),
            accepted_stop_reasons=accepted_stop_reasons,
        ),
        successful_length=SuccessfulLengthPenaltyConfig(
            free_tokens=int(successful_length.get("free_tokens", defaults.successful_length.free_tokens)),
            penalty_per_token=float(
                successful_length.get("penalty_per_token", defaults.successful_length.penalty_per_token)
            ),
            max_penalty=float(successful_length.get("max_penalty", defaults.successful_length.max_penalty)),
        ),
    )
    _validate_config(parsed)
    return parsed


def _validate_config(config: TrajectoryRewardShapingConfig) -> None:
    if config.schema_version != SHAPING_SCHEMA_VERSION:
        raise ValueError(
            f"trajectory reward shaping schema_version must be {SHAPING_SCHEMA_VERSION}, got {config.schema_version}"
        )
    if config.loop.window_tokens < 1:
        raise ValueError("loop.window_tokens must be at least 1")
    if config.loop.minimum_occurrences < 2:
        raise ValueError("loop.minimum_occurrences must be at least 2")
    if config.successful_length.free_tokens < 0:
        raise ValueError("successful_length.free_tokens must be non-negative")

    magnitudes = {
        "loop.penalty_per_occurrence": config.loop.penalty_per_occurrence,
        "loop.max_penalty": config.loop.max_penalty,
        "non_termination.penalty": config.non_termination.penalty,
        "successful_length.penalty_per_token": config.successful_length.penalty_per_token,
        "successful_length.max_penalty": config.successful_length.max_penalty,
    }
    for name, value in magnitudes.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    if config.non_termination.penalty > 0 and not config.non_termination.accepted_stop_reasons:
        raise ValueError("non_termination.accepted_stop_reasons cannot be empty when its penalty is positive")


def _active_segments(token_ids: Sequence[int], loss_mask: Sequence[int]) -> list[tuple[list[int], list[int]]]:
    if len(token_ids) != len(loss_mask):
        raise ValueError("response IDs and loss masks must have the same length")

    segments: list[tuple[list[int], list[int]]] = []
    segment_tokens: list[int] = []
    segment_positions: list[int] = []
    for position, (token_id, active) in enumerate(zip(token_ids, loss_mask)):
        if active:
            segment_tokens.append(int(token_id))
            segment_positions.append(position)
        elif segment_tokens:
            segments.append((segment_tokens, segment_positions))
            segment_tokens = []
            segment_positions = []
    if segment_tokens:
        segments.append((segment_tokens, segment_positions))
    return segments


def _window_hashes(tokens: Sequence[int], window: int) -> list[int]:
    if len(tokens) < window:
        return []

    high_place = pow(_HASH_BASE, window - 1, 1 << 64)
    current = 0
    for token in tokens[:window]:
        current = ((current * _HASH_BASE) + int(token) + 1) & _HASH_MASK
    hashes = [current]
    for start in range(1, len(tokens) - window + 1):
        outgoing = int(tokens[start - 1]) + 1
        incoming = int(tokens[start + window - 1]) + 1
        current = (current - outgoing * high_place) & _HASH_MASK
        current = ((current * _HASH_BASE) + incoming) & _HASH_MASK
        hashes.append(current)
    return hashes


def _merge_spans(spans: list[tuple[int, int]]) -> list[RewardShapingLoopSpan]:
    if not spans:
        return []
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [{"start": start, "end": end} for start, end in merged]


def _repeated_spans(
    response_ids: Sequence[int],
    loss_mask: Sequence[int],
    config: LoopPenaltyConfig,
) -> tuple[list[RewardShapingLoopSpan], int]:
    repeated_spans: list[tuple[int, int]] = []
    penalized_occurrences = 0
    window = config.window_tokens

    # Tool observations split active regions, so repeated commands after changed
    # observations are not collapsed into one textual loop.
    for segment_tokens, segment_positions in _active_segments(response_ids, loss_mask):
        states: dict[int, list[_RepeatedWindow]] = {}
        for start, window_hash in enumerate(_window_hashes(segment_tokens, window)):
            matching_state = None
            for state in states.get(window_hash, []):
                representative = segment_tokens[state.representative_start : state.representative_start + window]
                if representative == segment_tokens[start : start + window]:
                    matching_state = state
                    break

            if matching_state is None:
                states.setdefault(window_hash, []).append(_RepeatedWindow(representative_start=start))
                continue

            matching_state.occurrences += 1
            if matching_state.occurrences < config.minimum_occurrences:
                continue
            penalized_occurrences += 1
            repeated_spans.append((segment_positions[start], segment_positions[start + window - 1] + 1))

    return _merge_spans(repeated_spans), penalized_occurrences


def _outcome_reward(reward: float | Sequence[float]) -> float:
    if isinstance(reward, Sequence) and not isinstance(reward, (str, bytes)):
        return float(reward[-1]) if reward else 0.0
    return float(reward)


def _optimization_reward(reward: float | Sequence[float]) -> float:
    if isinstance(reward, Sequence) and not isinstance(reward, (str, bytes)):
        return float(sum(reward))
    return float(reward)


def _add_penalty(
    reward: float | Sequence[float],
    loss_mask: Sequence[int],
    penalty: float,
) -> float | list[float]:
    if not isinstance(reward, Sequence) or isinstance(reward, (str, bytes)):
        return float(reward) + penalty
    shaped = [float(value) for value in reward]
    if penalty == 0:
        return shaped
    active_positions = [position for position, active in enumerate(loss_mask) if active]
    if not active_positions:
        raise ValueError("cannot apply a trajectory penalty without a trainable response token")
    shaped[active_positions[-1]] += penalty
    return shaped


def _metric_key(value: str | None) -> str:
    normalized = "missing" if value is None else str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_") or "missing"


def _empty_components() -> RewardShapingComponents:
    return {"loop": 0.0, "non_termination": 0.0, "successful_length": 0.0}


def infer_stop_reason(response_ids: Sequence[int], eos_token_id: int | None, max_generate_length: int) -> str:
    """Infer normalized stop metadata for adapters that omit backend finish reasons."""
    if response_ids and eos_token_id is not None and response_ids[-1] == eos_token_id:
        return "stop"
    if len(response_ids) >= max_generate_length:
        return "length"
    return "stop"


def _trajectory_groups(output: GeneratorOutput, batch_size: int) -> list[list[int]]:
    is_last_step = output.get("is_last_step")
    if is_last_step is None:
        return [[index] for index in range(batch_size)]
    if len(is_last_step) != batch_size:
        raise ValueError("is_last_step must have one entry per generated row")

    groups: list[list[int]] = []
    current: list[int] = []
    for index, last in enumerate(is_last_step):
        current.append(index)
        if last:
            groups.append(current)
            current = []
    if current:
        raise ValueError("step-wise generator output must mark the final row of every trajectory")
    return groups


def refresh_trajectory_reward_shaping_metrics(output: GeneratorOutput) -> None:
    """Recompute shaping metrics after concatenation, filtering, or replacement."""
    components = output.get("reward_shaping_components")
    if components is None:
        return

    batch_size = len(output["response_ids"])
    if len(components) != batch_size:
        raise ValueError("reward shaping components must have one entry per generated row")
    outcomes = output.get("unshaped_rewards")
    spans = output.get("reward_shaping_loop_spans")
    if outcomes is None or spans is None or len(outcomes) != batch_size or len(spans) != batch_size:
        raise ValueError("shaped outputs must retain outcomes and loop spans for every generated row")

    stop_reasons = output.get("stop_reasons")
    if stop_reasons is None:
        stop_reasons = [None] * batch_size
    groups = _trajectory_groups(output, batch_size)
    response_lengths = [sum(bool(value) for value in loss_mask) for loss_mask in output["loss_masks"]]
    trajectory_components = [components[group[-1]] for group in groups]
    shaped_totals = [_optimization_reward(output["rewards"][group[-1]]) for group in groups]
    penalties = [sum(values.values()) for values in trajectory_components]
    trajectory_lengths = [sum(response_lengths[index] for index in group) for group in groups]
    trajectory_stops = [stop_reasons[group[-1]] for group in groups]
    trajectory_outcomes = [float(outcomes[group[-1]]) for group in groups]

    metrics = output.get("rollout_metrics") or {}
    for key in [key for key in metrics if key.startswith(f"{SHAPING_METRIC_PREFIX}/")]:
        del metrics[key]
    metrics.update(
        {
            f"{SHAPING_METRIC_PREFIX}/outcome_reward_mean": float(np.mean(trajectory_outcomes)),
            f"{SHAPING_METRIC_PREFIX}/optimization_reward_before_mean": float(
                np.mean([shaped - penalty for shaped, penalty in zip(shaped_totals, penalties)])
            ),
            f"{SHAPING_METRIC_PREFIX}/shaped_reward_mean": float(np.mean(shaped_totals)),
            f"{SHAPING_METRIC_PREFIX}/penalty_mean": float(np.mean(penalties)),
            f"{SHAPING_METRIC_PREFIX}/loop_penalty_mean": float(
                np.mean([values["loop"] for values in trajectory_components])
            ),
            f"{SHAPING_METRIC_PREFIX}/non_termination_penalty_mean": float(
                np.mean([values["non_termination"] for values in trajectory_components])
            ),
            f"{SHAPING_METRIC_PREFIX}/successful_length_penalty_mean": float(
                np.mean([values["successful_length"] for values in trajectory_components])
            ),
            f"{SHAPING_METRIC_PREFIX}/loop_incidence": float(
                np.mean([any(spans[index] for index in group) for group in groups])
            ),
            f"{SHAPING_METRIC_PREFIX}/non_termination_incidence": float(
                np.mean([values["non_termination"] < 0 for values in trajectory_components])
            ),
            f"{SHAPING_METRIC_PREFIX}/successful_length_incidence": float(
                np.mean([values["successful_length"] < 0 for values in trajectory_components])
            ),
            f"{SHAPING_METRIC_PREFIX}/response_tokens_mean": float(np.mean(trajectory_lengths)),
            f"{SHAPING_METRIC_PREFIX}/response_tokens_max": float(max(trajectory_lengths, default=0)),
        }
    )
    for stop_reason in trajectory_stops:
        key = f"{SHAPING_METRIC_PREFIX}/stop_reason/{_metric_key(stop_reason)}"
        metrics[key] = metrics.get(key, 0.0) + 1.0
    output["rollout_metrics"] = metrics


def shape_trajectory_rewards(output: GeneratorOutput, raw_config: Mapping[str, Any] | None) -> None:
    """Apply generator-independent additive penalties to normalized trajectories.

    Raw task outcomes are copied to ``unshaped_rewards`` before any shared
    shaping. Existing generator-specific optimization shaping remains intact;
    these components are additive to the optimization reward and cannot alter
    pass-rate or verifier-accuracy metrics that consume the raw channel.
    """
    config = parse_trajectory_reward_shaping_config(raw_config)
    if not config.enabled:
        return

    response_ids = output["response_ids"]
    loss_masks = output["loss_masks"]
    rewards = output["rewards"]
    batch_size = len(response_ids)
    if not (len(loss_masks) == len(rewards) == batch_size):
        raise ValueError("response IDs, loss masks, and rewards must have the same batch size")

    stop_reasons = output.get("stop_reasons")
    if stop_reasons is None:
        stop_reasons = [None] * batch_size
    if len(stop_reasons) != batch_size:
        raise ValueError("stop reasons must have one entry per trajectory")

    existing_outcomes = output.get("unshaped_rewards")
    if existing_outcomes is not None and len(existing_outcomes) != batch_size:
        raise ValueError("unshaped rewards must have one entry per trajectory")
    outcomes = (
        [float(value) for value in existing_outcomes]
        if existing_outcomes is not None
        else [_outcome_reward(reward) for reward in rewards]
    )
    output["unshaped_rewards"] = outcomes

    excluded = output.get("exclude_from_baseline")
    if excluded is None:
        excluded = [False] * batch_size
    if len(excluded) != batch_size:
        raise ValueError("exclude_from_baseline must have one entry per trajectory")

    components = [_empty_components() for _ in range(batch_size)]
    all_loop_spans: list[list[RewardShapingLoopSpan]] = []
    repeated_occurrences: list[int] = []
    shaped_rewards = [list(reward) if isinstance(reward, list) else float(reward) for reward in rewards]
    response_lengths = [sum(bool(value) for value in loss_mask) for loss_mask in loss_masks]
    accepted_stops = set(config.non_termination.accepted_stop_reasons)
    for response, loss_mask in zip(response_ids, loss_masks):
        loop_spans, occurrences = _repeated_spans(response, loss_mask, config.loop)
        all_loop_spans.append(loop_spans)
        repeated_occurrences.append(occurrences)

    groups = _trajectory_groups(output, batch_size)
    for group in groups:
        final_index = group[-1]
        trajectory_length = sum(response_lengths[index] for index in group)
        sample_components = _empty_components()

        if not excluded[final_index]:
            sample_components["loop"] = -min(
                config.loop.max_penalty,
                sum(repeated_occurrences[index] for index in group) * config.loop.penalty_per_occurrence,
            )
            normalized_stop = (
                None if stop_reasons[final_index] is None else str(stop_reasons[final_index]).strip().lower()
            )
            if normalized_stop not in accepted_stops:
                sample_components["non_termination"] = -config.non_termination.penalty
            if outcomes[final_index] > 0:
                charged_tokens = max(0, trajectory_length - config.successful_length.free_tokens)
                sample_components["successful_length"] = -min(
                    config.successful_length.max_penalty,
                    charged_tokens * config.successful_length.penalty_per_token,
                )

        penalty = sum(sample_components.values())
        shaped_rewards[final_index] = _add_penalty(rewards[final_index], loss_masks[final_index], penalty)
        components[final_index] = sample_components

    output["rewards"] = shaped_rewards
    output["reward_shaping_components"] = components
    output["reward_shaping_loop_spans"] = all_loop_spans
    output["reward_shaping_versions"] = [config.schema_version] * batch_size
    refresh_trajectory_reward_shaping_metrics(output)
