"""Capture named GenRM preparation; core Judge grades bounded cohorts."""

import dataclasses
import hashlib
import json
import math
import time

from harbor_config.errors import error_category
from verifyit.execution.worker import call_bounded
from verifyit.grade import Aggregation, InvalidTask, Reward, Status, aggregate_rewards, finalize_preparation_failure
from verifyit.preparation.errors import InvalidPreparation, PreparationError, PreparationFailure

from skyrl_gym.envs.nemotron_ultra.genrm import collect_genrm_comparisons
from skyrl_gym.envs.nemotron_ultra.genrm_utils import (
    apply_length_bonuses,
    extract_from_response_obj,
    generate_comparison_pairs,
)
from skyrl_gym.envs.nemotron_ultra.judge import OpenAIJudge


@dataclasses.dataclass(frozen=True)
class CapturedGenRM:
    history: list
    responses: list
    principle: str
    raw_assistants: list | None
    cohort_evidence: dict | None


@dataclasses.dataclass(frozen=True)
class PreparedGenRM:
    capture: CapturedGenRM
    policies: dict[str, str]
    pairs: list[tuple[int, int]]
    lengths: list[dict[str, int]]


@dataclasses.dataclass(frozen=True)
class GenRMCohortResult:
    rewards: list[float]
    metrics: dict[str, float]
    verdicts: list[Reward]
    preparation: dict
    protected: dict


@dataclasses.dataclass(frozen=True)
class GenRMCohortFailure:
    failure: PreparationFailure
    protected: dict


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _validate(history, responses, principle, config):
    if not isinstance(history, list) or not history or not isinstance(responses, list) or len(responses) < 2:
        raise InvalidTask("GenRM requires a conversation and a comparison cohort")
    if any(not isinstance(response, dict) for response in responses):
        raise InvalidTask("GenRM response objects must be mappings")
    if not isinstance(principle, str) or not principle.strip():
        raise InvalidTask("GenRM principle must be nonempty")
    for name in (
        "default_score",
        "reasoning_bonus",
        "answer_bonus",
        "top_percentile",
        "group_reasoning_length_penalty_coeff",
        "group_answer_length_penalty_coeff",
        "temperature",
        "top_p",
        "genrm_parse_retry_sleep_seconds",
    ):
        if name in config and not _finite(config[name]):
            raise InvalidTask(f"Invalid finite GenRM configuration: {name}")
    if "top_percentile" in config and not 0 <= config["top_percentile"] <= 1:
        raise InvalidTask("GenRM top percentile must lie in [0, 1]")
    if "genrm_parse_retry_sleep_seconds" in config and config["genrm_parse_retry_sleep_seconds"] < 0:
        raise InvalidTask("GenRM retry sleep must be nonnegative")
    if "group_answer_length_penalty_coeff" not in config:
        raise InvalidTask("Missing GenRM answer penalty policy")
    for name in (
        "max_concurrent_comparisons",
        "max_output_tokens",
        "genrm_parse_retries",
    ):
        if name in config and (
            type(config[name]) is not int or config[name] < (0 if name == "genrm_parse_retries" else 1)
        ):
            raise InvalidTask(f"Invalid GenRM integer configuration: {name}")


def _validate_scores(rewards, metrics, size):
    if not isinstance(rewards, list) or len(rewards) != size or not all(_finite(value) for value in rewards):
        raise ValueError("GenRM cohort reward vector is incomplete or nonfinite")
    if not isinstance(metrics, dict) or not all(
        isinstance(key, str) and _finite(value) for key, value in metrics.items()
    ):
        raise ValueError("GenRM metrics are nonfinite")
    expected = {f"verification_reward_{index}" for index in range(size)}
    if {key for key in metrics if key.startswith("verification_reward_")} != expected or any(
        not 0 <= metrics[key] <= 1 for key in expected
    ):
        raise ValueError("GenRM normalized verification vector is incomplete or invalid")


def capture_genrm_cohort(history, responses, principle, raw_assistants=None, cohort_evidence=None):
    """Snapshot supported JSON producer evidence before any field selection."""
    try:
        raw = json.loads(json.dumps([history, responses, principle, raw_assistants, cohort_evidence], allow_nan=False))
    except (ValueError, TypeError, RecursionError) as error:
        raise InvalidTask("GenRM producer evidence must be finite JSON") from error
    return CapturedGenRM(*raw)


def prepare_genrm_cohort(capture, config):
    """Validate the complete producer cohort and select explicit source policies."""
    from verifyit.modes.grade_json_schema import grade_json_schema_candidate

    _validate(capture.history, capture.responses, capture.principle, config)
    text_item = {"type": "object", "properties": {"type": {"type": "string"}, "text": {"type": "string"}}}
    response_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "output": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "summary": {"type": "array", "items": text_item},
                            "content": {"type": "array", "items": text_item},
                        },
                    },
                }
            },
        },
    }
    history_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "required": ["role", "content"],
            "properties": {"role": {"type": "string", "minLength": 1}, "content": {"type": ["string", "array"]}},
        },
    }
    if any(
        grade_json_schema_candidate(schema, value).reward != 1
        for schema, value in (
            (history_schema, capture.history),
            (response_schema, capture.responses),
        )
    ):
        raise InvalidTask("GenRM producer topology is unsupported")
    if capture.raw_assistants is not None and (
        not isinstance(capture.raw_assistants, list)
        or len(capture.raw_assistants) != len(capture.responses)
        or any(not isinstance(message, dict) for message in capture.raw_assistants)
    ):
        raise InvalidTask("GenRM raw assistant cohort is incomplete")
    policies = {
        "response": "source_response_text_missing_to_none_v1",
        "score_json": config.get("verifyit_score_json_policy", "strict_single_object_v1"),
        "peers": config.get("verifyit_peer_policy", "source_valid_peers_v1"),
        "pairing": "source_circular_neighbors_v1",
        "shaping": "source_length_bonuses_v1",
    }
    if policies["score_json"] not in ("strict_single_object_v1", "source_last_object_v1") or policies["peers"] not in (
        "source_valid_peers_v1",
        "require_all_peers_v1",
    ):
        raise InvalidTask("GenRM preparation policy is unsupported")
    lengths = []
    for response in capture.responses:
        reasoning, answer = extract_from_response_obj(response)
        lengths.append({"reasoning": len(reasoning.strip()), "answer": len(answer.strip())})
    return PreparedGenRM(capture, policies, generate_comparison_pairs("circular", len(capture.responses)), lengths)


def _grade_genrm_cohort(
    conversation_history, response_objects, principle, judge, config, raw_assistants, cohort_evidence
):
    captured = prepared = None
    provider_receipts = []
    stage = "structural_capture"
    try:
        from verifyit.modes.grade_judge import grade_paired_ordinal

        captured = capture_genrm_cohort(
            conversation_history, response_objects, principle, raw_assistants, cohort_evidence
        )
        stage = "reference_preparation"
        prepared = prepare_genrm_cohort(captured, config)
        stage = "provider_transport"
        pairs, comparisons = collect_genrm_comparisons(
            conversation_history=captured.history,
            response_objects=captured.responses,
            principle=captured.principle,
            judge=OpenAIJudge(**{**judge, "strict_completion": True}),
            config={**config, "verifyit_strict_json": prepared.policies["score_json"] == "strict_single_object_v1"},
            provider_receipts=provider_receipts,
        )
        if pairs != prepared.pairs:
            raise RuntimeError("GenRM provider cohort changed the prepared comparison pairs")
        ratings = [
            {"left": left, "right": right, "score_left": first, "score_right": second, "ranking": ranking}
            for (left, right), (first, second, ranking) in zip(pairs, comparisons, strict=True)
        ]
        stage = "core_grading"
        verdicts = grade_paired_ordinal(len(captured.responses), pairs, ratings)
        rewards = [verdict.detail["raw_score"] for verdict in verdicts]
        stage = "training_shaping"
        unshaped = list(rewards)
        if any(
            float(config.get(name, default)) > 0
            for name, default in (
                ("reasoning_bonus", 0.5),
                ("answer_bonus", 0.5),
                ("group_reasoning_length_penalty_coeff", 0.1),
                ("group_answer_length_penalty_coeff", 0.0),
            )
        ):
            rewards, _ = apply_length_bonuses(
                scores=rewards,
                response_objs=captured.responses,
                reasoning_bonus=float(config.get("reasoning_bonus", 0.5)),
                answer_bonus=float(config.get("answer_bonus", 0.5)),
                top_percentile=float(config.get("top_percentile", 0.2)),
                group_reasoning_length_penalty_coeff=float(config.get("group_reasoning_length_penalty_coeff", 0.1)),
                group_answer_length_penalty_coeff=float(config["group_answer_length_penalty_coeff"]),
            )
        metrics = dict(verdicts[0].detail["comparison_metrics"])
        metrics.update({f"verification_reward_{index}": verdict.reward for index, verdict in enumerate(verdicts)})
        _validate_scores(rewards, metrics, len(captured.responses))
        checked = aggregate_rewards(verdicts, expected_total=len(captured.responses), policy=Aggregation.MEAN)
        if checked.status is not Status.SCORED:
            raise RuntimeError("GenRM core cohort is unscored")
        stage = "public_receipt"
        public = {
            "policies": prepared.policies,
            "input_sha256": hashlib.sha256(
                json.dumps(dataclasses.asdict(captured), sort_keys=True).encode()
            ).hexdigest(),
            "comparison_count": len(pairs),
            "shaping_options": {
                name: config.get(name, default)
                for name, default in (
                    ("reasoning_bonus", 0.5),
                    ("answer_bonus", 0.5),
                    ("top_percentile", 0.2),
                    ("group_reasoning_length_penalty_coeff", 0.1),
                    ("group_answer_length_penalty_coeff", 0.0),
                )
            },
        }
        protected = {
            "capture": captured,
            "prepared": prepared,
            "provider_attempts": provider_receipts,
            "ratings": ratings,
            "core_verdicts": verdicts,
            "unshaped_training_rewards": unshaped,
        }
        return GenRMCohortResult(rewards, metrics, verdicts, public, protected)
    except Exception as error:
        status = Status.INVALID_TASK if isinstance(error, InvalidTask) else Status.INFRA_ERROR
        failure = PreparationFailure(
            status,
            error_category(type(error).__name__),
            type(error).__name__,
            "GenRM cohort verification failed",
            stage,
        )
        return GenRMCohortFailure(
            failure, {"capture": captured, "prepared": prepared, "provider_attempts": provider_receipts}
        )


def grade_genrm_cohort(
    *,
    conversation_history,
    response_objects,
    principle,
    judge,
    config,
    raw_assistants=None,
    cohort_evidence=None,
    started_at=None,
):
    """Bound capture, preparation, provider transport and grading once per cohort."""
    started = time.monotonic() if started_at is None else started_at
    try:
        if not _finite(started) or started > time.monotonic():
            raise InvalidTask("Invalid GenRM cohort start time")
        if not isinstance(config, dict):
            raise InvalidTask("GenRM configuration must be a mapping")
        timeout = config.get("verifyit_timeout_seconds", 120.0)
        if not _finite(timeout) or not 0 < timeout <= 3600:
            raise InvalidTask("Invalid GenRM total deadline")
        judge_options = dataclasses.asdict(judge)
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("GenRM total cohort deadline exceeded")
        result = call_bounded(
            _grade_genrm_cohort,
            conversation_history,
            response_objects,
            principle,
            judge_options,
            config,
            raw_assistants,
            cohort_evidence,
            timeout=float(remaining),
        )
        if isinstance(result, GenRMCohortFailure):
            exception = InvalidPreparation if result.failure.status is Status.INVALID_TASK else PreparationError
            error = exception(result.failure, finalize_preparation_failure(**dataclasses.asdict(result.failure)))
            error.protected = result.protected
            raise error
        if time.monotonic() - started >= timeout:
            raise TimeoutError("GenRM total cohort deadline exceeded")
        return result
    except PreparationError:
        raise
    except Exception as error:
        status = Status.INVALID_TASK if isinstance(error, InvalidTask) else Status.INFRA_ERROR
        failure = PreparationFailure(
            status,
            error_category(type(error).__name__),
            type(error).__name__,
            "GenRM cohort verification failed",
            "bounded_cohort",
        )
        exception = InvalidPreparation if status is Status.INVALID_TASK else PreparationError
        raise exception(failure, finalize_preparation_failure(**dataclasses.asdict(failure))) from error


def grade_genrm_verifyit(*, conversation_history, response_objects, principle, judge, config):
    """Preserve the existing public rewards/finite-metrics tuple API."""
    result = grade_genrm_cohort(
        conversation_history=conversation_history,
        response_objects=response_objects,
        principle=principle,
        judge=judge,
        config=config,
    )
    return result.rewards, result.metrics
