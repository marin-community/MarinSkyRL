"""Bound judge transport; core Judge grades, and framework shaping follows."""

import dataclasses
import json
import math
import os
from pathlib import Path
import shlex
import sys
import tempfile

from verifyit.grade import Aggregation, InvalidTask, Reward, Status, aggregate_rewards, run
from verifyit.spec import ScriptSpec, render_spec

from skyrl_gym.envs.nemotron_ultra.genrm import collect_genrm_comparisons
from skyrl_gym.envs.nemotron_ultra.genrm_utils import apply_length_bonuses
from skyrl_gym.envs.nemotron_ultra.judge import OpenAIJudge


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


def grade_genrm_verifyit(*, conversation_history, response_objects, principle, judge, config):
    _validate(conversation_history, response_objects, principle, config)
    timeout = config.get("verifyit_timeout_seconds", 120.0)
    if not _finite(timeout) or not 0 < timeout <= 3600:
        raise InvalidTask("Invalid GenRM total deadline")
    with tempfile.TemporaryDirectory(prefix="skyrl-genrm-") as directory:
        root = Path(directory)
        payload = {
            "conversation_history": conversation_history,
            "response_objects": response_objects,
            "principle": principle,
            "judge": dataclasses.asdict(judge),
            "config": {
                **config,
                "verifyit_enabled": False,
                "verifyit_strict_json": True,
            },
        }
        (root / "input.json").write_text(json.dumps(payload, allow_nan=False))
        (root / "check.sh").write_text(
            "set -eu\nexec "
            + shlex.quote(sys.executable)
            + " -m skyrl_gym.envs.nemotron_ultra.genrm_verifyit "
            + shlex.quote(str(root / "input.json"))
            + "\n"
        )
        spec = root / "verifier.toml"
        spec.write_text(render_spec(ScriptSpec(path="check.sh", timeout=float(timeout), verdict_file="cohort.json")))
        result = run(spec, root)
        if result.status is Status.INVALID_TASK:
            raise InvalidTask(result.detail.get("error", "Invalid GenRM task"))
        if result.status is not Status.SCORED:
            raise RuntimeError(f"GenRM cohort verification failed: {result.status}")
        rewards, metrics = (
            result.detail["cohort_rewards"],
            result.detail["cohort_metrics"],
        )
        records = result.detail.get("core_verdicts")
        try:
            verdicts = [Reward(record["reward"], Status(record["status"]), record["detail"]) for record in records]
        except (TypeError, ValueError, KeyError) as error:
            raise RuntimeError("GenRM core verdict vector is malformed") from error
        checked = aggregate_rewards(verdicts, expected_total=len(response_objects), policy=Aggregation.MEAN)
        if checked.status is not Status.SCORED or len(verdicts) != len(response_objects):
            raise RuntimeError("GenRM core verdict vector is incomplete or unscored")
        if any(metrics.get(f"verification_reward_{index}") != verdict.reward for index, verdict in enumerate(verdicts)):
            raise RuntimeError("GenRM verification projection differs from core verdict")
        _validate_scores(rewards, metrics, len(response_objects))
        return rewards, metrics


def _main():
    data = json.loads(Path(sys.argv[1]).read_text())
    _validate(
        data["conversation_history"],
        data["response_objects"],
        data["principle"],
        data["config"],
    )
    pairs, comparisons = collect_genrm_comparisons(
        conversation_history=data["conversation_history"],
        response_objects=data["response_objects"],
        principle=data["principle"],
        judge=OpenAIJudge(**{**data["judge"], "strict_completion": True}),
        config=data["config"],
    )
    from verifyit.modes.grade_judge import grade_paired_ordinal

    ratings = [
        {"left": left, "right": right, "score_left": first, "score_right": second, "ranking": ranking}
        for (left, right), (first, second, ranking) in zip(pairs, comparisons, strict=True)
    ]
    verdicts = grade_paired_ordinal(len(data["response_objects"]), pairs, ratings)
    # Only successfully graded cohorts reach native framework reward shaping.
    rewards = [verdict.detail["raw_score"] for verdict in verdicts]
    config = data["config"]
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
            response_objs=data["response_objects"],
            reasoning_bonus=float(config.get("reasoning_bonus", 0.5)),
            answer_bonus=float(config.get("answer_bonus", 0.5)),
            top_percentile=float(config.get("top_percentile", 0.2)),
            group_reasoning_length_penalty_coeff=float(config.get("group_reasoning_length_penalty_coeff", 0.1)),
            group_answer_length_penalty_coeff=float(config["group_answer_length_penalty_coeff"]),
        )
    metrics = dict(verdicts[0].detail["comparison_metrics"])
    metrics.update({f"verification_reward_{index}": verdict.reward for index, verdict in enumerate(verdicts)})
    _validate_scores(rewards, metrics, len(data["response_objects"]))
    # The scalar is only ScriptSpec's execution envelope. Training consumes the vector.
    result = {
        "schema_version": 1,
        "status": "scored",
        "reward": 0.0,
        "detail": {
            "core_verdicts": [dataclasses.asdict(verdict) for verdict in verdicts],
            "cohort_rewards": rewards,
            "cohort_metrics": metrics,
            "comparison_count": len(data["response_objects"]),
            "runtime": "core_paired_judge",
        },
    }
    (Path(os.environ["VERIFYIT_LOGS_DIR"]) / "cohort.json").write_text(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    try:
        _main()
    except InvalidTask as error:
        (Path(os.environ["VERIFYIT_LOGS_DIR"]) / "cohort.json").write_text(
            json.dumps({"schema_version": 1, "status": "invalid_task", "reward": 0.0, "detail": {"error": str(error)}})
        )
