"""Typed, bounded preparation of source judge profiles for core composition."""

from __future__ import annotations

import dataclasses
import json
import hashlib
import math
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

from harbor_config.errors import error_category
from verifyit.execution.worker import call_bounded
from verifyit.grade import Aggregation, InvalidTask, Reward, Status, aggregate_rewards, finalize_preparation_failure
from verifyit.modes.grade_judge import JudgeConnection, grade_judge_candidate, validate_judge_spec
from verifyit.spec import JudgeSpec

from skyrl_gym.envs.nemotron_ultra.answer_extraction import final_answer_text, last_boxed_answer
from skyrl_gym.envs.nemotron_ultra.jailbreak import _configuration
from skyrl_gym.envs.nemotron_ultra.judge_verifiers import (
    _MULTICHALLENGE_PROMPT,
    _MULTICHALLENGE_SYSTEM,
    _normalize_answer,
)

POLICY_DEFAULTS = {
    "response": "nemotron_final_answer_v1",
    "abstention": "nemotron_articles_punctuation_case_v1",
    "rubric": "nemotron_yes_unless_no_v1",
    "labels": "source_alias_lines_no_contradictions_v1",
    "composition": "source_v1",
}
POLICY_OPTIONS = {
    "response": {"nemotron_final_answer_v1", "literal_v1"},
    "abstention": {"nemotron_articles_punctuation_case_v1", "literal_v1"},
    "rubric": {"nemotron_yes_unless_no_v1", "binary_only_v1"},
    "labels": {"source_alias_lines_no_contradictions_v1", "bracketed_only_no_contradictions_v1"},
    "composition": {"source_v1", "mean_v1", "product_v1"},
}
SELECTED_POLICIES = {
    "engagement_with_disclaimer",
    "hard_refusal_no_redirection",
    "hard_refusal_with_helplines",
    "refusal_with_explanation",
}


@dataclass(frozen=True)
class ProfileInputs:
    response: str
    record: dict[str, Any]
    kind: str
    policies: dict[str, str]


@dataclass(frozen=True)
class PreparedProfile:
    inputs: ProfileInputs
    policies: dict[str, str]
    candidate: str
    gate_candidate: str | None
    components: tuple[tuple[str, JudgeSpec], ...]
    aggregation: Aggregation
    source_configuration: dict[str, Any]


def _text(value: Any, name: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise InvalidTask(f"Invalid {name}")
    return value


def _literal(value: str) -> str:
    return value.replace("{", "{{").replace("}", "}}")


def structure_profile(response: str, record: dict[str, Any], kind: str, policies: Mapping[str, str]) -> ProfileInputs:
    """Snapshot JSON task evidence and response without applying source policies."""
    _text(response, "response", nonempty=False)
    if not isinstance(record, dict) or not isinstance(policies, Mapping):
        raise InvalidTask("Judge record and policies must be objects")
    snapshot = json.loads(json.dumps({"record": record, "policies": dict(policies)}, allow_nan=False))
    return ProfileInputs(response, snapshot["record"], kind, snapshot["policies"])


def prepare_profile(inputs: ProfileInputs, judge: dict[str, Any]) -> PreparedProfile:
    """Prepare and validate every trusted component before provider execution."""
    policies = POLICY_DEFAULTS | inputs.policies
    if set(policies) != set(POLICY_DEFAULTS) or any(
        not isinstance(value, str) or value not in POLICY_OPTIONS[name] for name, value in policies.items()
    ):
        raise InvalidTask("Unsupported judge preparation policy")
    record = inputs.record
    candidate = (
        final_answer_text(inputs.response) if policies["response"] == "nemotron_final_answer_v1" else inputs.response
    )
    components = []
    gate_candidate = None
    source_configuration = {}

    def component(name, reference, question, template, labels, *, system="", scan="literal", gate=False):
        spec = JudgeSpec(
            rubric="labels",
            references=(reference,),
            question=question,
            prompt_template=template,
            system_prompt=system,
            label_scores=labels,
            strip_reasoning_blocks=True,
            label_scan=scan,
            request_timeout=judge["timeout_seconds"],
            model=_text(judge["model"], "judge model"),
            max_completion_tokens=8192,
            reasoning_effort=judge.get("reasoning_effort") or "",
            exact_gate_answers=("idk",) if gate else (),
            exact_gate_label="C" if gate else "",
        )
        validate_judge_spec(spec)
        components.append((name, spec))

    if inputs.kind == "abstention":
        question = _text(record.get("question"), "question")
        expected = _text(record.get("answer"), "answer")
        if policies["response"] == "nemotron_final_answer_v1":
            candidate = last_boxed_answer(candidate) or candidate
        gate_candidate = (
            _normalize_answer(candidate)
            if policies["abstention"] == "nemotron_articles_punctuation_case_v1"
            else candidate
        )
        template = (
            files(__package__)
            .joinpath("abstention_prompt.txt")
            .read_text()
            .replace("{target}", "{reference}")
            .replace("{predicted_answer}", "{candidate}")
        )
        component("abstention", expected, question, template, {"A": 1.0, "B": 0.0, "C": 0.5}, scan="lines", gate=True)
        aggregation = Aggregation.MEAN
    elif inputs.kind == "multichallenge":
        metadata = record.get("metadata", {})
        if not isinstance(metadata, dict):
            raise InvalidTask("Malformed rubric metadata")
        rubric = record.get("rubric") or metadata.get("rubric") or []
        if not isinstance(rubric, list) or not rubric:
            raise InvalidTask("Rubric must contain meaningful criteria")
        context = _text(record.get("context", ""), "context", nonempty=False)
        template = (
            _MULTICHALLENGE_PROMPT.replace("{response}", "{candidate}")
            .replace("{pass_criteria}", "{reference}")
            .replace("{context}", _literal(context))
        )
        for index, item in enumerate(rubric):
            if not isinstance(item, dict):
                raise InvalidTask("Malformed rubric criterion")
            question = _text(item.get("question"), "criterion")
            expected = _text(item.get("pass_criteria", "YES"), "pass criteria")
            if policies["rubric"] == "binary_only_v1" and expected.upper() not in {"YES", "NO"}:
                raise InvalidTask("Binary rubric requires YES or NO")
            correct = "NO" if expected.upper() == "NO" else "YES"
            labels = {f"[[{label}]]": float(label == correct) for label in ("YES", "NO")}
            if policies["labels"] == "source_alias_lines_no_contradictions_v1":
                labels.update({label: float(label == correct) for label in ("YES", "NO")})
            component(str(index), expected, question, template, labels, system=_MULTICHALLENGE_SYSTEM, scan="lines")
        aggregation = Aggregation.MEAN
    elif inputs.kind == "jailbreak":
        config = _configuration()
        policy = record.get("response_policy_mapped")
        if not isinstance(policy, str) or policy not in SELECTED_POLICIES:
            raise InvalidTask("Unsupported response policy")
        question = _text(record.get("adversarial_prompt"), "adversarial prompt")
        policy_config = config["policy_verifier_map"][policy]
        names = policy_config["verifiers"]
        if not isinstance(names, list) or not names or policy_config.get("reward_combination", "product") != "product":
            raise InvalidTask("Malformed selected policy composition")
        source_configuration = {"policy": policy, "composition": policy_config, "verifiers": {}}
        for name in names:
            verifier = config[name]
            source_configuration["verifiers"][name] = verifier
            positive = _text(verifier["labels"]["positive"], "positive label")
            negative = _text(verifier["labels"]["negative"], "negative label")
            if positive == negative:
                raise InvalidTask("Policy labels collide")
            labels = {
                positive: verifier["rewards"].get("positive", 1.0),
                negative: verifier["rewards"].get("negative", 0.0),
            }
            prompt = _text(verifier["prompt_template"], "policy template")
            if prompt.count("{model_response}") != 1:
                raise InvalidTask("Malformed policy prompt")
            prefix, suffix = prompt.split("{model_response}")
            prefix = prefix.format(adversarial_prompt=question)
            suffix = suffix.replace("{adversarial_prompt}", "{question}")
            component(name, prefix, question, "{reference}{candidate}" + suffix, labels)
        aggregation = Aggregation.PRODUCT
    else:
        raise InvalidTask("Unknown judge profile")
    if policies["composition"] != "source_v1":
        aggregation = Aggregation.MEAN if policies["composition"] == "mean_v1" else Aggregation.PRODUCT
    return PreparedProfile(
        inputs, policies, candidate, gate_candidate, tuple(components), aggregation, source_configuration
    )


def _failure(error: Exception, status: Status, stage: str) -> Reward:
    return finalize_preparation_failure(
        status=status,
        category=error_category("InvalidTask" if status == Status.INVALID_TASK else type(error).__name__),
        error_type=type(error).__name__,
        message=str(error),
        stage=stage,
    )


def _evaluate(inputs: ProfileInputs, judge: dict[str, Any]) -> Reward:
    started = time.monotonic()
    receipt = {"raw": dataclasses.asdict(inputs), "primitive_calls": []}
    try:
        prepared = prepare_profile(inputs, judge)
        connection = JudgeConnection(
            _text(judge["base_url"], "judge endpoint"),
            os.environ.get(judge.get("api_key_env") or "", judge.get("api_key", "dummy_key")),
        )
    except (InvalidTask, KeyError, TypeError, ValueError) as error:
        verdict = _failure(error, Status.INVALID_TASK, "judge_profile_preparation")
        return dataclasses.replace(verdict, detail=verdict.detail | {"preparation": receipt})
    receipt.update(policies=prepared.policies, prepared=dataclasses.asdict(prepared))
    verdicts = []
    for name, spec in prepared.components:
        remaining = judge["total_timeout_seconds"] - (time.monotonic() - started)
        call = {
            "name": name,
            "candidate": prepared.candidate,
            "gate_candidate": prepared.gate_candidate,
            "spec": dataclasses.asdict(spec),
        }
        receipt["primitive_calls"].append(call)
        try:
            if remaining <= 0:
                raise TimeoutError("Judge profile deadline exhausted")
            spec = dataclasses.replace(spec, request_timeout=min(spec.request_timeout, remaining))
            verdict = grade_judge_candidate(
                spec, prepared.candidate, connection=connection, gate_candidate=prepared.gate_candidate
            )
        except InvalidTask as error:
            verdict = _failure(error, Status.INVALID_TASK, "judge_component")
        except Exception as error:
            verdict = _failure(error, Status.INFRA_ERROR, "judge_provider")
        call["verdict"] = dataclasses.asdict(verdict)
        verdicts.append(verdict)
        if verdict.status != Status.SCORED:
            break
    result = aggregate_rewards(verdicts, expected_total=len(prepared.components), policy=prepared.aggregation)
    feedback = {
        "verifier_rewards": {name: verdict.reward for (name, _), verdict in zip(prepared.components, verdicts)},
        "verifier_labels": {
            name: verdict.detail.get("verdict") for (name, _), verdict in zip(prepared.components, verdicts)
        },
    }
    return dataclasses.replace(result, detail=result.detail | {"source_feedback": feedback, "preparation": receipt})


def grade_judge_profile_verifyit(
    text: str,
    record: dict[str, Any],
    judge,
    *,
    kind: str,
    timeout_seconds: float = 120.0,
    policies: dict[str, str] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Run the prepared profile under one hard process-group deadline."""
    try:
        inputs = structure_profile(text, record, kind, {} if policies is None else policies)
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= 3600
        ):
            raise InvalidTask("Invalid judge deadline")
        config = dataclasses.asdict(judge) | {"total_timeout_seconds": float(timeout_seconds)}
    except (InvalidTask, TypeError, ValueError) as error:
        result = _failure(error, Status.INVALID_TASK, "judge_profile_structure")
    else:
        try:
            result = call_bounded(_evaluate, inputs, config, timeout=timeout_seconds)
        except Exception as error:
            result = _failure(error, Status.INFRA_ERROR, "judge_profile_execution")
            result = dataclasses.replace(
                result, detail=result.detail | {"preparation": {"raw": dataclasses.asdict(inputs)}}
            )
    # Trusted references and provider prose remain in the bounded transport receipt.
    receipt = result.detail.get("preparation", {})
    raw_digest = hashlib.sha256(
        json.dumps(receipt.get("raw", {}), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    public_verdict = {
        "reward": result.reward,
        "status": result.status.value,
        "detail": {key: result.detail[key] for key in ("passed", "total", "missing") if key in result.detail},
    }
    public_preparation = {
        "policies": receipt.get("policies", POLICY_DEFAULTS),
        "raw_sha256": raw_digest,
    }
    if "prepared" in receipt:
        public_preparation["prepared_sha256"] = hashlib.sha256(
            json.dumps(receipt["prepared"], sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        public_preparation["aggregation"] = receipt["prepared"]["aggregation"]
    details = {
        "verifyit_status": result.status.value,
        "verifyit_verdict": public_verdict,
        "preparation": public_preparation,
        **result.detail.get("source_feedback", {}),
    }
    if result.status != Status.SCORED:
        cause = result.detail.get("cause", result.detail)
        details.update(
            error_type="schema_error" if result.status == Status.INVALID_TASK else "verification_error",
            cause_error_type=cause.get("error_type", "VerificationFailure"),
            error_category=cause.get("category"),
            preparation_stage=cause.get("stage"),
        )
    return result.reward, details
