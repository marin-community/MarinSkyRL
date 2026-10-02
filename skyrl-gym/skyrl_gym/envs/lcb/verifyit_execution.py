"""LiveCodeBench sandbox transport using core comparison and aggregation."""

from __future__ import annotations

import inspect
import hashlib
import json
import logging
import math
import time
import traceback
import uuid
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


logger = logging.getLogger(__name__)
CLEANUP_TIMEOUT_SECONDS = 10.0


class CodePolicy(StrEnum):
    PREPARED = "lcb_prepared_code_v1"
    LCB = "lcb_source_v1"
    NEMOTRON = "nemotron_code_source_v1"


@dataclass(frozen=True)
class CodeInputs:
    reference: Any
    response: str
    reasoning_content: Any = None


@dataclass(frozen=True)
class PreparedCode:
    inputs: CodeInputs
    tests: list
    code: str
    policy: str = "lcb_source_v1"

    def provenance(self) -> dict:
        return {
            "policy": self.policy,
            "reference_sha256": hashlib.sha256(repr(self.inputs.reference).encode()).hexdigest(),
            "raw_response": self.inputs.response,
            "raw_reasoning_content": self.inputs.reasoning_content,
            "effective_tests_sha256": hashlib.sha256(json.dumps(self.tests, sort_keys=True).encode()).hexdigest(),
            "effective_code": self.code,
            "extraction": "last_fenced_block_strip",
            "comparison": "schema_const_or_decimal_lines",
            "callable_output": "top_level_tuple_to_list_nested_tuple_rejected",
        }


def capture_code_inputs(reference: Any, response: str, *, reasoning_content: Any = None) -> CodeInputs:
    return CodeInputs(deepcopy(reference), response, deepcopy(reasoning_content))


def code_verification_failure(status: str, detail: dict, *, stage: str):
    from harbor_config.errors import error_category
    from verifyit.grade import Status, finalize_preparation_failure
    from verifyit.preparation.errors import InvalidPreparation, PreparationError, PreparationFailure

    invalid = status == "invalid_task"
    failure = PreparationFailure(
        Status(status),
        error_category("InvalidTask" if invalid else "VerifierRuntimeError"),
        str(detail.get("error_type", "InvalidTask" if invalid else "VerifierRuntimeError")),
        str(detail.get("reason", "Code verification unavailable")),
        stage,
    )
    verdict = finalize_preparation_failure(
        status=failure.status,
        category=failure.category,
        error_type=failure.error_type,
        message=failure.message,
        stage=stage,
    )
    verdict.detail.update(detail)
    return (InvalidPreparation if invalid else PreparationError)(failure, verdict)


def prepare_code(inputs: CodeInputs) -> PreparedCode:
    from skyrl_gym.envs.lcb.livecodebench import normalize_lcb_ground_truth, extract_code_from_model

    try:
        tests = json.loads(normalize_lcb_ground_truth(inputs.reference))
    except (KeyError, TypeError, ValueError) as error:
        raise code_verification_failure(
            "invalid_task",
            {
                "reason": "trusted test preparation failed",
                "error_type": type(error).__name__,
                "preparation": {
                    "policy": "lcb_source_v1",
                    "reference_sha256": hashlib.sha256(repr(inputs.reference).encode()).hexdigest(),
                    "raw_response": inputs.response,
                },
            },
            stage="code_tests",
        ) from error
    return PreparedCode(inputs, tests, extract_code_from_model(inputs.response) or "")


def _wire(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return [type(value).__name__, value]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [_wire(item) for item in value]]
    if isinstance(value, dict):
        return ["dict", [[_wire(key), _wire(item)] for key, item in value.items()]]
    raise ValueError("unsupported output")


def _unwire(value):
    kind, payload = value
    if kind == "tuple":
        raise ValueError("nested tuple result is not a JSON array")
    if kind == "list":
        return [_unwire(item) for item in payload]
    if kind == "dict":
        return {_unwire(key): _unwire(item) for key, item in payload}
    if kind in {"NoneType", "str", "bool", "int", "float"} and type(payload).__name__ == kind:
        return payload
    raise ValueError("malformed output")


def _execute(data: dict) -> dict:
    from verifyit.grade import Aggregation, InvalidTask, aggregate_rewards, scored
    from verifyit.modes.grade_stdio import grade_stdio_candidate
    from verifyit.modes.grade_json_schema import grade_json_schema_candidate
    from verifyit.spec import Compare, StdioSpec

    from skyrl_gym.envs.lcb import livecodebench as runtime

    stdio = StdioSpec(command="", compare=Compare.DECIMAL_LINES)
    tests = data["tests"]
    if not isinstance(tests, list) or not tests:
        return {"status": "invalid_task", "reward": 0.0, "detail": {"reason": "missing_tests"}}
    try:
        reference = json.loads(runtime.postprocess_lcb_sample(tests)["input_output"])
        function = reference.get("fn_name")
        inputs = reference["inputs"]
        expected = [json.loads(value) for value in reference["outputs"]] if function else reference["outputs"]
        if function:
            inputs = [[json.loads(line) for line in value.split("\n")] for value in inputs]
        if function:
            for value in expected:
                grade_json_schema_candidate({"const": value}, value)
        else:
            for value in expected:
                grade_stdio_candidate(stdio, "", value)
        if len(inputs) != len(expected) or not expected:
            raise ValueError("misaligned tests")
    except (InvalidTask, AssertionError, AttributeError, KeyError, TypeError, ValueError) as error:
        return {
            "status": "invalid_task",
            "reward": 0.0,
            "detail": {"reason": "malformed_tests", "error_type": type(error).__name__, "error_message": str(error)},
        }
    if not data["code"]:
        return {
            "status": "scored",
            "reward": 0.0,
            "detail": {"reason": "missing_code", "test_results": [], "total_tests": len(expected)},
        }
    from skyrl_gym.envs.nemotron_ultra.sandbox import SandboxClient

    sandbox = SandboxClient(host=data["host"], port=data["port"])
    session = data["session"]
    timeout = data["timeout"]
    helpers = (
        inspect.getsource(_wire)
        + "\n"
        + "\n".join(inspect.getsource(value) for value in (runtime._TextStdin, runtime.call_method))
    )
    code = (
        runtime.import_string + "\n\n" + data["code"]
        if function
        else runtime.make_function(runtime.clean_if_name(data["code"]))
    )
    # The remote namespace receives no references, verifier logs or trusted file paths.
    setup = (
        "import json, contextlib, io, types, resource\nfrom io import StringIO, BytesIO\nfrom unittest.mock import patch, mock_open\n"
        + helpers
    )
    setup += "\n" + runtime.import_string + "\nprint('VERIFYIT_RUNTIME_READY')\n"
    initialized = sandbox.execute(
        setup,
        language="ipython",
        session_id=session,
        timeout_seconds=data["setup_timeout"],
        max_output_characters=65536,
    )
    if (
        initialized.get("process_status") != "completed"
        or initialized.get("stdout", "").strip() != "VERIFYIT_RUNTIME_READY"
        or initialized.get("output_truncated")
    ):
        return {
            "status": "infra_error",
            "reward": 0.0,
            "detail": {"reason": "sandbox_runtime_setup_failed", "runtime": initialized},
        }
    startup = "_verifyit_namespace = {}\n"
    startup += (
        "try:\n    with contextlib.redirect_stdout(io.StringIO()):\n        exec("
        + repr(code)
        + ", _verifyit_namespace)\n"
    )
    if function:
        startup += "        _verifyit_object = _verifyit_namespace['Solution']() if 'Solution' in _verifyit_namespace else types.SimpleNamespace(**_verifyit_namespace)\n"
        startup += "        _verifyit_method = getattr(_verifyit_object, " + repr(function) + ")\n"
    else:
        startup += "        _verifyit_method = _verifyit_namespace['wrapped_function']\n"
    startup += "    print('VERIFYIT_INIT_READY')\nexcept (Exception, SystemExit):\n    print('VERIFYIT_INIT_CANDIDATE_FAILURE')\n"
    if data["memory_limit"] is not None:
        memory_program = "import resource\n"
        memory_program += (
            "for _kind in (resource.RLIMIT_AS, resource.RLIMIT_DATA):\n    _soft, _hard = resource.getrlimit(_kind)\n    _bound = "
            + str(data["memory_limit"])
            + "\n    _bound = min(_bound, _hard) if _hard != resource.RLIM_INFINITY else _bound\n    _bound = min(_bound, _soft) if _soft != resource.RLIM_INFINITY else _bound\n    resource.setrlimit(_kind, (_bound, _bound))\nprint('VERIFYIT_LIMIT_READY')\n"
        )
        bounded = sandbox.execute(
            memory_program, language="ipython", session_id=session, timeout_seconds=timeout, max_output_characters=65536
        )
        if (
            bounded.get("process_status") != "completed"
            or bounded.get("stderr")
            or bounded.get("stdout", "").strip() != "VERIFYIT_LIMIT_READY"
        ):
            return {
                "status": "infra_error",
                "reward": 0.0,
                "detail": {"reason": "sandbox_limit_setup_failed", "runtime": bounded},
            }
    initial = sandbox.execute(
        startup, language="ipython", session_id=session, timeout_seconds=timeout, max_output_characters=65536
    )
    if initial.get("process_status") != "completed" or initial.get("output_truncated"):
        return {
            "status": "infra_error",
            "reward": 0.0,
            "detail": {"reason": "sandbox_initialization_incomplete", "runtime": initial},
        }
    if initial.get("stdout", "").strip() != "VERIFYIT_INIT_READY":
        return {"status": "scored", "reward": 0.0, "detail": {"test_results": [-4], "total_tests": len(expected)}}
    results = []
    grades = []
    comparisons = []
    runtime_results = []
    for arguments, target in zip(inputs, expected):
        call = (
            "_verifyit_input = "
            + repr(arguments)
            + "\ntry:\n    _verifyit_capture = io.StringIO()\n    with contextlib.redirect_stdout(_verifyit_capture):\n"
        )
        if function:
            call += "        _verifyit_prediction = _verifyit_method(*_verifyit_input)\n        if isinstance(_verifyit_prediction, tuple): _verifyit_prediction = list(_verifyit_prediction)\n"
        else:
            call += "        def _verifyit_checked():\n            try: return _verifyit_method()\n            except SystemExit as _exit:\n                if _exit.code not in (None, 0): raise RuntimeError('candidate exit')\n        call_method(_verifyit_checked, _verifyit_input)\n        _verifyit_prediction = _verifyit_capture.getvalue()\n"
        call += "    print('VERIFYIT_VALUE:' + json.dumps(_wire(_verifyit_prediction), allow_nan=False))\nexcept (Exception, SystemExit):\n    print('VERIFYIT_CANDIDATE_FAILURE')\n"
        output = sandbox.execute(
            call, language="ipython", session_id=session, timeout_seconds=timeout, max_output_characters=65536
        )
        runtime_results.append(output)
        if output.get("process_status") == "timeout":
            results.append(-3)
            grades.append(scored(0.0))
        elif output.get("process_status") != "completed" or output.get("output_truncated"):
            return {
                "status": "infra_error",
                "reward": 0.0,
                "detail": {"reason": "sandbox_execution_incomplete", "runtime": output},
            }
        else:
            stdout = output.get("stdout", "").strip()
            if not stdout.startswith("VERIFYIT_VALUE:"):
                results.append(-4)
                grades.append(scored(0.0))
            else:
                try:
                    prediction = _unwire(json.loads(stdout[len("VERIFYIT_VALUE:") :]))
                    actual = prediction
                except (ValueError, TypeError):
                    results.append(-2)
                    grades.append(scored(0.0))
                else:
                    if function:
                        verdict = grade_json_schema_candidate({"const": target}, actual)
                    else:
                        verdict = grade_stdio_candidate(stdio, actual, target)
                    if verdict.status.value != "scored":
                        return {
                            "status": verdict.status.value,
                            "reward": 0.0,
                            "detail": {**verdict.detail, "reason": "comparison_unavailable"},
                        }
                    grades.append(verdict)
                    results.append(True if verdict.reward == 1 else -2)
                    comparisons.append(
                        {
                            "module": (
                                grade_json_schema_candidate if function else grade_stdio_candidate
                            ).__code__.co_filename,
                            "spec": {"const": target} if function else {"compare": stdio.compare.value},
                            "expected": target,
                            "candidate": actual,
                            "reward": verdict.reward,
                            "verdict": {
                                "status": verdict.status.value,
                                "reward": verdict.reward,
                                "detail": verdict.detail,
                            },
                        }
                    )
        if results[-1] is not True and data["stop_on_failure"]:
            break
    aggregate = aggregate_rewards(
        grades,
        expected_total=len(expected),
        policy=Aggregation.MEAN if data["fractional"] else Aggregation.ALL,
    )
    return {
        "status": aggregate.status.value,
        "reward": aggregate.reward,
        "detail": {
            "test_results": results,
            "total_tests": len(expected),
            "aggregation": {
                "policy": "mean" if data["fractional"] else "all",
                "expected_total": len(expected),
                "components": [{"status": grade.status.value, "reward": grade.reward} for grade in grades],
                "verdict": {"status": aggregate.status.value, "reward": aggregate.reward},
            },
            "comparisons": comparisons,
            "runtime_results": runtime_results,
        },
    }


def _prepare_and_execute(data: dict) -> dict:
    from verifyit.grade import Aggregation, InvalidTask, aggregate_rewards, scored
    from verifyit.modes.grade_json_schema import grade_json_schema_candidate
    from verifyit.preparation.errors import PreparationError
    from skyrl_gym.envs.nemotron_ultra.answer_extraction import final_answer_text

    stage = "code_capture"
    provenance = {"policy": data["policy"]}
    try:
        policy = CodePolicy(data["policy"])
        inputs = capture_code_inputs(data["reference"], data["response"], reasoning_content=data["assistant_message"])
        if policy is CodePolicy.PREPARED:
            return _execute({**data, "tests": inputs.reference, "code": inputs.response})
        stage = "code_policy"
        text = inputs.response
        reference = inputs.reference
        reasoning = None
        metadata = {}
        if policy is CodePolicy.NEMOTRON:
            text = final_answer_text(text)
            metadata = reference.get("verifier_metadata") if isinstance(reference, dict) else None
            reference = metadata.get("unit_tests") if isinstance(metadata, dict) else None
            reasoning = (inputs.reasoning_content or {}).get("reasoning_content", "")
        elif not isinstance(reference, str):
            raise InvalidTask("LCB ground truth must be JSON text")
        prepared = prepare_code(CodeInputs(reference, text, reasoning))
        provenance = prepared.provenance()
        provenance["raw_response"] = inputs.response
        if policy is CodePolicy.NEMOTRON:
            provenance.update(
                response_policy="nemotron_final_answer_text_v1",
                grading_response=text,
                format_policy="nemotron_reasoning_tags_v1_nonstring_reasoning_empty",
                reasoning_format_penalty=data["reasoning_format_penalty"],
            )
        stage = "code_execution"
        verdict = _execute({**data, "tests": prepared.tests, "code": prepared.code})
        execution = verdict["detail"]
        execution["preparation"] = provenance
        execution["parsed_code"] = prepared.code
        if verdict["status"] != "scored" or policy is CodePolicy.LCB:
            return verdict
        if not prepared.code:
            execution["framework_output"] = {"extracted_model_code": None, "result": "missing_code"}
            return verdict
        stage = "code_format"
        format_verdict = grade_json_schema_candidate(
            {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "pattern": r"^(?![\s\S]*</?think>)[\s\S]*$"},
                    "reasoning": {
                        "type": "string",
                        "pattern": r"^(?![\s\S]*<think>[\s\S]*<think>)(?![\s\S]*</think>[\s\S]*</think>)[\s\S]*$",
                    },
                },
                "required": ["text", "reasoning"],
            },
            {"text": text, "reasoning": reasoning if isinstance(reasoning, str) else ""},
        )
        combined = aggregate_rewards(
            [scored(verdict["reward"]), format_verdict], expected_total=2, policy=Aggregation.ALL
        )
        execution["format_verdict"] = {"status": format_verdict.status.value, "reward": format_verdict.reward}
        execution["format_aggregation"] = {"status": combined.status.value, "reward": combined.reward}
        if combined.status.value != "scored":
            return {"status": combined.status.value, "reward": 0.0, "detail": execution}
        correct = verdict["reward"] == 1.0
        format_violation = format_verdict.reward == 0.0
        # The configured penalty shapes RL reward; core owns correctness.
        verdict["reward"] = data["reasoning_format_penalty"] if format_violation else combined.reward
        results = execution["test_results"]
        execution["framework_output"] = {
            "extracted_model_code": prepared.code,
            "test_results": results,
            "executed_tests": len(results),
            "total_tests": len(prepared.tests),
            "result": "pass" if correct else "failed_tests",
            "reasoning_format_violation_rate": float(format_violation),
            "difficulty": metadata.get("difficulty"),
        }
        return verdict
    except PreparationError as error:
        return {
            "status": error.verdict.status.value,
            "reward": 0.0,
            "detail": {**error.verdict.detail, "protected_traceback": traceback.format_exc()},
        }
    except Exception as error:
        return {
            "status": "invalid_task" if isinstance(error, InvalidTask) else "infra_error",
            "reward": 0.0,
            "detail": {
                "reason": "Code verification unavailable",
                "error_type": type(error).__name__,
                "stage": stage,
                "preparation": provenance,
                "protected_traceback": traceback.format_exc(),
            },
        }


def _execute_code_verifyit(
    reference,
    response: str,
    *,
    timeout: int = 6,
    fractional: bool = False,
    limits=None,
    sandbox=None,
    policy: CodePolicy = CodePolicy.PREPARED,
    assistant_message=None,
    reasoning_format_penalty: float = 0.0,
    started: float,
):
    from verifyit.bounded import call_bounded
    from verifyit.grade import InvalidTask

    from skyrl_gym.envs.lcb.livecodebench import DEFAULT_LIMITS, verifier_slots
    from skyrl_gym.envs.nemotron_ultra.sandbox import SandboxClient
    import requests

    limits = DEFAULT_LIMITS if limits is None else limits
    budget = limits.total_timeout_seconds
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget <= 0:
        raise code_verification_failure(
            "invalid_task",
            {"reason": "Opt-in code verification requires a finite positive total_timeout_seconds"},
            stage="code_budget",
        )
    deadline = started + budget
    session = str(uuid.uuid4())
    cleanup = SandboxClient() if sandbox is None else sandbox
    data = {
        "reference": reference,
        "response": response,
        "policy": policy.value,
        "assistant_message": assistant_message,
        "reasoning_format_penalty": reasoning_format_penalty,
        "timeout": timeout,
        "fractional": fractional,
        "stop_on_failure": not fractional,
        "memory_limit": limits.max_memory_bytes,
        "setup_timeout": budget,
        "session": session,
        "host": cleanup.host,
        "port": cleanup.port,
    }
    verdict = None
    detail = {}
    slots = verifier_slots()
    acquired = False
    stage = "code_queue"
    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not slots.acquire(timeout=remaining):
            raise TimeoutError("Code verification exceeded its total budget while queued")
        acquired = True
        stage = "code_worker"
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Code verification exceeded its total budget")
        data["setup_timeout"] = remaining
        verdict = call_bounded(_prepare_and_execute, data, timeout=remaining)
    except Exception as error:
        logger.exception("Code verifier worker failed")
        verdict = {
            "status": "invalid_task" if isinstance(error, InvalidTask) else "infra_error",
            "reward": 0.0,
            "detail": {
                "reason": "Code verification unavailable",
                "error_type": type(error).__name__,
                "stage": stage,
                "preparation": {"policy": policy.value},
            },
        }
    finally:
        if acquired:
            slots.release()
        if verdict is not None:
            detail = verdict["detail"]
        protected_traceback = detail.pop("protected_traceback", None)
        if protected_traceback:
            logger.error("Code worker traceback:\n%s", protected_traceback)
        detail["total_timeout_seconds"] = budget
        detail["cleanup_timeout_seconds"] = CLEANUP_TIMEOUT_SECONDS
        detail["sandbox_session_id"] = session
        try:
            response = requests.delete(
                f"http://{cleanup.host}:{cleanup.port}/sessions/{session}",
                headers={"X-Session-ID": session},
                timeout=CLEANUP_TIMEOUT_SECONDS,
            )
            detail["sandbox_cleanup_status"] = response.status_code
            if response.status_code != 404:
                response.raise_for_status()
        except requests.RequestException as error:
            detail["cleanup_error"] = {
                "error_type": type(error).__name__,
                "message": str(error),
                "stage": "code_cleanup",
            }
            if verdict is not None and verdict["status"] == "scored":
                verdict = {"status": "infra_error", "reward": 0.0, "detail": detail}
                detail.update(reason="sandbox_cleanup_failed", stage="code_cleanup")
            elif verdict is None:
                # An unexpected primary exception still propagates after cleanup.
                primary = error.__context__
                if primary is not None:
                    primary.add_note(f"Code session cleanup failed: {error}")
    assert verdict is not None
    if verdict["status"] != "scored":
        raise code_verification_failure(verdict["status"], detail, stage=detail.get("stage", "code_execution"))
    return verdict["reward"], detail


def execute_code_verifyit(
    reference,
    response: str,
    *,
    timeout: int = 6,
    fractional: bool = False,
    limits=None,
    sandbox=None,
    policy: CodePolicy = CodePolicy.PREPARED,
    assistant_message=None,
    reasoning_format_penalty: float = 0.0,
):
    started = time.monotonic()
    from verifyit.grade import InvalidTask
    from verifyit.preparation.errors import PreparationError

    try:
        return _execute_code_verifyit(
            reference,
            response,
            timeout=timeout,
            fractional=fractional,
            limits=limits,
            sandbox=sandbox,
            policy=policy,
            assistant_message=assistant_message,
            reasoning_format_penalty=reasoning_format_penalty,
            started=started,
        )
    except PreparationError:
        raise
    except Exception as error:
        logger.exception("Code verification boundary failed")
        raise code_verification_failure(
            "invalid_task" if isinstance(error, InvalidTask) else "infra_error",
            {"reason": "Code verification unavailable", "error_type": type(error).__name__},
            stage="code_execution",
        ) from error
