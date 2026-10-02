"""LiveCodeBench sandbox transport using core comparison and aggregation."""

from __future__ import annotations

import inspect
import json
import uuid


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
    except (InvalidTask, KeyError, TypeError, ValueError):
        return {"status": "invalid_task", "reward": 0.0, "detail": {"reason": "malformed_tests"}}
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
                        return {"status": "infra_error", "reward": 0.0, "detail": {"reason": "comparison_unavailable"}}
                    grades.append(verdict)
                    results.append(True if verdict.reward == 1 else -2)
                    comparisons.append(
                        {
                            "module": (
                                grade_json_schema_candidate if function else grade_stdio_candidate
                            ).__code__.co_filename,
                            "expected": target,
                            "candidate": actual,
                            "reward": verdict.reward,
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
            "comparisons": comparisons,
            "runtime_results": runtime_results,
        },
    }


def _execute_code_verifyit(
    tests: list, code: str, *, timeout: int = 6, fractional: bool = False, limits=None, sandbox=None
):
    from verifyit.bounded import call_bounded

    from skyrl_gym.envs.lcb.livecodebench import DEFAULT_LIMITS, verifier_slots
    from skyrl_gym.envs.nemotron_ultra.sandbox import SandboxClient
    import requests

    limits = DEFAULT_LIMITS if limits is None else limits
    deadline = (timeout + 1) * len(tests) + 5
    if limits.total_timeout_seconds is not None:
        deadline = min(deadline, limits.total_timeout_seconds)
    session = str(uuid.uuid4())
    cleanup = SandboxClient() if sandbox is None else sandbox
    data = {
        "tests": tests,
        "code": code,
        "timeout": timeout,
        "fractional": fractional,
        "stop_on_failure": not fractional,
        "memory_limit": limits.max_memory_bytes,
        "setup_timeout": deadline,
        "session": session,
        "host": cleanup.host,
        "port": cleanup.port,
    }
    try:
        with verifier_slots():
            verdict = call_bounded(_execute, data, timeout=deadline)
    finally:
        response = requests.delete(
            f"http://{cleanup.host}:{cleanup.port}/sessions/{session}",
            headers={"X-Session-ID": session},
            timeout=deadline + 10.0,
        )
        if response.status_code != 404:
            response.raise_for_status()
    verdict["detail"]["sandbox_session_id"] = session
    verdict["detail"]["sandbox_cleanup_status"] = response.status_code
    if verdict["status"] != "scored":
        raise RuntimeError(
            f"Code verification unavailable ({verdict['status']}: {verdict['detail'].get('reason', 'runtime failure')})"
        )
    return verdict["reward"], verdict["detail"]


def execute_code_verifyit(
    tests: list, code: str, *, timeout: int = 6, fractional: bool = False, limits=None, sandbox=None
):
    try:
        return _execute_code_verifyit(
            tests, code, timeout=timeout, fractional=fractional, limits=limits, sandbox=sandbox
        )
    except (ImportError, OSError, TypeError, ValueError, RuntimeError) as error:
        raise RuntimeError("Code verification unavailable") from error
