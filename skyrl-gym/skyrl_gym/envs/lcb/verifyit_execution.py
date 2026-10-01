"""Trusted LCB runtime with existing exact comparisons and source aggregation."""

from __future__ import annotations

import inspect
import json
import math
import os
import shlex
import sys
import uuid
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory


def _typed(value):
    if value is None or isinstance(value, (str, bool)):
        return [type(value).__name__, value]
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("nonfinite result")
        number = Fraction(value)
        return ["number", number.numerator, number.denominator]
    if isinstance(value, list):
        return ["list", [_typed(item) for item in value]]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("unsupported result key")
        return ["dict", [[key, _typed(value[key])] for key in sorted(value)]]
    raise ValueError("unsupported result type")


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
        return tuple(_unwire(item) for item in payload)
    if kind == "list":
        return [_unwire(item) for item in payload]
    if kind == "dict":
        return {_unwire(key): _unwire(item) for key, item in payload}
    if kind in {"NoneType", "str", "bool", "int", "float"} and type(payload).__name__ == kind:
        return payload
    raise ValueError("malformed output")


def _stdio(value: str):
    lines = []
    for line in value.strip().split("\n"):
        line = line.strip()
        try:
            numbers = [Decimal(token) for token in line.split()]
        except InvalidOperation:
            lines.append(["literal", line])
        else:
            if any(not number.is_finite() for number in numbers):
                raise ValueError("nonfinite output")
            # Decimal equality ignores token whitespace and harmless numeric spelling.
            lines.append(["numbers", [[Fraction(n).numerator, Fraction(n).denominator] for n in numbers]])
    return lines


def _execute(data: dict) -> dict:
    from verifyit.modes.grade_exact import grade_exact_candidate
    from verifyit.spec import ExactSpec

    from skyrl_gym.envs.lcb import livecodebench as runtime

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
        expected = [json.dumps(_typed(value) if function else _stdio(value), ensure_ascii=False) for value in expected]
        if len(inputs) != len(expected) or not expected:
            raise ValueError("misaligned tests")
    except (KeyError, TypeError, ValueError):
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
            else:
                try:
                    prediction = _unwire(json.loads(stdout[len("VERIFYIT_VALUE:") :]))
                    actual = json.dumps(_typed(prediction) if function else _stdio(prediction), ensure_ascii=False)
                except (ValueError, TypeError):
                    results.append(-2)
                else:
                    verdict = grade_exact_candidate(
                        ExactSpec(
                            expected=(target,), ignore_case=False, ignore_whitespace=False, strip_outer_whitespace=False
                        ),
                        actual,
                    )
                    if verdict.status.value != "scored":
                        return {"status": "infra_error", "reward": 0.0, "detail": {"reason": "comparison_unavailable"}}
                    results.append(True if verdict.reward == 1 else -2)
                    comparisons.append(
                        {
                            "module": grade_exact_candidate.__code__.co_filename,
                            "expected": target,
                            "candidate": actual,
                            "reward": verdict.reward,
                        }
                    )
        if results[-1] is not True and data["stop_on_failure"]:
            break
    reward = (
        sum(result is True for result in results) / len(results)
        if data["fractional"]
        else float(all(result is True for result in results))
    )
    return {
        "status": "scored",
        "reward": reward,
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
    from verifyit.grade import Status, run
    from verifyit.spec import ScriptSpec, render_spec

    from skyrl_gym.envs.lcb.livecodebench import DEFAULT_LIMITS, verifier_slots

    limits = DEFAULT_LIMITS if limits is None else limits
    deadline = (timeout + 1) * len(tests) + 5
    if limits.total_timeout_seconds is not None:
        deadline = min(deadline, limits.total_timeout_seconds)
    from skyrl_gym.envs.nemotron_ultra.sandbox import SandboxClient

    session = str(uuid.uuid4())
    cleanup = SandboxClient() if sandbox is None else sandbox
    with TemporaryDirectory(prefix="skyrl-code-") as directory:
        root = Path(directory)
        (root / "checker.py").write_text(Path(__file__).read_text())
        (root / "checker.sh").write_text(
            "#!/bin/sh\nexec " + shlex.quote(sys.executable) + " " + shlex.quote(str(root / "checker.py")) + "\n"
        )
        (root / "data.json").write_text(
            json.dumps(
                {
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
                },
                allow_nan=False,
            )
        )
        (root / "verifier.toml").write_text(
            render_spec(ScriptSpec(path="checker.sh", verdict_file="code-result.json", timeout=deadline))
        )
        try:
            with verifier_slots():
                verdict = run(root / "verifier.toml", root)
        finally:
            import requests

            response = requests.delete(
                f"http://{cleanup.host}:{cleanup.port}/sessions/{session}",
                headers={"X-Session-ID": session},
                timeout=deadline + 10.0,
            )
            if response.status_code != 404:
                response.raise_for_status()
        verdict.detail["sandbox_session_id"] = session
        verdict.detail["sandbox_cleanup_status"] = response.status_code
        if verdict.status != Status.SCORED:
            raise RuntimeError(
                f"Code verification unavailable ({verdict.status.value}: {verdict.detail.get('reason', 'runtime failure')})"
            )
        return verdict.reward, verdict.detail


def execute_code_verifyit(
    tests: list, code: str, *, timeout: int = 6, fractional: bool = False, limits=None, sandbox=None
):
    try:
        return _execute_code_verifyit(
            tests, code, timeout=timeout, fractional=fractional, limits=limits, sandbox=sandbox
        )
    except (ImportError, OSError, TypeError, ValueError, RuntimeError) as error:
        raise RuntimeError("Code verification unavailable") from error


if __name__ == "__main__":
    data = json.loads((Path(os.environ["VERIFYIT_TESTS_DIR"]) / "data.json").read_text())
    verdict = _execute(data)
    (Path(os.environ["VERIFYIT_LOGS_DIR"]) / "code-result.json").write_text(json.dumps(verdict, allow_nan=False))
