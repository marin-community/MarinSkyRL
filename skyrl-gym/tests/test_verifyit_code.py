"""Hidden references and reward artifacts must remain outside candidate execution."""

import pytest

from skyrl_gym.envs.lcb import verifyit_execution
from skyrl_gym.envs.lcb.verifyit_execution import execute_code_verifyit
from skyrl_gym.envs.nemotron_ultra import code_gen


@pytest.mark.parametrize(
    "code",
    [
        """def solve(x):
    import sys, json
    frame = sys._getframe()
    while frame:
        data = frame.f_locals.get('data')
        if isinstance(data, dict) and 'tests' in data:
            return json.loads(data['tests'][0]['output'])
        frame = frame.f_back
    return -999
""",
        """def solve(x):
    import os, json
    from pathlib import Path
    logs = os.environ.get('VERIFYIT_LOGS_DIR')
    if logs:
        Path(logs, 'code-result.json').write_text(json.dumps({'status':'scored','reward':1.0,'detail':{'test_results':[True]}}))
        os._exit(0)
    return -999
""",
    ],
)
def test_candidate_cannot_recover_reference_or_overwrite_verdict(code):
    tests = [{"input": "7", "output": "938171", "testtype": "functional", "metadata": {"func_name": "solve"}}]
    try:
        reward, _ = execute_code_verifyit(tests, code)
    except RuntimeError:
        reward = 0
    assert reward == 0


@pytest.mark.parametrize(
    "code,expected,reward",
    [
        ("def solve(x): return x*2", "14", 1),
        ("def solve(x): return (x, x)", "[7,7]", 1),
        ("def solve(x): return [(x, x)]", "[[7,7]]", 0),
        ("def solve(x): return True", "1", 0),
        ("def solve(x): return 1.0", "1", 1),
        ('def solve(x): return {"1": [x, True]}', '{"1": [7, true]}', 1),
        ("def solve(x): return {1: [x, True]}", '{"1": [7, true]}', 0),
        ('def solve(x): return float("nan")', "14", 0),
        ('def solve(x): raise RuntimeError("bad")', "14", 0),
    ],
)
def test_callable_runtime_preserves_types_and_rejects_failures(code, expected, reward):
    tests = [{"input": "7", "output": expected, "testtype": "functional", "metadata": {"func_name": "solve"}}]
    assert execute_code_verifyit(tests, code)[0] == reward


def test_stateful_callable_and_fractional_aggregation():
    tests = [
        {"input": "0", "output": str(answer), "testtype": "functional", "metadata": {"func_name": "solve"}}
        for answer in (1, 2)
    ]
    stateful = "counter=0\ndef solve(x):\n    global counter\n    counter += 1\n    return counter"
    assert execute_code_verifyit(tests, stateful)[0] == 1
    assert execute_code_verifyit(tests, "def solve(x): return 1", fractional=True)[0] == 0.5


@pytest.mark.parametrize(
    "code,reward",
    [
        ('print("1.000 2")', 1),
        ('print("1 2"); raise RuntimeError("after output")', 0),
        ('import sys; print("1 2"); sys.exit(1)', 0),
        ('print("nan 2")', 0),
    ],
)
def test_stdio_runtime_numeric_equivalence_and_failed_execution(code, reward):
    tests = [{"input": "", "output": "1 2", "testtype": "stdin"}]
    assert execute_code_verifyit(tests, code)[0] == reward


def test_invalid_reference_never_awards_partial_credit():
    tests = [{"input": "1", "output": "NaN", "testtype": "functional", "metadata": {"func_name": "solve"}}]
    with pytest.raises(RuntimeError, match="unavailable"):
        execute_code_verifyit(tests, "def solve(x): return 1", fractional=True)


def test_sandbox_session_deleted_after_success():
    import requests

    tests = [{"input": "1", "output": "1", "testtype": "functional", "metadata": {"func_name": "solve"}}]
    reward, detail = execute_code_verifyit(tests, "def solve(x): return x")
    assert reward == 1
    assert detail["sandbox_cleanup_status"] == 200
    session = detail["sandbox_session_id"]
    response = requests.delete(
        f"http://127.0.0.1:6000/sessions/{session}", headers={"X-Session-ID": session}, timeout=5
    )
    assert response.status_code == 404


def test_sandbox_session_deleted_after_checker_deadline(monkeypatch):
    import requests

    from skyrl_gym.envs.lcb.livecodebench import VerifierLimits

    original = requests.delete
    deletions = []

    def observed_delete(url, **kwargs):
        response = original(url, **kwargs)
        deletions.append((url, kwargs, response.status_code))
        return response

    monkeypatch.setattr(requests, "delete", observed_delete)
    tests = [{"input": "1", "output": "1", "testtype": "functional", "metadata": {"func_name": "solve"}}]
    with pytest.raises(RuntimeError, match="unavailable"):
        execute_code_verifyit(
            tests, "def solve(x):\n    while True: pass", timeout=10, limits=VerifierLimits(total_timeout_seconds=3)
        )
    assert deletions[0][2] == 200
    url, kwargs, _ = deletions[0]
    assert original(url, **kwargs).status_code == 404


def test_memory_bound_is_applied_before_candidate_module_execution():
    from skyrl_gym.envs.lcb.livecodebench import VerifierLimits

    bound = 4 * 1024**3
    tests = [{"input": "0", "output": str(bound), "testtype": "functional", "metadata": {"func_name": "solve"}}]
    code = (
        "import resource\nmodule_limit = resource.getrlimit(resource.RLIMIT_DATA)[0]\ndef solve(x): return module_limit"
    )
    assert execute_code_verifyit(tests, code, limits=VerifierLimits(max_memory_bytes=bound))[0] == 1


def test_candidate_input_mutation_cannot_change_frozen_reference():
    tests = [
        {
            "input": '{"value": 1}',
            "output": '{"value": 1}',
            "testtype": "functional",
            "metadata": {"func_name": "solve"},
        }
    ]
    code = 'def solve(x):\n    x["value"] = 99\n    return x'
    assert execute_code_verifyit(tests, code)[0] == 0


def test_runtime_failure_discards_prior_partial_credit():
    tests = [
        {"input": str(x), "output": "1", "testtype": "functional", "metadata": {"func_name": "solve"}} for x in (0, 1)
    ]
    code = "def solve(x):\n    import os\n    if x: os._exit(0)\n    return 1"
    with pytest.raises(RuntimeError, match="unavailable"):
        execute_code_verifyit(tests, code, fractional=True)


def test_candidate_cannot_raise_disposable_session_memory_bound():
    tests = [{"input": "0", "output": "true", "testtype": "functional", "metadata": {"func_name": "solve"}}]
    code = """import resource
soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
try:
    resource.setrlimit(resource.RLIMIT_DATA, (hard * 2, hard * 2))
except (ValueError, PermissionError):
    rejected = True
else:
    rejected = False
def solve(x): return rejected and resource.getrlimit(resource.RLIMIT_DATA) == (soft, hard)
"""
    assert execute_code_verifyit(tests, code)[0] == 1


@pytest.mark.parametrize(
    "body,sentinel",
    [
        ("return x + 1", -2),
        ('raise ValueError("candidate fault")', -4),
    ],
)
def test_source_code_boundary_retains_failure_sentinels(body, sentinel):
    from skyrl_gym.envs.nemotron_ultra.code_gen import grade_code

    record = {
        "verifier_metadata": {
            "unit_tests": [{"input": "7", "output": "7", "testtype": "functional", "metadata": {"func_name": "solve"}}]
        }
    }
    reward, details = grade_code(
        "```python\ndef solve(x):\n    " + body + "\n```", record, timeout_seconds=1, verifyit_enabled=True
    )
    assert reward == 0
    assert details["test_results"] == [sentinel]
    assert details["executed_tests"] == details["total_tests"] == 1


@pytest.mark.parametrize("lost_session", [False, True])
def test_explicit_protocol_timeout_and_lost_session_have_distinct_statuses(lost_session):
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from skyrl_gym.envs.nemotron_ultra.code_gen import grade_code
    from skyrl_gym.envs.nemotron_ultra.sandbox import SandboxClient

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            code = data["generated_code"]
            if "VERIFYIT_RUNTIME_READY" in code:
                result = {"process_status": "completed", "stdout": "VERIFYIT_RUNTIME_READY\n"}
            elif "VERIFYIT_LIMIT_READY" in code:
                result = {"process_status": "completed", "stdout": "VERIFYIT_LIMIT_READY\n"}
            elif "VERIFYIT_INIT_READY" in code:
                result = {"process_status": "completed", "stdout": "VERIFYIT_INIT_READY\n"}
            else:
                result = {"process_status": "timeout", "stdout": "", "new_session_created": lost_session}
            body = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_DELETE(self):
            self.send_response(200)
            self.end_headers()

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        record = {
            "verifier_metadata": {
                "unit_tests": [
                    {"input": "7", "output": "7", "testtype": "functional", "metadata": {"func_name": "solve"}}
                ]
            }
        }
        kwargs = {"timeout_seconds": 1, "verifyit_enabled": True, "sandbox": SandboxClient(port=server.server_port)}
        if lost_session:
            with pytest.raises(RuntimeError, match="unavailable"):
                grade_code("```python\ndef solve(x): return x\n```", record, **kwargs)
        else:
            reward, details = grade_code("```python\ndef solve(x): return x\n```", record, **kwargs)
            assert reward == 0
            assert details["test_results"] == [-3]
            assert details["executed_tests"] == details["total_tests"] == 1
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


@pytest.mark.parametrize(
    "text,reasoning,violation",
    [
        ("", "", False),
        ("<THINK>", "<THINK><THINK>", False),
        ("<thinking>", "<think></think>", False),
        ("prefix<think>", "", True),
        ("line\n</think>", "", True),
        ("", "<think>\n<think>", True),
        ("", "</think></think>", True),
        ("", "<think<think>", False),
        ("", None, False),
    ],
)
@pytest.mark.parametrize("penalty", [0.0, -0.25])
def test_code_format_schema_preserves_native_tags_and_rl_penalty(monkeypatch, text, reasoning, violation, penalty):
    # Execution is the external boundary; both paths receive one passing case.
    monkeypatch.setattr(code_gen, "lcb_execution_result", lambda *args, **kwargs: ([True], {}))
    monkeypatch.setattr(
        verifyit_execution, "execute_code_verifyit", lambda *args, **kwargs: (1.0, {"test_results": [True]})
    )
    answer = text + "\n```python\nprint(1)\n```"
    record = {"verifier_metadata": {"unit_tests": [{"input": "", "output": "1", "testtype": "stdin"}]}}
    rewards = []
    for enabled in (False, True):
        reward, detail = code_gen.grade_code(
            answer,
            record,
            assistant_message={"reasoning_content": reasoning},
            reasoning_format_penalty=penalty,
            verifyit_enabled=enabled,
        )
        rewards.append(reward)
        assert detail["reasoning_format_violation_rate"] == float(violation)
    assert rewards == [penalty if violation else 1.0] * 2
