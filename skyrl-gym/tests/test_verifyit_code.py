"""Hidden references and reward artifacts must remain outside candidate execution."""

import pytest

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
    from verifyit.grade import InvalidTask

    with pytest.raises(InvalidTask):
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
    with pytest.raises(RuntimeError):
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
    with pytest.raises(RuntimeError):
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
            with pytest.raises(RuntimeError):
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
        ("line\n</think>", "", False),
        ("", "<think>\n<think>", True),
        ("", "</think></think>", True),
        ("", "<think<think>", False),
        ("", None, False),
    ],
)
@pytest.mark.parametrize("penalty", [0.0, -0.25])
def test_code_format_schema_preserves_native_tags_and_rl_penalty(text, reasoning, violation, penalty):
    from skyrl_gym.envs.nemotron_ultra.answer_extraction import final_answer_text

    answer = text + "\n```python\nprint(1)\n```"
    missing_code = text in {"<thinking>", "prefix<think>"}
    record = {"verifier_metadata": {"unit_tests": [{"input": "", "output": "1", "testtype": "stdin"}]}}
    rewards = []
    for enabled in (False, True):
        reward, detail = code_gen.grade_code(
            answer if enabled else final_answer_text(answer),
            record,
            assistant_message={"reasoning_content": reasoning},
            reasoning_format_penalty=penalty,
            verifyit_enabled=enabled,
        )
        rewards.append(reward)
        if missing_code:
            assert detail["result"] == "missing_code"
        else:
            assert detail["reasoning_format_violation_rate"] == float(violation)
    assert rewards == [0.0 if missing_code else penalty if violation else 1.0] * 2


@pytest.mark.parametrize("route", ["lcb", "nemotron_ultra"])
@pytest.mark.parametrize(
    "reference",
    [None, [], [{"input": "1", "output": "NaN", "testtype": "functional", "metadata": {"func_name": "solve"}}]],
)
def test_code_framework_invalid_reference_remains_unscored(route, reference):
    import json
    from omegaconf import OmegaConf
    from skyrl_gym.envs.lcb.env import LCBEnv
    from skyrl_gym.envs.nemotron_ultra.env import NemotronUltraEnv
    from skyrl_gym.verification import VerificationStatus

    if route == "lcb":
        env = LCBEnv(
            OmegaConf.create({"verifyit_enabled": True}),
            extras={"reward_model": {"ground_truth": json.dumps(reference)}},
        )
    else:
        env = NemotronUltraEnv(
            OmegaConf.create({"verifyit_enabled": True}),
            extras={
                "extra_info": {
                    "nemotron_ultra": {
                        "route": "skyrl_gym",
                        "agent": "code_gen_simple_agent",
                        "request_json": "{}",
                        "record_json": json.dumps({"verifier_metadata": {"unit_tests": reference}}),
                    }
                }
            },
        )
    result = env.step("")
    assert result["reward"] == 0
    assert result["verification"].status == VerificationStatus.ERROR
    assert result["verification"].score is None
    assert result["metadata"]["verifyit_status"] == "invalid_task"
    assert result["metadata"]["preparation_stage"] in {"code_tests", "code_execution", "code_policy"}


def test_invalid_task_cleanup_failure_preserves_primary_origin(monkeypatch):
    import requests
    from verifyit.grade import InvalidTask
    from verifyit.preparation.errors import PreparationError

    def failed_cleanup(*args, **kwargs):
        raise requests.ConnectionError("cleanup disconnected")

    monkeypatch.setattr(requests, "delete", failed_cleanup)
    with pytest.raises(InvalidTask) as caught:
        execute_code_verifyit([], "")
    error = caught.value
    assert isinstance(error, PreparationError)
    assert error.verdict.status.value == "invalid_task"
    assert error.verdict.reward == 0
    assert error.verdict.detail["reason"] == "missing_tests"
    assert error.verdict.detail["cleanup_error"]["error_type"] == "ConnectionError"


def test_prepared_code_preserves_snapshot_and_records_source_extraction():
    from skyrl_gym.envs.lcb.verifyit_execution import capture_code_inputs, prepare_code

    reference = [{"input": "", "output": "1", "testtype": "stdin", "source_id": "retained"}]
    response = "```python\nprint(0)\n```\n```python\n print(1) \n```"
    captured = capture_code_inputs(reference, response)
    reference[0]["output"] = "changed"
    prepared = prepare_code(captured)
    assert prepared.inputs.reference[0]["output"] == "1"
    assert prepared.inputs.reference[0]["source_id"] == "retained"
    assert prepared.code == "print(1)"
    assert prepared.provenance()["raw_response"] == response
    assert "raw_reference" not in prepared.provenance()


@pytest.mark.parametrize("route", ["lcb", "nemotron_ultra"])
def test_code_framework_diagnostics_do_not_reveal_trusted_output(route):
    import json
    from dataclasses import asdict
    from omegaconf import OmegaConf
    from skyrl_gym.envs.lcb.env import LCBEnv
    from skyrl_gym.envs.nemotron_ultra.env import NemotronUltraEnv

    secret = "trusted-hidden-output-917532"
    tests = [{"input": "1", "output": json.dumps(secret), "testtype": "functional", "metadata": {"func_name": "solve"}}]
    if route == "lcb":
        env = LCBEnv(
            OmegaConf.create({"verifyit_enabled": True}), extras={"reward_model": {"ground_truth": json.dumps(tests)}}
        )
    else:
        env = NemotronUltraEnv(
            OmegaConf.create({"verifyit_enabled": True}),
            extras={
                "extra_info": {
                    "nemotron_ultra": {
                        "route": "skyrl_gym",
                        "agent": "code_gen_simple_agent",
                        "request_json": "{}",
                        "record_json": json.dumps({"verifier_metadata": {"unit_tests": tests}}),
                    }
                }
            },
        )
    result = env.step("```python\ndef solve(x): return 'wrong'\n```")
    assert result["reward"] == 0
    assert result["verification"].score == 0
    assert secret not in json.dumps(result["metadata"])
    assert secret not in json.dumps(asdict(result["verification"]))


@pytest.mark.parametrize("route", ["lcb", "nemotron_ultra"])
@pytest.mark.parametrize("dependency_error", ["ModuleNotFoundError", "AssertionError"])
def test_code_optional_dependency_absence_preserves_native_path_and_enabled_error(route, dependency_error):
    import subprocess
    import sys
    import json

    program = r"""
import importlib.abc, sys, json, builtins
class MissingVerifyit(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "verifyit" or fullname.startswith("verifyit."):
            raise getattr(builtins, sys.argv[2])(fullname)
sys.meta_path.insert(0, MissingVerifyit())
from omegaconf import OmegaConf
from skyrl_gym.envs.lcb.env import LCBEnv
from skyrl_gym.envs.nemotron_ultra.env import NemotronUltraEnv
results = []
for enabled in (False, True):
    config = OmegaConf.create({"verifyit_enabled": enabled})
    tests = [{"input": "", "output": "1", "testtype": "stdin"}]
    if sys.argv[1] == "lcb":
        env = LCBEnv(config, extras={"reward_model": {"ground_truth": json.dumps(tests)}})
    else:
        env = NemotronUltraEnv(config, extras={"extra_info": {"nemotron_ultra": {
            "route": "skyrl_gym", "agent": "code_gen_simple_agent", "request_json": "{}",
            "record_json": json.dumps({"verifier_metadata": {"unit_tests": tests}}),
        }}})
    result = env.step("")
    verification = result.get("verification")
    results.append({"reward": result["reward"], "error": verification is not None and verification.status.value == "error", "metadata": result["metadata"]})
print(json.dumps(results))
"""
    result = subprocess.run(
        [sys.executable, "-c", program, route, dependency_error], capture_output=True, text=True, check=True, timeout=20
    )
    native, enabled = json.loads(result.stdout)
    assert (native["reward"], native["error"]) == (0.0, False)
    assert (enabled["reward"], enabled["error"]) == (0.0, True)
    assert enabled["metadata"]["verifyit_status"] == "infra_error"
    assert enabled["metadata"]["preparation_stage"] == "code_boundary"
    assert enabled["metadata"]["preparation"]["policy"] == "lcb_source_v1"


def test_scored_result_cleanup_failure_becomes_unscored_infrastructure(monkeypatch):
    import requests
    from verifyit.preparation.errors import PreparationError

    def failed_cleanup(*args, **kwargs):
        raise requests.ConnectionError("cleanup disconnected")

    monkeypatch.setattr(requests, "delete", failed_cleanup)
    with pytest.raises(PreparationError) as caught:
        execute_code_verifyit([{"input": "", "output": "1", "testtype": "stdin"}], "")
    assert caught.value.verdict.reward == 0
    assert caught.value.verdict.status.value == "infra_error"
    assert caught.value.failure.stage == "code_cleanup"
    assert caught.value.verdict.detail["cleanup_error"]["error_type"] == "ConnectionError"


@pytest.mark.parametrize("route", ["lcb", "nemotron_ultra"])
def test_code_framework_unexpected_launch_failure_is_typed_and_logged(route, monkeypatch, caplog):
    import json
    import subprocess
    import requests
    from omegaconf import OmegaConf
    from skyrl_gym.envs.lcb.env import LCBEnv
    from skyrl_gym.envs.nemotron_ultra.env import NemotronUltraEnv
    from skyrl_gym.verification import VerificationStatus

    def failed_launch(*args, **kwargs):
        raise AssertionError("private launch fault traceback")

    deleted = []

    def cleanup(url, **kwargs):
        deleted.append(url)
        response = requests.Response()
        response.status_code = 404
        return response

    tests = [{"input": "", "output": "1", "testtype": "stdin"}]
    config = OmegaConf.create({"verifyit_enabled": True})
    if route == "lcb":
        env = LCBEnv(config, extras={"reward_model": {"ground_truth": json.dumps(tests)}})
    else:
        env = NemotronUltraEnv(
            config,
            extras={
                "extra_info": {
                    "nemotron_ultra": {
                        "route": "skyrl_gym",
                        "agent": "code_gen_simple_agent",
                        "request_json": "{}",
                        "record_json": json.dumps({"verifier_metadata": {"unit_tests": tests}}),
                    }
                }
            },
        )
    monkeypatch.setattr(subprocess, "Popen", failed_launch)
    monkeypatch.setattr(requests, "delete", cleanup)
    result = env.step("```python\nprint(1)\n```")
    assert result["reward"] == 0
    assert result["verification"].status == VerificationStatus.ERROR
    assert result["verification"].score is None
    assert result["metadata"]["verifyit_status"] == "infra_error"
    assert result["metadata"]["error_type"] == "AssertionError"
    assert result["metadata"]["preparation_stage"] == "code_worker"
    assert result["metadata"]["sandbox_cleanup_status"] == 404
    assert deleted
    assert any(record.exc_info and record.exc_info[0] is AssertionError for record in caplog.records)
    assert "private launch fault traceback" not in json.dumps(result["metadata"])


def code_environment(route, reference, **config):
    import json
    from omegaconf import OmegaConf
    import skyrl_gym

    env_config = OmegaConf.create({"verifyit_enabled": True, **config})
    if route == "lcb":
        return skyrl_gym.make("lcb", env_config=env_config, extras={"reward_model": {"ground_truth": reference}})
    return skyrl_gym.make(
        "nemotron_ultra",
        env_config=env_config,
        extras={
            "extra_info": {
                "nemotron_ultra": {
                    "route": "skyrl_gym",
                    "agent": "code_gen_simple_agent",
                    "request_json": "{}",
                    "record_json": json.dumps({"verifier_metadata": {"unit_tests": reference}}),
                }
            }
        },
    )


@pytest.mark.parametrize("route", ["lcb", "nemotron_ultra"])
def test_code_full_route_requires_finite_total_budget(route):
    import json

    reference = [{"input": "", "output": "1", "testtype": "stdin"}]
    env = code_environment(
        route, json.dumps(reference) if route == "lcb" else reference, code_verifier={"total_timeout_seconds": None}
    )
    result = env.step("```python\nprint(1)\n```")
    assert result["reward"] == 0
    assert result["verification"].score is None
    assert result["metadata"]["verifyit_status"] == "invalid_task"
    assert result["metadata"]["preparation_stage"] == "code_budget"


@pytest.mark.parametrize("route", ["lcb", "nemotron_ultra"])
def test_code_queue_wait_consumes_total_budget(route, monkeypatch):
    import json
    import threading
    import requests
    from skyrl_gym.envs.lcb.livecodebench import verifier_slots

    response = requests.Response()
    response.status_code = 404
    monkeypatch.setattr(requests, "delete", lambda *args, **kwargs: response)
    slots = verifier_slots()
    held = 0
    while slots.acquire(blocking=False):
        held += 1
    assert held > 0
    finished = threading.Event()
    released = threading.Event()

    def release_after_stall():
        # Deliberately occupy the queue longer than the operation's budget.
        if not finished.wait(2):
            for _ in range(held):
                slots.release()
            released.set()

    owner = threading.Thread(target=release_after_stall)
    owner.start()
    try:
        reference = [{"input": "", "output": "1", "testtype": "stdin"}]
        env = code_environment(
            route, json.dumps(reference) if route == "lcb" else reference, code_verifier={"total_timeout_seconds": 0.1}
        )
        result = env.step("")
    finally:
        finished.set()
        owner.join()
        if not released.is_set():
            for _ in range(held):
                slots.release()
    assert result["reward"] == 0
    assert result["verification"].score is None
    assert result["metadata"]["verifyit_status"] == "infra_error"
    assert result["metadata"]["preparation_stage"] == "code_queue"
    assert result["metadata"]["error_type"] == "TimeoutError"
    assert result["metadata"]["sandbox_cleanup_status"] == 404


@pytest.mark.parametrize("route", ["lcb", "nemotron_ultra"])
def test_code_expensive_source_preparation_is_inside_total_budget(route, monkeypatch):
    import json
    import time
    import requests

    response = requests.Response()
    response.status_code = 404
    monkeypatch.setattr(requests, "delete", lambda *args, **kwargs: response)
    if route == "lcb":
        # APPS normalization creates and serializes hundreds of thousands of cases.
        reference = json.dumps({"inputs": [""] * 400_000, "outputs": ["1"] * 400_000})
        action = ""
    else:
        reference = [{"input": "", "output": "1", "testtype": "stdin"}]
        # Source reasoning extraction scans each unmatched opening delimiter.
        action = "<think>" * 30_000
    env = code_environment(route, reference, code_verifier={"total_timeout_seconds": 0.25})
    started = time.monotonic()
    result = env.step(action)
    elapsed = time.monotonic() - started
    assert result["reward"] == 0
    assert result["verification"].score is None
    assert result["metadata"]["verifyit_status"] == "infra_error"
    assert result["metadata"]["error_type"] == "TimeoutError"
    assert elapsed < 1.5


@pytest.mark.parametrize("prefix,expected", [("<thinking>", 0.0), ("prefix<think>", 0.0), ("line\n</think>", 1.0)])
def test_registered_code_route_applies_source_extraction_once(prefix, expected):
    from skyrl_gym.verification import RolloutEvidence

    action = prefix + "\n```python\nprint(1)\n```"
    reference = [{"input": "", "output": "1", "testtype": "stdin"}]
    results = []
    for enabled in (False, True):
        env = code_environment("nemotron_ultra", reference, verifyit_enabled=enabled)
        env.set_rollout_evidence(RolloutEvidence(response=action, metadata={"assistant_message": {"content": action}}))
        result = env.step(action)
        results.append(result)
        assert result["reward"] == expected
        assert result["metadata"]["result"] == ("pass" if expected else "missing_code")
        env.close()
    assert results[1]["metadata"]["execution_output"]["preparation"]["raw_response"] == action
