"""Math and symmetric judge contracts at the actual source boundary."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from skyrl_gym.envs.nemotron_ultra.judge import OpenAIJudge
from skyrl_gym.envs.nemotron_ultra.math_judge_verifyit import grade_math_verifyit
from skyrl_gym.envs.nemotron_ultra.math_with_judge import grade_math


@pytest.fixture
def judge_server():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            server.requests.append(body)
            if hasattr(server, "entered"):
                server.entered.set()
                server.release.wait(timeout=10)
            index = min(len(server.requests) - 1, len(server.replies) - 1)
            response = {
                "id": "source-judge",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": server.finish_reason,
                        "message": {
                            "role": "assistant",
                            "content": server.replies[index],
                        },
                    }
                ],
            }
            encoded = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            try:
                self.wfile.write(encoded)
            except BrokenPipeError:
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.requests = []
    server.replies = ["[[A=B]]"]
    server.finish_reason = "stop"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    judge = OpenAIJudge(
        base_url=f"http://127.0.0.1:{server.server_port}/v1",
        model="recorded-test-judge",
        timeout_seconds=2,
    )
    try:
        yield server, judge
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize(
    "candidate,expected,question,score",
    [
        (r"\boxed{2}", "2", "1+1?", 1.0),
        (r"\boxed{xy}", "xy", "Product?", 1.0),
        (r"\boxed{x-y}", "x-y", "Difference?", 1.0),
        (r"\boxed{\sin x}", r"\sin x", "Function?", 1.0),
        (r"\boxed{cat}", "cat", "Symbolic letters?", 1.0),
        (r"\boxed{x^2+3}", "x^2", "Find an antiderivative", 1.0),
        (r"\boxed{2x}", "x", "Find an antiderivative", 0.0),
        ("<think>unfinished", "2", "1+1?", 0.0),
    ],
)
def test_actual_source_math_before_after(candidate, expected, question, score, judge_server):
    server, judge = judge_server
    server.replies = ["[[A!=B]]"]
    record = {"expected_answer": expected, "question": question}
    native_score, native_detail = grade_math(candidate, record, judge=judge)
    server.requests.clear()
    cutover_score, cutover_detail = grade_math_verifyit(candidate, record, judge=judge)
    assert native_score == cutover_score == score
    assert native_detail == {
        key: value for key, value in cutover_detail.items() if key not in {"preparation", "verifyit_verdict"}
    }


@pytest.mark.parametrize(
    "replies,score",
    [
        (["reasoning\n[[A=B]]", "[[A=B]]"], 1.0),
        (["[[A=B]]", "[[A!=B]]"], 0.0),
        (["[[A!=B]]"], 0.0),
    ],
)
def test_source_symmetric_judge_before_after(judge_server, replies, score):
    server, judge = judge_server
    server.replies = replies
    record = {"question": "Which animal?", "expected_answer": "cat"}
    native_score, native_detail = grade_math("It is a feline.", record, judge=judge)
    native_requests = list(server.requests)
    native_detail["judge_outputs"] = [reply.splitlines()[-1] for reply in native_detail["judge_outputs"]]
    server.requests.clear()
    cutover_score, cutover_detail = grade_math_verifyit("It is a feline.", record, judge=judge)
    assert native_score == cutover_score == score
    assert native_detail == {
        key: value for key, value in cutover_detail.items() if key not in {"preparation", "verifyit_verdict"}
    }
    assert server.requests == native_requests


def test_second_judge_failure_discards_first_positive(judge_server):
    server, judge = judge_server
    server.replies = ["[[A=B]]", "ambiguous [[A=B]] and [[A!=B]]"]
    reward, detail = grade_math_verifyit(
        "a feline", {"question": "Which animal?", "expected_answer": "cat"}, judge=judge
    )
    assert reward == 0.0
    assert detail["error_type"] == "verification_error"


def test_malformed_reference_is_not_judged(judge_server):
    server, judge = judge_server
    reward, detail = grade_math_verifyit("cat", {"question": "Which animal?", "expected_answer": ""}, judge=judge)
    assert reward == 0.0
    assert detail["error_type"] == "schema_error"
    assert server.requests == []


@pytest.mark.parametrize("label,score", [("[[A=B]]", 1.0), ("[[A!=B]]", 0.0)])
def test_trusted_prose_reference_routes_to_judge_before_after(judge_server, label, score):
    server, judge = judge_server
    server.replies = [label]
    record = {
        "question": "Which strategy?",
        "expected_answer": "Always choose the next box until reaching Box 8.",
    }
    native_score, native_detail = grade_math(r"\boxed{977}", record, judge=judge)
    native_requests = list(server.requests)
    server.requests.clear()
    reward, detail = grade_math_verifyit(r"\boxed{977}", record, judge=judge)
    assert native_score == reward == score
    assert native_detail == {
        key: value for key, value in detail.items() if key not in {"preparation", "verifyit_verdict"}
    }
    assert server.requests == native_requests


@pytest.mark.parametrize("candidate", [r"\boxed{2}", "I think the answer is two.", "<think>unfinished"])
def test_explicit_symbolic_reference_cannot_fall_back_to_positive_judge(judge_server, candidate):
    server, judge = judge_server
    server.replies = ["[[A=B]]"]
    reward, detail = grade_math_verifyit(
        candidate,
        {
            "question": "Compute",
            "expected_answer": r"\unknownmacro{???}",
            "math_reference_kind": "symbolic",
        },
        judge=judge,
    )
    assert reward == 0.0
    assert detail["error_type"] == "schema_error"
    assert server.requests == []


def test_symbolic_infrastructure_failure_cannot_fall_back_to_positive_judge(judge_server, tmp_path, monkeypatch):
    import dataclasses
    from verifyit.modes import grade_math as primitive
    from skyrl_gym.envs.nemotron_ultra.math_judge_verifyit import _evaluate

    def broken_backend(*args, **kwargs):
        raise RuntimeError("symbolic backend failed")

    server, judge = judge_server
    server.replies = ["[[A=B]]"]
    monkeypatch.setattr(primitive, "_verify", broken_backend)
    result = _evaluate(
        {
            "text": r"\boxed{2}",
            "record": {"question": "Compute", "expected_answer": "2"},
            "judge": dataclasses.asdict(judge),
        },
        tmp_path,
    )
    assert (result["status"], result["reward"]) == ("infra_error", 0.0)
    assert server.requests == []


@pytest.mark.parametrize(
    "candidate,replies,score",
    [
        (r"\boxed{1}", ["[[A=B]]"], 1.0),
        (r"\boxed{1}", ["[[A!=B]]"], 0.0),
        ("<think>unfinished", [], 0.0),
    ],
)
def test_valid_typographic_reference_retains_native_fallback(judge_server, candidate, replies, score):
    server, judge = judge_server
    server.replies = replies
    record = {
        "question": "Evaluate integral",
        "expected_answer": r"\displaystyle \frac{\pi}{4}\,\arctan\!\Bigl(\sqrt{\frac{\sqrt{2}-1}{2}}\Bigr)",
    }
    native_score, native_detail = grade_math(candidate, record, judge=judge)
    native_requests = list(server.requests)
    server.requests.clear()
    reward, detail = grade_math_verifyit(candidate, record, judge=judge)
    assert native_score == reward == score
    assert native_detail == {
        key: value for key, value in detail.items() if key not in {"preparation", "verifyit_verdict"}
    }
    assert server.requests == native_requests


@pytest.mark.parametrize("expected", [r"\frac{1}{2", r"\Bigl(2", r"\unknownmacro{???}"])
def test_malformed_math_reference_is_invalid_before_any_candidate_gate(judge_server, expected):
    server, judge = judge_server
    server.replies = ["[[A=B]]"]
    reward, detail = grade_math_verifyit(
        "The correct answer is two.",
        {
            "question": "Compute",
            "expected_answer": expected,
            "math_reference_kind": "symbolic",
        },
        judge=judge,
    )
    assert reward == 0.0
    assert detail["error_type"] == "schema_error"
    assert server.requests == []


@pytest.mark.parametrize(
    "kind,reference",
    [
        ("symbolic", "Choose a box (or none)."),
        ("symbolic", r"\unknownmacro{???}"),
        ("semantic", ""),
        ("semantic", "answer\x00suffix"),
        ("semantic", "\ud800"),
        ("unknown", "answer"),
        (None, "answer"),
    ],
)
def test_declared_reference_contract_rejects_invalid_task_before_positive_judge(kind, reference, judge_server):
    server, judge = judge_server
    server.replies = ["[[A=B]]"]
    reward, detail = grade_math_verifyit(
        "A possible answer.",
        {
            "question": "Question?",
            "expected_answer": reference,
            "math_reference_kind": kind,
        },
        judge=judge,
    )
    assert reward == 0.0
    assert detail["error_type"] == "schema_error"
    assert server.requests == []


def test_declared_symbolic_reference_still_uses_math_without_judge(judge_server):
    server, judge = judge_server
    reward, detail = grade_math_verifyit(
        r"\boxed{2}",
        {
            "question": "Compute 1+1",
            "expected_answer": "2",
            "math_reference_kind": "symbolic",
        },
        judge=judge,
    )
    assert reward == 1.0
    assert detail["library_reward"] == 1.0
    assert server.requests == []


@pytest.mark.parametrize(
    "reference",
    [
        "Choose a box (unless it is empty).\nThen stop — there is no unique answer.",
        r"\unknownmacro{???}",
    ],
)
@pytest.mark.parametrize("label,score", [("[[A=B]]", 1.0), ("[[A!=B]]", 0.0)])
def test_native_hybrid_reference_policy_requires_actual_symmetric_judge(reference, label, score, judge_server):
    server, judge = judge_server
    server.replies = [label]
    record = {
        "question": "Compare the supplied answer with the reference.",
        "expected_answer": reference,
    }
    candidate = "The candidate answer is the supplied reference interpretation."
    native_score, native_detail = grade_math(candidate, record, judge=judge)
    native_requests = list(server.requests)
    server.requests.clear()
    reward, detail = grade_math_verifyit(candidate, record, judge=judge)
    assert reward == native_score == score
    assert {
        key: value for key, value in detail.items() if key not in {"preparation", "verifyit_verdict"}
    } == native_detail
    assert server.requests == native_requests
    assert len(server.requests) == (2 if score else 1)


@pytest.mark.parametrize("kind", [None, "symbolic"])
@pytest.mark.parametrize("failure", [RuntimeError, OSError, MemoryError])
def test_unexpected_reference_parser_failure_is_unscored(kind, failure, monkeypatch, tmp_path, judge_server):
    import dataclasses
    import math_verify
    from skyrl_gym.envs.nemotron_ultra.math_judge_verifyit import _evaluate

    def failed_parse(*args, **kwargs):
        raise failure("Reference parser failed")

    monkeypatch.setattr(math_verify, "parse", failed_parse)
    server, judge = judge_server
    server.replies = ["[[A=B]]"]
    record = {"question": "Explain the result.", "expected_answer": "A textual answer."}
    if kind is not None:
        record["math_reference_kind"] = kind
    verdict = _evaluate(
        {
            "record": record,
            "text": "The supplied textual answer.",
            "judge": dataclasses.asdict(judge),
        },
        tmp_path,
    )
    assert verdict["status"] == "infra_error"
    assert verdict["reward"] == 0.0
    assert server.requests == []


@pytest.mark.parametrize(
    "replies,missing,reward", [(["[[A!=B]]", "broken provider"], 1, 0.0), (["[[A=B]]", "[[A=B]]"], 0, 1.0)]
)
def test_core_composition_preserves_source_short_circuit_and_raw_inputs(judge_server, replies, missing, reward):
    server, judge = judge_server
    server.replies = replies
    text = "<think>private reasoning</think>It is a feline."
    record = {"question": "Which animal?", "expected_answer": "cat"}
    score, detail = grade_math_verifyit(text, record, judge=judge)
    verdict = detail["verifyit_verdict"]
    assert score == reward
    assert len(server.requests) == 2 - missing
    assert verdict["detail"]["total"] == 2
    assert verdict["detail"]["missing"] == missing
    assert "raw" not in detail["preparation"]
    assert len(detail["preparation"]["raw_sha256"]) == 64
    assert detail["preparation"]["text"] == "It is a feline."


def test_structure_math_judge_detaches_nested_raw_record():
    from skyrl_gym.envs.nemotron_ultra.math_judge_verifyit import structure_math_judge

    record = {"question": "Compute", "expected_answer": "2", "metadata": {"source": ["original"]}}
    captured = structure_math_judge("answer", record)
    record["metadata"]["source"].append("later")
    assert captured.record["metadata"]["source"] == ["original"]
    assert captured.text == "answer"


def test_invalid_reference_diagnostics_do_not_expose_trusted_text(judge_server):
    server, judge = judge_server
    secret = "trusted_private_answer_83f9"
    reward, detail = grade_math_verifyit(
        "response", {"question": "Question", "expected_answer": secret, "math_reference_kind": "invalid"}, judge=judge
    )
    assert reward == 0.0
    assert detail["error_type"] == "schema_error"
    assert secret not in json.dumps(detail)
    assert not server.requests


@pytest.mark.parametrize("agent", ["ns_tools_simple_agent", "math_with_judge_simple_agent"])
def test_framework_judge_diagnostics_keep_trusted_reference_private(agent, judge_server):
    import dataclasses
    import skyrl_gym

    server, judge = judge_server
    secret = "PRIVATE_EXPECTED_CANARY"
    server.replies = [f"The trusted reference is {secret}.\n[[A=B]]"]
    ultra = {
        "route": "skyrl_gym",
        "agent": agent,
        "record_json": json.dumps({"question": "Compare the answer", "expected_answer": secret}),
        "request_json": "{}",
    }
    for enabled in (False, True):
        server.requests.clear()
        env = skyrl_gym.make(
            "nemotron_ultra",
            env_config={"verifyit_enabled": enabled, "judges": {"general": dataclasses.asdict(judge)}},
            extras={"extra_info": {"nemotron_ultra": ultra}},
        )
        try:
            result = env.step("A candidate interpretation.")
        finally:
            env.close()
        assert result["reward"] == 1.0
        assert len(server.requests) == 2
        if enabled:
            assert secret not in json.dumps(result, default=str)
            assert result["metadata"]["judge_outputs"] == ["[[A=B]]", "[[A=B]]"]
        else:
            assert secret in json.dumps(result, default=str)


def test_malformed_worker_result_is_unscored(monkeypatch):
    from skyrl_gym.envs.nemotron_ultra import math_judge_verifyit as adapter

    monkeypatch.setattr(adapter, "call_bounded", lambda *args, **kwargs: None)
    reward, detail = adapter.grade_math_verifyit("2", {"expected_answer": "2", "question": "1+1?"}, judge=None)
    assert reward == 0
    assert detail["verifyit_verdict"]["status"] == "infra_error"


@pytest.mark.parametrize("value,expected", [(float("nan"), 0), (float("inf"), 0), (True, 0), (2, 0), (1, 1)])
def test_worker_reward_validation(value, expected, monkeypatch):
    from verifyit.grade import Reward, Status
    from skyrl_gym.envs.nemotron_ultra import math_judge_verifyit as adapter

    monkeypatch.setattr(
        adapter,
        "call_bounded",
        lambda *args, **kwargs: Reward(value, Status.SCORED, {"source_feedback": {"library_reward": 1}}),
    )
    reward, detail = adapter.grade_math_verifyit("2", {"expected_answer": "2", "question": "1+1?"}, judge=None)
    assert reward == expected
    assert detail["verifyit_verdict"]["status"] == ("scored" if expected else "infra_error")


def test_deserialization_uses_total_deadline(monkeypatch):
    from skyrl_gym.envs.nemotron_ultra import math_judge_verifyit as adapter

    clock = [10.0]
    loads = adapter.json.loads

    def slow_decode(payload):
        clock[0] += 2
        return loads(payload)

    monkeypatch.setattr(adapter.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(adapter.json, "loads", slow_decode)
    reward, detail = adapter.grade_math_verifyit(
        "2", {"expected_answer": "2", "question": "1+1?"}, judge=None, timeout_seconds=1
    )
    assert reward == 0
    assert detail["verifyit_verdict"]["status"] == "infra_error"


def test_worker_timeout_cleans_process_group_and_parent_directory(monkeypatch, tmp_path, judge_server):
    import dataclasses
    import os
    from verifyit.execution import command
    from skyrl_gym.envs.nemotron_ultra import math_judge_verifyit as adapter

    processes = []
    popen = command.subprocess.Popen

    def observe_process(*args, **kwargs):
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(command.subprocess, "Popen", observe_process)
    monkeypatch.setattr(adapter.tempfile, "tempdir", str(tmp_path))
    server, judge = judge_server
    judge = dataclasses.replace(judge, timeout_seconds=20)
    server.entered = threading.Event()
    server.release = threading.Event()
    try:
        reward, detail = adapter.grade_math_verifyit(
            "A dog.", {"expected_answer": "A cat.", "question": "Which animal?"}, judge=judge, timeout_seconds=5
        )
    finally:
        server.release.set()
    assert server.entered.is_set()
    assert reward == 0
    assert detail["verifyit_verdict"]["status"] == "infra_error"
    assert len(processes) == 1
    assert processes[0].poll() is not None
    with pytest.raises(ProcessLookupError):
        os.killpg(processes[0].pid, 0)
    assert list(tmp_path.iterdir()) == []


def test_actual_worker_exception_discards_credit():
    from verifyit.execution.worker import call_bounded
    from skyrl_gym.envs.nemotron_ultra.math_judge_verifyit import _evaluate_bounded
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        result = call_bounded(
            _evaluate_bounded,
            {
                "text": "A dog.",
                "record": {"expected_answer": "A cat.", "question": "Which animal?"},
                "judge": {"private": "secret"},
            },
            directory,
            timeout=10,
        )
    assert result.reward == 0
    assert result.status.value == "infra_error"


def test_public_projection_cannot_return_late_credit(monkeypatch):
    from verifyit.grade import Reward, Status
    from skyrl_gym.envs.nemotron_ultra import math_judge_verifyit as adapter

    clock = [10.0]
    sha256 = adapter.hashlib.sha256

    def slow_digest(payload):
        clock[0] += 2
        return sha256(payload)

    monkeypatch.setattr(adapter.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(adapter.hashlib, "sha256", slow_digest)
    monkeypatch.setattr(
        adapter,
        "call_bounded",
        lambda *args, **kwargs: Reward(1, Status.SCORED, {"source_feedback": {"library_reward": 1}}),
    )
    reward, detail = adapter.grade_math_verifyit(
        "2", {"expected_answer": "2", "question": "1+1?"}, judge=None, timeout_seconds=1
    )
    assert reward == 0
    assert detail["verifyit_verdict"]["status"] == "infra_error"
