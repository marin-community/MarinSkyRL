"""Whole-cohort source parity through a real model protocol and ScriptSpec child."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from skyrl_gym.envs.nemotron_ultra.genrm import grade_genrm_group, response_object
from skyrl_gym.envs.nemotron_ultra.judge import OpenAIJudge


@pytest.fixture
def genrm_server():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            server.requests.append(request)
            metadata = request.get("metadata") or json.loads(
                request["messages"][-1]["content"]
            )
            first = metadata["response_1"]
            second = metadata["response_2"]
            raw = server.outputs[(first, second)]
            if self.path.endswith("/responses"):
                body = {
                    "status": server.status,
                    "output": [
                        {
                            "type": "message",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": raw}],
                        }
                    ],
                }
            else:
                body = {
                    "choices": [
                        {
                            "finish_reason": (
                                "stop" if server.status == "completed" else "length"
                            ),
                            "message": {"content": raw},
                        }
                    ]
                }
            encoded = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.requests, server.outputs, server.status = [], {}, "completed"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join()


def cohort(server, transport="responses_metadata", variant=0):
    responses = [
        response_object(
            {
                "role": "assistant",
                "content": text,
                "reasoning_content": "reason " * (i + 1),
            }
        )
        for i, text in enumerate(["first answer", "second long answer", "third answer"])
    ]
    scores = (
        [(5, 2, 1), (2, 4, 6), (4, 5, 4)]
        if variant == 0
        else [(3, 3, 1), (3, 3, 6), (3, 3, 3.5)]
    )
    texts = ["first answer", "second long answer", "third answer"]
    for i, (a, b, rank) in enumerate(scores):
        server.outputs[(texts[i], texts[(i + 1) % 3])] = json.dumps(
            {"overall": {"score_1": a, "score_2": b, "ranking": rank}}
        )
    return {
        "conversation_history": [{"role": "user", "content": "Question"}],
        "response_objects": responses,
        "principle": "Compare answer quality",
        "judge": OpenAIJudge(
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            model="genrm",
            response_transport=transport,
            timeout_seconds=5,
        ),
        "config": {
            "group_answer_length_penalty_coeff": 0.1 if variant == 2 else 0.0,
            "reasoning_bonus": 0.5 if variant == 2 else 0.0,
            "answer_bonus": 0.5 if variant == 2 else 0.0,
            "group_reasoning_length_penalty_coeff": 0.1 if variant == 2 else 0.0,
            "genrm_parse_retries": 0,
            "verifyit_timeout_seconds": 20,
        },
    }


@pytest.mark.parametrize("transport", ["responses_metadata", "chat_completions"])
@pytest.mark.parametrize("variant", [0, 1, 2])
def test_whole_cohort_retains_pairing_ties_and_length_shaping(
    genrm_server, transport, variant
):
    args = cohort(genrm_server, transport, variant)
    native = grade_genrm_group(**args)
    requests = list(genrm_server.requests)
    genrm_server.requests.clear()
    args["config"]["verifyit_enabled"] = True
    cutover = grade_genrm_group(**args)
    assert cutover == native
    assert sorted(
        genrm_server.requests, key=lambda x: json.dumps(x, sort_keys=True)
    ) == sorted(requests, key=lambda x: json.dumps(x, sort_keys=True))


@pytest.mark.parametrize(
    "bad", ["nan", "missing", "truncated", "duplicate", "multiple", "overflow"]
)
def test_one_bad_comparison_discards_entire_cohort(genrm_server, bad):
    args = cohort(genrm_server)
    key = next(iter(genrm_server.outputs))
    if bad == "nan":
        genrm_server.outputs[key] = '{"score_1":NaN,"score_2":5,"ranking":1}'
    elif bad == "multiple":
        genrm_server.outputs[key] = (
            '{"score_1":1,"score_2":1,"ranking":1} {"score_1":5,"score_2":5,"ranking":1}'
        )
    elif bad == "overflow":
        genrm_server.outputs[key] = (
            '{"score_1":5,"score_2":5,"ranking":1,"extra":1e999}'
        )
    elif bad == "duplicate":
        genrm_server.outputs[key] = '{"score_1":1,"score_1":5,"score_2":5,"ranking":1}'
    elif bad == "missing":
        genrm_server.outputs[key] = '{"score_1":5,"score_2":5}'
    else:
        genrm_server.outputs[key] = '{"score_1":5,"score_2":5,"ranking":1'
    args["config"]["verifyit_enabled"] = True
    with pytest.raises(RuntimeError, match="cohort verification failed"):
        grade_genrm_group(**args)


def test_failed_response_status_cannot_carry_positive_scores(genrm_server):
    args = cohort(genrm_server)
    genrm_server.status = "failed"
    assert grade_genrm_group(**args)[0]
    args["config"]["verifyit_enabled"] = True
    with pytest.raises(RuntimeError, match="cohort verification failed"):
        grade_genrm_group(**args)
