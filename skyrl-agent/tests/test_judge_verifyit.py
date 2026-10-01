"""Source judge APIs over the real local OpenAI-compatible HTTP boundary."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import importlib
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "skyrl_agent"
for name, path in [
    ("skyrl_agent", ROOT),
    ("skyrl_agent.tasks", ROOT / "tasks"),
    ("skyrl_agent.tasks.verifiers", ROOT / "tasks/verifiers"),
]:
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module

stem = importlib.import_module("skyrl_agent.tasks.verifiers.web_search.stem_llm_judge")
compute_score = stem.compute_score

@pytest.fixture
def judge_server(monkeypatch):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            server.requests.append(request)
            body = json.dumps(
                {
                    "id": "test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": request["model"],
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": server.finish,
                            "message": {"role": "assistant", "content": server.reply},
                        }
                    ],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.requests, server.reply, server.finish = [], "", "stop"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("OPENAI_API_BASE", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "local-test")
    monkeypatch.setenv("STEM_LLM_JUDGE_URL", f"http://127.0.0.1:{server.server_port}")
    yield server
    server.shutdown()
    server.server_close()
    thread.join()


def invoke(route, enabled):
    return compute_score(
        "",
        r"\boxed{candidate}",
        "reference",
        {"question": "Question?"},
        verifyit_enabled=enabled,
    )


@pytest.mark.parametrize("route", ["stem"])
@pytest.mark.parametrize("answer,score", [("yes", 1.0), ("no", 0.0)])
def test_original_prompts_and_scores_survive_core_cutover(
    judge_server, route, answer, score
):
    judge_server.reply = f"Final Decision: {answer}"
    native = invoke(route, False)
    requests = list(judge_server.requests)
    judge_server.requests.clear()
    cutover = invoke(route, True)
    assert cutover == native == score
    assert judge_server.requests == requests


@pytest.mark.parametrize("route", ["stem"])
def test_truncated_positive_completion_never_scores(judge_server, route):
    judge_server.reply = (
        "Final Decision: Yes" if route == "stem" else '{"correct":"yes"}'
    )
    judge_server.finish = "length"
    if route == "stem":
        assert invoke(route, True) == 0.0
    else:
        with pytest.raises(ValueError, match="incomplete"):
            invoke(route, True)


def test_contradictory_stem_judge_loses_last_positive_credit(judge_server):
    judge_server.reply = "Final Decision: No\nFinal Decision: Yes"
    assert invoke("stem", False) == 1.0
    assert invoke("stem", True) == 0.0


@pytest.mark.parametrize(
    "text",
    [
        "Not Final Decision: Yes",
        "Final Decision: Yes because",
        "Final Decision: Yes\nUnfinished decision:",
    ],
)
def test_stem_requires_complete_final_decision_line(judge_server, text):
    judge_server.reply = text
    assert invoke("stem", False) == 1.0
    assert invoke("stem", True) == 0.0
