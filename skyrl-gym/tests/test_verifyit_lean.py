"""Public source boundary tests use a local HTTP compiler-protocol fixture."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from skyrl_gym.envs.nemotron_ultra.lean import verify_lean_attempt
from skyrl_gym.envs.nemotron_ultra.sandbox import SandboxClient


@pytest.mark.parametrize(
    "output,reward,status",
    [
        ({"process_status": "completed", "stdout": "", "stderr": ""}, 1, "completed"),
        ({"process_status": "completed", "stdout": "error: proof failed", "stderr": ""}, 0, "failed"),
        ({"process_status": "completed", "stdout": "warning: uses sorry", "stderr": ""}, 0, "has_sorry"),
        ({"process_status": "timeout", "stdout": "", "stderr": ""}, 0, "timeout"),
        ({"process_status": "completed", "stdout": "", "stderr": "", "output_truncated": True}, None, None),
        ({"stdout": "", "stderr": ""}, None, None),
    ],
)
def test_lean_source_boundary_preserves_completion_and_fails_closed(output, reward, status):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps(output).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        args = {
            "generation": "```lean4\nby trivial\n```",
            "record": {"header": "import Mathlib\n", "formal_statement": "theorem example_task : True := "},
            "sandbox": SandboxClient(port=server.server_port),
            "verifyit_enabled": True,
        }
        if reward is None:
            with pytest.raises(RuntimeError, match="unavailable"):
                verify_lean_attempt(**args)
        else:
            actual, details, feedback = verify_lean_attempt(**args)
            assert actual == reward
            assert details["proof_status"] == status
            assert (feedback is None) == (reward == 1)
        assert requests[0]["language"] == "lean4"
        assert "theorem example_task" in requests[0]["generated_code"]
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
