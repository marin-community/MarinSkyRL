"""Public source boundary tests use a local HTTP compiler-protocol fixture."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from verifyit.grade import InvalidTask

from skyrl_gym.envs.nemotron_ultra.lean import verify_lean_attempt
from skyrl_gym.envs.nemotron_ultra.sandbox import SandboxClient


@pytest.mark.parametrize(
    "output,reward,status",
    [
        ({"process_status": "completed", "stdout": "", "stderr": ""}, 1, "completed"),
        ({"process_status": "completed", "stdout": "", "stderr": "", "kernel_checked": False}, 0, "failed"),
        ({"process_status": "completed", "stdout": "", "stderr": "", "axioms": ["forged"]}, 0, "failed"),
        ({"process_status": "completed", "stdout": "", "stderr": "", "missing_audit": True}, None, None),
        ({"process_status": "completed", "stdout": "error: proof failed", "stderr": ""}, 0, "failed"),
        ({"process_status": "completed", "stdout": "warning: uses sorry", "stderr": ""}, 0, "has_sorry"),
        ({"process_status": "timeout", "stdout": "", "stderr": ""}, 0, "timeout"),
        ({"process_status": "completed", "stdout": "", "stderr": "", "output_truncated": True}, None, None),
        ({"stdout": "", "stderr": ""}, None, None),
        (
            {
                "process_status": "error",
                "stdout": "",
                "stderr": "unknown trusted type",
                "error_type": "trusted_task_compilation",
            },
            None,
            "invalid_task",
        ),
    ],
)
def test_lean_source_boundary_preserves_completion_and_fails_closed(output, reward, status):
    output = dict(output)
    if output.get("process_status") == "completed" and not output.pop("missing_audit", False):
        output["lean_audit"] = {
            "process_status": "completed",
            "theorem_name": "example_task",
            "constant_kind": "theorem",
            "kernel_checked": output.pop("kernel_checked", True),
            "axioms": output.pop("axioms", []),
        }
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
            "generation": "```lean4\nexact True.intro\n```",
            "record": {
                "name": "example_task",
                "header": "import Mathlib\n",
                "formal_statement": "theorem example_task : True := by\n",
            },
            "sandbox": SandboxClient(port=server.server_port),
            "verifyit_enabled": True,
        }
        if status == "invalid_task":
            with pytest.raises(InvalidTask, match="trusted task"):
                verify_lean_attempt(**args)
        elif reward is None:
            with pytest.raises(RuntimeError, match="unavailable"):
                verify_lean_attempt(**args)
        else:
            actual, details, feedback = verify_lean_attempt(**args)
            assert actual == reward
            assert details["proof_status"] == status
            assert (feedback is None) == (reward == 1)
        assert requests[0]["lean_audit"]["theorem_name"] == "example_task"
        assert requests[0]["language"] == "lean4"
        assert "theorem example_task" in requests[0]["generated_code"]
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


@pytest.mark.parametrize("generation", ["", "```lean4\nexact True.intro\n```"])
def test_invalid_lean_task_is_reported_before_candidate_or_compilation(generation):
    import skyrl_gym
    from omegaconf import OmegaConf
    from skyrl_gym.verification import VerificationStatus

    extras = {
        "extra_info": {
            "nemotron_ultra": {
                "route": "skyrl_gym",
                "agent": "math_formal_lean_refinement_agent",
                "record_json": json.dumps({"name": "target", "header": "", "formal_statement": ""}),
                "request_json": "{}",
            }
        }
    }
    env = skyrl_gym.make("nemotron_ultra", env_config=OmegaConf.create({"verifyit_enabled": True}), extras=extras)
    try:
        result = env.step(generation)
        assert result["reward"] == 0
        assert result["verification"].status is VerificationStatus.ERROR
        assert result["verification"].diagnostics["verifyit_status"] == "invalid_task"
    finally:
        env.close()
