"""Execute Lean's trusted compiler runtime under ScriptSpec ownership."""

from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def compile_lean_verifyit(proof: str, sandbox, timeout_seconds: float) -> tuple[str, dict]:
    from verifyit.grade import Status, run
    from verifyit.spec import ScriptSpec, render_spec

    with TemporaryDirectory(prefix="skyrl-lean-") as directory:
        root = Path(directory)
        (root / "checker.py").write_text(Path(__file__).read_text())
        (root / "checker.sh").write_text(
            "#!/bin/sh\nexec " + shlex.quote(sys.executable) + " " + shlex.quote(str(root / "checker.py")) + "\n"
        )
        (root / "compiler-input.json").write_text(
            json.dumps(
                {"proof": proof, "host": sandbox.host, "port": sandbox.port, "timeout": timeout_seconds},
                allow_nan=False,
            )
        )
        (root / "verifier.toml").write_text(
            render_spec(ScriptSpec(path="checker.sh", verdict_file="lean-result.json", timeout=timeout_seconds + 10))
        )
        verdict = run(root / "verifier.toml", root)
        if verdict.status != Status.SCORED:
            raise RuntimeError("Lean verification unavailable")
        return verdict.detail["proof_status"], verdict.detail["compiler_output"]


def _compile(data: dict) -> dict:
    from skyrl_gym.envs.nemotron_ultra.lean_proof_utils import determine_proof_status
    from skyrl_gym.envs.nemotron_ultra.sandbox import MAX_VERIFIER_OUTPUT_CHARACTERS, SandboxClient

    output = SandboxClient(host=data["host"], port=data["port"]).execute(
        data["proof"],
        language="lean4",
        timeout_seconds=data["timeout"],
        max_output_characters=MAX_VERIFIER_OUTPUT_CHARACTERS,
    )
    status = determine_proof_status(output)
    if status not in {"completed", "failed", "has_sorry", "timeout"}:
        return {"status": "infra_error", "reward": 0.0, "detail": {"reason": "incomplete_compiler_result"}}
    return {
        "status": "scored",
        "reward": float(status == "completed"),
        "detail": {"proof_status": status, "compiler_output": output},
    }


if __name__ == "__main__":
    try:
        data = json.loads((Path(os.environ["VERIFYIT_TESTS_DIR"]) / "compiler-input.json").read_text())
        verdict = _compile(data)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, ImportError):
        # Transport, dependency and malformed response errors fail closed in the
        # verifier process; no source format credit may survive these failures.
        verdict = {"status": "infra_error", "reward": 0.0, "detail": {"reason": "compiler_runtime_failure"}}
    (Path(os.environ["VERIFYIT_LOGS_DIR"]) / "lean-result.json").write_text(json.dumps(verdict, allow_nan=False))
