"""Execute Lean's trusted compiler runtime under ScriptSpec ownership."""

from __future__ import annotations

import dataclasses
import json
import os
import shlex
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def compile_lean_verifyit(proof: str, sandbox, timeout_seconds: float, record: dict) -> tuple[float, str, dict]:
    from verifyit.grade import InvalidTask, Status, run
    from verifyit.spec import ScriptSpec, render_spec

    with TemporaryDirectory(prefix="skyrl-lean-") as directory:
        root = Path(directory)
        (root / "checker.py").write_text(Path(__file__).read_text())
        (root / "checker.sh").write_text(
            "#!/bin/sh\nexec " + shlex.quote(sys.executable) + " " + shlex.quote(str(root / "checker.py")) + "\n"
        )
        (root / "compiler-input.json").write_text(
            json.dumps(
                {
                    "proof": proof,
                    "host": sandbox.host,
                    "port": sandbox.port,
                    "timeout": timeout_seconds,
                    "task": {
                        "theorem_name": record["name"],
                        "header": record["header"],
                        "formal_statement": record["formal_statement"],
                    },
                },
                allow_nan=False,
            )
        )
        (root / "verifier.toml").write_text(
            render_spec(ScriptSpec(path="checker.sh", verdict_file="lean-result.json", timeout=timeout_seconds + 10))
        )
        verdict = run(root / "verifier.toml", root)
        if verdict.status == Status.INVALID_TASK:
            raise InvalidTask("Lean trusted task is invalid")
        if verdict.status != Status.SCORED:
            raise RuntimeError("Lean verification unavailable")
        return verdict.reward, verdict.detail["proof_status"], verdict.detail["compiler_output"]


def _compile(data: dict) -> dict:
    from verifyit.grade import Aggregation, InvalidTask, aggregate_rewards
    from verifyit.modes.grade_json_schema import grade_json_schema_candidate
    from skyrl_gym.envs.nemotron_ultra.sandbox import MAX_VERIFIER_OUTPUT_CHARACTERS, SandboxClient

    output = SandboxClient(host=data["host"], port=data["port"]).execute(
        data["proof"],
        language="lean4",
        timeout_seconds=data["timeout"],
        max_output_characters=MAX_VERIFIER_OUTPUT_CHARACTERS,
        lean_audit=data["task"],
    )
    audit = {
        "type": "object",
        "required": ["theorem_name", "constant_kind", "axioms", "process_status", "kernel_checked"],
        "properties": {
            "theorem_name": {"const": data["task"]["theorem_name"]},
            "constant_kind": {"type": "string"},
            "kernel_checked": {"type": "boolean"},
            "axioms": {"type": "array", "items": {"type": "string"}},
            "process_status": {"const": "completed"},
        },
    }
    if output.get("error_type") == "trusted_task_compilation":
        raise InvalidTask("Lean trusted task does not compile under the configured toolchain")
    protocol = {
        "type": "object",
        "required": ["process_status", "stdout", "stderr"],
        "properties": {
            "process_status": {"enum": ["completed", "failed", "timeout"]},
            "stdout": {"type": "string"},
            "stderr": {"type": "string"},
            "output_truncated": {"const": False},
            "truncated": {"const": False},
            "exit_code_types": {"additionalProperties": {"const": "int"}},
        },
        "if": {"properties": {"process_status": {"const": "completed"}}},
        "then": {"required": ["lean_audit"], "properties": {"lean_audit": audit}},
    }
    # Case normalization and Python type tags are data transformations.
    instance = dict(output)
    instance["stdout"] = output["stdout"].lower()
    instance["stderr"] = output["stderr"].lower()
    instance["exit_code_types"] = {
        key: type(output[key]).__name__ for key in ("returncode", "exit_code") if key in output
    }
    checked = grade_json_schema_candidate(protocol, instance)
    if not checked.reward:
        raise RuntimeError("Lean compiler or trusted audit protocol is unavailable")
    success = {
        "type": "object",
        "required": ["process_status", "lean_audit"],
        "properties": {
            "process_status": {"const": "completed"},
            "stdout": {"not": {"pattern": r"\berror\b"}},
            "stderr": {"not": {"pattern": r"\berror\b"}},
            "returncode": {"const": 0},
            "exit_code": {"const": 0},
            "exit_code_types": {"additionalProperties": {"const": "int"}},
            "lean_audit": {
                "properties": {
                    "constant_kind": {"const": "theorem"},
                    "kernel_checked": {"const": True},
                    "axioms": {"items": {"enum": ["propext", "Classical.choice", "Quot.sound"]}},
                },
            },
        },
    }
    compiler = grade_json_schema_candidate(success, instance)
    sorry_text = grade_json_schema_candidate(
        {"type": "string", "not": {"pattern": r"\bsorry\b"}},
        instance["stdout"] + "\n" + instance["stderr"],
    )
    sorry_axiom = grade_json_schema_candidate(
        {"type": "array", "not": {"contains": {"const": "sorryAx"}}},
        output.get("lean_audit", {}).get("axioms", []),
    )
    result = aggregate_rewards([compiler, sorry_text, sorry_axiom], expected_total=3, policy=Aggregation.ALL)
    # Status affects feedback only; each scoring decision is a primitive verdict.
    if output["process_status"] != "completed":
        status = output["process_status"]
    elif not sorry_text.reward or not sorry_axiom.reward:
        status = "has_sorry"
    else:
        status = "completed" if result.reward else "failed"
    verdict = dataclasses.asdict(result)
    verdict["detail"].update(proof_status=status, compiler_output=output)
    return verdict


if __name__ == "__main__":
    from verifyit.grade import InvalidTask

    try:
        data = json.loads((Path(os.environ["VERIFYIT_TESTS_DIR"]) / "compiler-input.json").read_text())
        verdict = _compile(data)
    except InvalidTask as error:
        verdict = {"status": "invalid_task", "reward": 0.0, "detail": {"reason": str(error)}}
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, ImportError):
        # Transport, dependency and malformed response errors fail closed in the
        # verifier process; no source format credit may survive these failures.
        verdict = {"status": "infra_error", "reward": 0.0, "detail": {"reason": "compiler_runtime_failure"}}
    (Path(os.environ["VERIFYIT_LOGS_DIR"]) / "lean-result.json").write_text(json.dumps(verdict, allow_nan=False))
