"""Source-format predicates registered as client IFEval constraints."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any


def _line_regex(text: str, params: dict) -> tuple[bool, str]:
    text = json.loads(text)
    patterns = [re.compile(pattern) for pattern in params["verify_regex"]]
    matching_lines = sum(any(pattern.search(line) for pattern in patterns) for line in text.split("\n"))
    minimum = params["verify_min_matches"]
    passed = matching_lines >= minimum
    return passed, json.dumps({"matching_lines": matching_lines, "min_matches": minimum, "passed": passed})


def _markers(text: str, params: dict) -> tuple[bool, str]:
    text = json.loads(text)
    expected = params["expected_markers"]
    missing = [marker for marker in expected if marker not in text]
    expected_set = set(expected)
    spurious = [
        match.group(0)
        for pattern in params["patterns"]
        for match in re.finditer(pattern, text)
        if match.group(0) not in expected_set
    ]
    passed = not missing and not spurious
    return passed, json.dumps(
        {
            "expected": expected,
            "missing": missing,
            "spurious": spurious,
            "passed": passed,
        }
    )


def _grade_format_ifeval(text: str, verifier: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """Validate task parameters before asking IFEval to evaluate the candidate."""
    try:
        from verifyit.grade import Status, run
        from verifyit.modes.ifeval import CONSTRAINTS
        from verifyit.spec import Constraint, IfevalSpec, render_spec
    except ImportError:
        return 0.0, {
            "error_type": "verification_error",
            "error_message": "verifyit IFEval dependency unavailable",
        }

    try:
        if not isinstance(verifier, dict):
            raise ValueError("Format verifier must be an object")
        kind = verifier.get("type")
        if kind in {"regex", "inline_prose"}:
            name = "marin_skyrl:format_line_regex"
            params = {
                "verify_regex": verifier.get("verify_regex", []),
                "verify_min_matches": verifier.get("verify_min_matches", 1),
            }
            minimum = params["verify_min_matches"]
            if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum <= 0:
                raise ValueError("Match threshold must be a positive integer")
            patterns = params["verify_regex"]
            check = _line_regex
        elif kind == "string_match":
            name = "marin_skyrl:format_markers"
            params = {
                "expected_markers": verifier.get("expected_markers", []),
                "patterns": verifier.get("patterns", []),
            }
            if not isinstance(params["expected_markers"], list) or not all(
                isinstance(marker, str) and marker for marker in params["expected_markers"]
            ):
                raise ValueError("Expected markers must be strings")
            patterns = params["patterns"]
            check = _markers
        else:
            raise ValueError("Unsupported format verifier type")
        if not isinstance(patterns, list) or not all(isinstance(pattern, str) and pattern for pattern in patterns):
            raise ValueError("Regex patterns must be strings")
        if not patterns and (kind != "string_match" or not params["expected_markers"]):
            raise ValueError("Format policy must contain a nonempty constraint")
        for pattern in patterns:
            re.compile(pattern)
        existing = CONSTRAINTS.get(name)
        if existing is not None and existing is not check:
            raise ValueError("Format constraint registry collision")
        CONSTRAINTS[name] = check
    except (ValueError, TypeError, re.error):
        return 0.0, {
            "error_type": "schema_error",
            "error_message": "Invalid format verifier configuration",
        }

    try:
        with tempfile.TemporaryDirectory(prefix="skyrl-format-") as temporary:
            root = Path(temporary)
            spec = IfevalSpec(
                constraints=(Constraint(name, params),),
                output=str(root / "response.txt"),
            )
            (root / "verifier.toml").write_text(render_spec(spec))
            (root / "response.txt").write_text(json.dumps(text))
            verdict = run(root / "verifier.toml", root)
        if verdict.status is not Status.SCORED:
            return 0.0, {
                "error_type": "verification_error",
                "error_message": "Format verifier did not produce a scored result",
            }
        detail = json.loads(verdict.detail["constraints"][0]["detail"])
        return verdict.reward, detail
    except Exception:
        return 0.0, {
            "error_type": "verification_error",
            "error_message": "Format verification failed",
        }


def grade_format_verifyit(text: str, verifier: dict[str, Any], timeout: float = 5.0) -> tuple[float, dict[str, Any]]:
    """Bound the trusted IFEval regex evaluation using ScriptSpec process cleanup."""
    import math
    import shlex
    import sys

    try:
        from verifyit.grade import Status, run
        from verifyit.spec import ScriptSpec, render_spec

        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not 0 < timeout <= 300
            or not math.isfinite(timeout)
        ):
            return 0.0, {
                "error_type": "schema_error",
                "error_message": "Invalid format verification deadline",
            }
        with tempfile.TemporaryDirectory(prefix="skyrl-format-runtime-") as temporary:
            root = Path(temporary)
            (root / "input.json").write_text(json.dumps({"text": text, "verifier": verifier}))
            (root / "checker.sh").write_text(
                "#!/bin/sh\nexec "
                + shlex.quote(sys.executable)
                + " "
                + shlex.quote(str(Path(__file__).resolve()))
                + " --check "
                + shlex.quote(str(root / "input.json"))
                + "\n"
            )
            spec = ScriptSpec(path="checker.sh", timeout=timeout, verdict_file="format-verdict.json")
            (root / "verifier.toml").write_text(render_spec(spec))
            verdict = run(root / "verifier.toml", root)
        if verdict.status is not Status.SCORED:
            category = "schema_error" if verdict.status is Status.INVALID_TASK else "verification_error"
            return 0.0, {
                "error_type": category,
                "error_message": "Format verifier failed or exceeded its deadline",
            }
        return verdict.reward, verdict.detail["source_feedback"]
    except Exception:
        return 0.0, {
            "error_type": "verification_error",
            "error_message": "Format verification failed",
        }


def _main() -> None:
    import dataclasses
    import os
    import sys
    from enum import Enum

    def plain(value):
        if dataclasses.is_dataclass(value):
            return plain(dataclasses.asdict(value))
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(item) for item in value]
        return value

    payload = json.loads(Path(sys.argv[2]).read_text())
    calls = []
    active = {}

    def observe(frame, event, arg):
        if frame.f_code.co_name != "grade" or not frame.f_code.co_filename.endswith("/verifyit/modes/grade_ifeval.py"):
            return
        if event == "call":
            item = {
                "path": frame.f_code.co_filename,
                "spec": plain(frame.f_locals["spec"]),
            }
            calls.append(item)
            active[id(frame)] = item
        elif event == "return" and id(frame) in active:
            active.pop(id(frame))["verdict"] = plain(arg)

    sys.setprofile(observe)
    try:
        reward, feedback = _grade_format_ifeval(payload["text"], payload["verifier"])
    finally:
        sys.setprofile(None)
    category = feedback.get("error_type")
    status = (
        "invalid_task"
        if category == "schema_error"
        else "infra_error"
        if category == "verification_error"
        else "scored"
    )
    verdict = {
        "status": status,
        "reward": reward,
        "detail": {"source_feedback": feedback, "ifeval_calls": calls},
    }
    (Path(os.environ["VERIFYIT_LOGS_DIR"]) / "format-verdict.json").write_text(json.dumps(verdict, allow_nan=False))


if __name__ == "__main__":
    _main()
