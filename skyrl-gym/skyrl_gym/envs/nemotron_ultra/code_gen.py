# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Competitive-code verifier adapter for the released NVIDIA row schema."""

from __future__ import annotations

import json
from typing import Any

from skyrl_gym.envs.lcb.livecodebench import (
    VerifierLimits,
    extract_code_from_model,
    lcb_check_correctness,
    normalize_lcb_ground_truth,
)

DEFAULT_PER_TEST_TIMEOUT_SECONDS = 10


def _has_reasoning_format_violation(text: str, assistant_message: dict[str, Any] | None) -> bool:
    """Match NVIDIA's malformed ``<think>``-tag penalty."""
    if "<think>" in text or "</think>" in text:
        return True
    reasoning = (assistant_message or {}).get("reasoning_content", "")
    return isinstance(reasoning, str) and (reasoning.count("<think>") > 1 or reasoning.count("</think>") > 1)


def grade_code(
    text: str,
    record: dict[str, Any],
    *,
    assistant_message: dict[str, Any] | None = None,
    timeout_seconds: int = DEFAULT_PER_TEST_TIMEOUT_SECONDS,
    reasoning_format_penalty: float = 0.0,
    limits: VerifierLimits | None = None,
) -> tuple[float, dict[str, Any]]:
    """Extract the final fenced program and run every NVIDIA LiveCodeBench test.

    ``limits`` bounds the verifier child; see ``VerifierLimits``.
    """
    code = extract_code_from_model(text)
    if not code:
        return 0.0, {"extracted_model_code": None, "result": "missing_code"}
    tests = json.loads(normalize_lcb_ground_truth(record["verifier_metadata"]["unit_tests"]))
    correct = lcb_check_correctness(tests, code, timeout=timeout_seconds, debug=False, limits=limits)
    format_violation = _has_reasoning_format_violation(text, assistant_message)
    reward = reasoning_format_penalty if format_violation else float(correct)
    return reward, {
        "extracted_model_code": code,
        "result": "pass" if correct else "failed_tests",
        "reasoning_format_violation_rate": float(format_violation),
        "difficulty": record.get("verifier_metadata", {}).get("difficulty"),
    }
