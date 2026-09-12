# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Symbolic-plus-LLM math verifier ported from NVIDIA NeMo Gym."""

from __future__ import annotations

import contextlib
import multiprocessing as mp
from io import StringIO
from typing import Any, Protocol

from math_verify import grader
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig


class Judge(Protocol):
    def generate(self, messages: list[dict[str, str]], *, max_tokens: int = 8192) -> str: ...


_JUDGE_SYSTEM = """Please act as an impartial judge and evaluate the equivalence of the solutions given by two AI assistants to the mathematical problem displayed below. You will be given AI assistant A's answer and AI assistant B's answer. Your job is to evaluate whether assistant A's answer is equivalent to assistant B's answer.

Consider the mathematical equivalence of the AI assistants' answers above all other considerations. If the problem requests special formatting instructions, you may disregard any formatting considerations when evaluating the answers -- consider only mathematical equivalence.

After evaluating both answers for equivalence, you must output only one of the following choices as your final verdict with a label:

1.  The AI assistants' answers are equivalent: [[A=B]]
2.  The AI assistants' answers are different: [[A!=B]]

Example output: "My final verdict is different [[A!=B]]"."""
_JUDGE_PROMPT = "<|Problem|>\n{question}\n\n<|Start of Assistant A's Answer|>\n{first}\n<|End of Assistant A's Answer|>\n\n<|Start of Assistant B's Answer|>\n{second}\n<|End of Assistant B's Answer|>"


def _strip_delimiters(value: str) -> str:
    value = value.strip()
    if value.startswith(r"\(") and value.endswith(r"\)"):
        value = value[2:-2].strip()
    if value.startswith("$") and value.endswith("$") and len(value) > 1:
        value = value[1:-1].strip()
    return value


def _library_child(expected: str, generated: str, connection) -> None:
    verifier = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    try:
        expected = r"\boxed{" + _strip_delimiters(expected) + "}"
        with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
            score, extracted = verifier([expected], [generated])
        chosen = None
        if extracted is not None:
            golds, predictions = extracted
            chosen = next(
                (prediction for prediction in predictions if any(grader.verify(gold, prediction) for gold in golds)),
                predictions[0] if predictions else None,
            )
        connection.send((float(score), None if chosen is None else str(chosen)))
    except (Exception, TimeoutException):
        connection.send((0.0, None))
    finally:
        connection.close()


def symbolic_math_reward(expected: str, generated: str, *, timeout_seconds: float = 10.0) -> tuple[float, str | None]:
    # NVIDIA uses fork in production so each verification does not have to
    # import SymPy and math-verify into a fresh interpreter.
    context = mp.get_context("fork")
    receiving, sending = context.Pipe(duplex=False)
    process = context.Process(target=_library_child, args=(expected, generated, sending))
    process.start()
    sending.close()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(1.0)
        if process.is_alive():
            process.kill()
            process.join()
        receiving.close()
        return 0.0, None
    try:
        return receiving.recv() if process.exitcode == 0 and receiving.poll() else (0.0, None)
    finally:
        receiving.close()


def _judge_equal(judge: Judge, question: str, first: str, second: str) -> tuple[bool, str]:
    output = judge.generate(
        [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": _JUDGE_PROMPT.format(question=question, first=first, second=second)},
        ]
    )
    equal_at = output.find("[[A=B]]")
    unequal_at = output.find("[[A!=B]]")
    return equal_at >= 0 and (unequal_at < 0 or equal_at < unequal_at), output


def grade_math(
    text: str,
    record: dict[str, Any],
    *,
    judge: Judge | None,
    timeout_seconds: float = 10.0,
) -> tuple[float, dict[str, Any]]:
    library_reward, extracted = symbolic_math_reward(record["expected_answer"], text, timeout_seconds=timeout_seconds)
    diagnostics: dict[str, Any] = {"library_reward": library_reward, "extracted_answer": extracted}
    if library_reward > 0.5:
        return library_reward, diagnostics
    if judge is None:
        raise RuntimeError("math_with_judge_simple_agent requires the configured general judge when math-verify fails")
    candidate = extracted or text
    first_equal, first_output = _judge_equal(judge, record["question"], record["expected_answer"], candidate)
    diagnostics["judge_outputs"] = [first_output]
    if not first_equal:
        return 0.0, diagnostics
    second_equal, second_output = _judge_equal(judge, record["question"], candidate, record["expected_answer"])
    diagnostics["judge_outputs"].append(second_output)
    return float(second_equal), diagnostics
