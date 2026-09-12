# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVIDIA's judge-backed abstention and MultiChallenge rewards."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
from typing import Any

from skyrl_gym.envs.nemotron_ultra.judge import OpenAIJudge

_MULTICHALLENGE_SYSTEM = (
    "You are a precise evaluator. Assess responses objectively based on the given criteria. "
    "Analyze the response carefully against the evaluation question."
)
_MULTICHALLENGE_PROMPT = """You are evaluating whether a model's response meets a specific criterion.

CONVERSATION CONTEXT:
{context}

MODEL'S FINAL RESPONSE:
{response}

EVALUATION QUESTION:
{question}

EXPECTED ANSWER: {pass_criteria}

Does the model's response satisfy the criterion described in the evaluation question?
Think step by step, then respond with exactly [[YES]] or [[NO]] on the last line."""


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL)
    return re.sub(r"^.*?</thinking>", "", text, flags=re.DOTALL).strip()


def _boxed_answer(text: str) -> str | None:
    index = text.rfind("\\boxed{")
    if index == -1:
        return None
    start = index + len("\\boxed{")
    depth = 1
    cursor = start
    while cursor < len(text) and depth:
        if text[cursor] == "{":
            depth += 1
        elif text[cursor] == "}":
            depth -= 1
        cursor += 1
    return text[start : cursor - 1] if depth == 0 else None


def _normalize_answer(value: str) -> str:
    value = re.sub(r"\b(a|an|the)\b", " ", value.lower().strip())
    return " ".join(re.sub(r"[^\w\s]", "", value).split())


def _parse_abstention_grade(text: str) -> str:
    cleaned = strip_thinking(text).strip()
    if cleaned in {"A", "B", "C"}:
        return cleaned
    last_line = cleaned.rsplit("\n", 1)[-1].strip()
    if last_line in {"A", "B", "C"}:
        return last_line
    for letter in ("A", "B", "C"):
        if letter in cleaned:
            return letter
    return "B"


def grade_abstention(response: str, record: dict[str, Any], judge: OpenAIJudge) -> tuple[float, dict[str, Any]]:
    response = strip_thinking(response)
    extracted = _boxed_answer(response) or response
    if _normalize_answer(extracted) == _normalize_answer("[IDK]"):
        return 0.5, {"verdict": "abstain", "extracted_answer": extracted}

    template = files(__package__).joinpath("abstention_prompt.txt").read_text()
    prompt = template.format(
        question=record.get("question", ""),
        target=record.get("answer", ""),
        predicted_answer=extracted,
    )
    judge_output = judge.generate([{"role": "user", "content": prompt}])
    grade = _parse_abstention_grade(judge_output)
    verdict = {"A": "correct", "B": "incorrect", "C": "abstain"}[grade]
    reward = {"A": 1.0, "B": 0.0, "C": 0.5}[grade]
    return reward, {
        "verdict": verdict,
        "extracted_answer": extracted,
        "judge_output": judge_output,
        "omniscience_index": 1.0 if grade == "A" else -1.0 if grade == "B" else 0.0,
    }


def _multichallenge_verdict(text: str) -> str:
    yes_position = text.rfind("[[YES]]")
    no_position = text.rfind("[[NO]]")
    if yes_position < 0 and no_position < 0:
        last_line = text.strip().rsplit("\n", 1)[-1].upper()
        return "YES" if "YES" in last_line else "NO"
    return "YES" if yes_position > no_position else "NO"


def grade_multichallenge(
    response: str,
    record: dict[str, Any],
    judge: OpenAIJudge,
) -> tuple[float, dict[str, Any]]:
    response = strip_thinking(response)
    rubric = record.get("rubric") or record.get("metadata", {}).get("rubric") or []
    context = record.get("context", "")

    def evaluate(item: dict[str, Any]) -> dict[str, Any]:
        pass_criteria = str(item.get("pass_criteria", "YES"))
        prompt = _MULTICHALLENGE_PROMPT.format(
            context=context,
            response=response,
            question=item.get("question", ""),
            pass_criteria=pass_criteria,
        )
        output = judge.generate(
            [
                {"role": "system", "content": _MULTICHALLENGE_SYSTEM},
                {"role": "user", "content": prompt},
            ]
        )
        verdict = _multichallenge_verdict(output)
        expected = pass_criteria.upper()
        score = float(verdict == expected) if expected in {"YES", "NO"} else float(verdict == "YES")
        return {"question": item.get("question", ""), "verdict": verdict, "score": score}

    with ThreadPoolExecutor(max_workers=max(1, len(rubric))) as executor:
        evaluations = list(executor.map(evaluate, rubric))
    reward = sum(item["score"] for item in evaluations) / len(evaluations) if evaluations else 0.0
    return reward, {
        "rubric_evaluations": evaluations,
        "num_passed": sum(item["score"] >= 0.99 for item in evaluations),
        "num_total": len(evaluations),
    }
