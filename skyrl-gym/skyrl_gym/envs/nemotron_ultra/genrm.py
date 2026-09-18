# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cohort-relative circular GenRM rewards from NVIDIA's Ultra recipe."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any

from loguru import logger

from skyrl_gym.envs.nemotron_ultra.genrm_utils import (
    aggregate_scores,
    extract_output_text,
    generate_comparison_pairs,
    parse_genrm_output,
)
from skyrl_gym.envs.nemotron_ultra.judge import OpenAIJudge


def response_object(assistant_message: dict[str, Any], fallback_text: str) -> dict[str, Any]:
    """Rebuild the Response-API fields consumed by NVIDIA's GenRM utilities."""
    output = []
    reasoning = assistant_message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        output.append({"type": "reasoning", "summary": [{"type": "summary_text", "text": reasoning}]})
    content = assistant_message.get("content")
    answer = content if isinstance(content, str) else fallback_text
    output.append(
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": answer}],
        }
    )
    return {"output": output}


def grade_genrm_group(
    *,
    conversation_history: list[dict[str, Any]],
    response_objects: list[dict[str, Any]],
    principle: str,
    judge: OpenAIJudge,
    config: dict[str, Any],
) -> tuple[list[float], dict[str, float]]:
    cohort_started = time.perf_counter()
    observer = getattr(judge, "observer", None)
    fallback_count = 0
    fallback_lock = Lock()

    def observe(method: str, **kwargs: Any) -> None:
        if observer is None:
            return
        try:
            getattr(observer, method)(**kwargs)
        except Exception:
            logger.warning("GenRM telemetry observer failed", exc_info=True)

    default_score = float(config.get("default_score", 3.0))
    default_ranking = float(config.get("default_ranking", 3.5))
    pairs = generate_comparison_pairs("circular", len(response_objects))
    max_workers = int(config.get("max_concurrent_comparisons", len(pairs)))
    if max_workers < 1:
        raise ValueError("max_concurrent_comparisons must be at least 1")

    def compare(pair: tuple[int, int]) -> tuple[float, float, float]:
        nonlocal fallback_count
        first, second = pair
        metadata = {
            "response_1": extract_output_text(response_objects[first]),
            "response_2": extract_output_text(response_objects[second]),
            "principle": principle,
        }
        attempts = int(config.get("genrm_parse_retries", 1)) + 1
        for attempt in range(attempts):
            output: str | None = None
            try:
                output = judge.generate_response(
                    conversation_history,
                    metadata=metadata,
                    max_output_tokens=int(config.get("max_output_tokens", 24576)),
                    temperature=float(config.get("temperature", 1.0)),
                    top_p=float(config.get("top_p", 0.95)),
                )
                return parse_genrm_output(output, default_score, default_ranking, raise_on_fail=True)
            except Exception as error:
                logger.warning("GenRM comparison attempt {} failed: {}", attempt + 1, error)
                outcome = "parse_retry" if output is not None else "comparison_retry"
                if attempt + 1 < attempts:
                    observe("parse", outcome=outcome, attempt=attempt + 1)
                    time.sleep(float(config.get("genrm_parse_retry_sleep_seconds", 0.2)))
                else:
                    outcome = "parse_fallback" if output is not None else "comparison_fallback"
                    observe("parse", outcome=outcome, attempt=attempt + 1)
                    with fallback_lock:
                        fallback_count += 1
        return default_score, default_score, default_ranking

    def compare_observed(pair: tuple[int, int], submitted_at: float) -> tuple[float, float, float]:
        started = time.perf_counter()
        outcome = "success"
        try:
            return compare(pair)
        except BaseException:
            outcome = "failure"
            raise
        finally:
            observe(
                "executor",
                queue_seconds=started - submitted_at,
                duration_seconds=time.perf_counter() - started,
                outcome=outcome,
            )

    with ThreadPoolExecutor(max_workers=min(max_workers, len(pairs))) as executor:
        futures = [executor.submit(compare_observed, pair, time.perf_counter()) for pair in pairs]
        comparisons = [future.result() for future in futures]
    metadata = [(first, second, 0) for first, second in pairs]
    rewards, metrics, _, _ = aggregate_scores(
        comparison_results=comparisons,
        comparison_metadata=metadata,
        response_objs=response_objects,
        aggregator_method="simple_tiebreaker",
        default_score=default_score,
        reasoning_bonus=float(config.get("reasoning_bonus", 0.5)),
        answer_bonus=float(config.get("answer_bonus", 0.5)),
        top_percentile=float(config.get("top_percentile", 0.2)),
        group_reasoning_length_penalty_coeff=float(config.get("group_reasoning_length_penalty_coeff", 0.1)),
        group_answer_length_penalty_coeff=float(config["group_answer_length_penalty_coeff"]),
    )
    observe(
        "cohort",
        duration_seconds=time.perf_counter() - cohort_started,
        outcome="defaulted" if fallback_count else "success",
    )
    return rewards, metrics
