"""Dataset profiles and response normalization for the four public Pivot releases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from skyrl_gym.envs.nemotron_ultra.terminal_pivot import grade_terminal_pivot
from skyrl_gym.envs.nemotron_ultra.tool_call import grade_expected_action

NEMO_REFERENCE_COMMIT = "1c8261080bdc881b3e9b7f870e6418f160516991"
TERMINAL_AGENT = "terminus_judge_string_only_simple_agent"


@dataclass(frozen=True)
class PivotProfile:
    dataset_id: str
    source_agent: str
    agent: str
    word_threshold: float | None
    filename: str = "train.jsonl"


PIVOT_PROFILES = {
    "function_calling": PivotProfile(
        "nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1",
        "toolcall_schema_single_step_tool_use_with_argument_comparison_agent",
        "toolcall_schema_single_step_tool_use_with_argument_comparison_agent",
        0.1,
    ),
    "swe": PivotProfile(
        "nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1",
        "single_step_tool_use_with_argument_comparison_swe",
        "swe_pivot_single_step_tool_use_with_argument_comparison_agent",
        0.0,
    ),
    "terminal": PivotProfile(
        "nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1",
        TERMINAL_AGENT,
        TERMINAL_AGENT,
        None,
        "atcb_terminal_pivot_release_final_v2.jsonl",
    ),
    "conversational_tool_use": PivotProfile(
        "nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1",
        "single_step_tool_use_with_argument_comparison_agent",
        "single_step_tool_use_with_argument_comparison_agent",
        0.1,
    ),
}
TOOL_COMPARISON_THRESHOLDS = {
    profile.agent: profile.word_threshold for profile in PIVOT_PROFILES.values() if profile.word_threshold is not None
}


def pivot_assistant_message(response: dict[str, Any]) -> dict[str, Any]:
    """Normalize a Responses result or Chat Completions assistant message.

    Only assistant text and structured function calls count as actions. Textual
    tool markup must first be parsed by the model's tool parser.
    """
    items = response["output"] if "output" in response else [response]
    calls: list[dict[str, Any]] = []
    texts: list[str] = []
    for item in items:
        if item.get("type") == "function_call":
            calls.append({"type": "function", "function": {"name": item["name"], "arguments": item["arguments"]}})
            continue
        if item.get("type", "message") != "message" or item.get("role", "assistant") != "assistant":
            continue
        calls.extend(item.get("tool_calls") or [])
        content = item.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.extend(part["text"] for part in content if part.get("type") == "output_text")
    return {"role": "assistant", "content": "\n".join(texts) if texts else None, "tool_calls": calls}


def grade_pivot_response(
    dataset: str, record: dict[str, Any], response: dict[str, Any]
) -> tuple[float, dict[str, Any]]:
    """Grade one released row and a structured model response locally."""
    profile = PIVOT_PROFILES[dataset]
    if record["agent_ref"]["name"] != profile.source_agent:
        raise ValueError(f"Row agent does not match dataset {dataset!r}")
    message = pivot_assistant_message(response)
    if profile.word_threshold is None:
        return grade_terminal_pivot(message["content"] or "", record)
    reward, category = grade_expected_action(
        record["expected_action"], message, word_count_similarity_threshold=profile.word_threshold
    )
    return reward, {"category": category.value}


def reference_response(record: dict[str, Any]) -> dict[str, Any]:
    """Build an oracle response for a dataset wiring smoke check."""
    if "expected_answer" in record:
        return {"role": "assistant", "content": record["expected_answer"]}
    action = record["expected_action"]
    if action["type"] == "message":
        return {"role": "assistant", "content": action["content"]}
    calls = action["calls"] if action["type"] == "function_call_batch" else [action]
    return {"output": calls}
