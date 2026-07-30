"""Source-specific transforms for supported RLVR datasets."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from infra.rl_data.contracts import VerifierDataContract

RLVR_MATH_DATASET = "allenai/RLVR-MATH"
RLVR_IFEVAL_DATASET = "allenai/RLVR-IFeval"
DAPO_MATH_DATASET = "BytedTsinghua-SIA/DAPO-Math-17k"
MATH500_DATASET = "HuggingFaceH4/MATH-500"

_DAPO_LEADING_MARKERS = ("The last line of your response", "Solve the following math problem")
_DAPO_TRAILING_MARKERS = ("Remember to put your answer", "The last line of your response")


PreparedRow = dict[str, Any]
PrepareRow = Callable[[Mapping[str, Any], int, VerifierDataContract], PreparedRow]


@dataclass(frozen=True)
class Source:
    """A downloadable RLVR source and its transform to the SkyRL row schema."""

    name: str
    dataset_id: str
    env_id: str
    split: str
    streaming: bool
    verification: str
    prepare_row: PrepareRow


def _last_user_content(messages: list[Mapping[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message["content"])
    raise ValueError("Expected a user message.")


def _strip_rlvr_math_fewshot(text: str) -> str:
    marker = "Question:"
    index = text.rfind(marker)
    return (text[index + len(marker) :] if index >= 0 else text).strip()


def _strip_dapo_boilerplate(text: str) -> str:
    body = text
    if any(marker in body[:200] for marker in _DAPO_LEADING_MARKERS):
        divider = body.find("\n\n")
        if divider >= 0:
            body = body[divider + 2 :]

    trailing_positions = [body.find(marker) for marker in _DAPO_TRAILING_MARKERS]
    trailing_positions = [position for position in trailing_positions if position >= 0]
    if trailing_positions:
        body = body[: min(trailing_positions)]
    return body.strip()


def _math_row(
    problem: str, ground_truth: Any, source: Source, index: int, contract: VerifierDataContract
) -> PreparedRow:
    instruction = contract.prompt_instruction
    if not instruction:
        raise ValueError(f"{source.name} requires a verifier prompt instruction.")
    if ground_truth is None:
        raise ValueError(f"{source.name} row is missing ground_truth.")

    normalized = contract.normalize_ground_truth(ground_truth)
    verified = contract.validate_example(normalized, f"Answer: \\boxed{{{normalized}}}", "")
    return {
        "data_source": source.dataset_id,
        "prompt": [{"role": "user", "content": problem + instruction}],
        "env_class": source.env_id,
        "reward_model": {"ground_truth": verified},
        "extra_info": {"split": "train", "index": index},
    }


def _prepare_rlvr_math(example: Mapping[str, Any], index: int, contract: VerifierDataContract) -> PreparedRow:
    source = rlvr_math_source()
    messages = example.get("messages")
    if not isinstance(messages, list):
        raise TypeError("RLVR-MATH row messages must be a list.")
    return _math_row(
        _strip_rlvr_math_fewshot(_last_user_content(messages)), example.get("ground_truth"), source, index, contract
    )


def _prepare_dapo_math(example: Mapping[str, Any], index: int, contract: VerifierDataContract) -> PreparedRow:
    source = dapo_math_source()
    messages = example.get("prompt")
    reward_model = example.get("reward_model")
    if not isinstance(messages, list) or not isinstance(reward_model, Mapping):
        raise TypeError("DAPO-Math row prompt messages and reward_model must be present.")
    return _math_row(
        _strip_dapo_boilerplate(_last_user_content(messages)),
        reward_model.get("ground_truth"),
        source,
        index,
        contract,
    )


def _prepare_math500(example: Mapping[str, Any], index: int, contract: VerifierDataContract) -> PreparedRow:
    problem = example.get("problem")
    if not isinstance(problem, str):
        raise TypeError("MATH-500 row problem must be a string.")
    return _math_row(problem, example.get("answer"), math500_source(), index, contract)


def _prepare_rlvr_ifeval(example: Mapping[str, Any], index: int, contract: VerifierDataContract) -> PreparedRow:
    messages = example.get("messages")
    if not isinstance(messages, list):
        raise TypeError("RLVR-IFeval row messages must be a list.")
    normalized = contract.normalize_ground_truth(example.get("ground_truth"))
    return {
        "data_source": RLVR_IFEVAL_DATASET,
        "prompt": [{"role": "user", "content": _last_user_content(messages)}],
        "env_class": "ifeval",
        "reward_model": {"ground_truth": normalized},
        "extra_info": {"split": "train", "index": index, "constraint_type": example.get("constraint_type")},
    }


def rlvr_math_source() -> Source:
    return Source("rlvr_math", RLVR_MATH_DATASET, "aime", "train", False, "two_sided", _prepare_rlvr_math)


def dapo_math_source() -> Source:
    return Source("dapo_math", DAPO_MATH_DATASET, "aime", "train", True, "two_sided", _prepare_dapo_math)


def math500_source() -> Source:
    return Source("math500", MATH500_DATASET, "aime", "test", False, "two_sided", _prepare_math500)


def rlvr_ifeval_source() -> Source:
    return Source("rlvr_ifeval", RLVR_IFEVAL_DATASET, "ifeval", "train", False, "schema_only", _prepare_rlvr_ifeval)


SOURCES = {
    source.name: source for source in (rlvr_math_source(), dapo_math_source(), math500_source(), rlvr_ifeval_source())
}


def source_by_name(name: str) -> Source:
    """Return a named supported source or explain the supported names."""
    try:
        return SOURCES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown RLVR source {name!r}; choose from {sorted(SOURCES)}.") from exc


def load_source_rows(source: Source, revision: str):
    """Load source rows only when the CLI is invoked, keeping core tests offline."""
    import datasets

    return datasets.load_dataset(source.dataset_id, split=source.split, revision=revision, streaming=source.streaming)
