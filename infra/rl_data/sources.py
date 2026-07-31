"""Source-specific transforms for supported RLVR datasets."""

from __future__ import annotations

import ast
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from infra.rl_data.contracts import VerifierDataContract

RLVR_MATH_DATASET = "allenai/RLVR-MATH"
RLVR_IFEVAL_DATASET = "allenai/RLVR-IFeval"
DAPO_MATH_DATASET = "BytedTsinghua-SIA/DAPO-Math-17k"
MATH500_DATASET = "HuggingFaceH4/MATH-500"
DEEPSCALER_DATASET = "agentica-org/DeepScaleR-Preview-Dataset"
GSM8K_DATASET = "openai/gsm8k"
VERIFIABLE_CODE_DATASET = "open-r1/verifiable-coding-problems-python"
APPS_DATASET = "codeparrot/apps"
GPQA_DATASET = "Idavidrein/gpqa"
OPENSCIENCE_DATASET = "nvidia/OpenScience"
KTO_MIX_DATASET = "trl-lib/kto-mix-14k"
HH_RLHF_DATASET = "Anthropic/hh-rlhf"

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


# ---------------------------------------------------------------------------
# Math: DeepScaleR and GSM8K
# ---------------------------------------------------------------------------


def _prepare_deepscaler(example: Mapping[str, Any], index: int, contract: VerifierDataContract) -> PreparedRow:
    source = deepscaler_source()
    problem = example.get("problem")
    if not isinstance(problem, str):
        raise TypeError("DeepScaleR row problem must be a string.")
    return _math_row(problem, example.get("answer"), source, index, contract)


def _gsm8k_extract_answer(answer_text: str) -> str:
    marker = "####"
    idx = answer_text.rfind(marker)
    if idx < 0:
        raise ValueError("GSM8K answer field missing '####' delimiter.")
    return answer_text[idx + len(marker) :].strip().replace(",", "").replace("$", "")


def _prepare_gsm8k(example: Mapping[str, Any], index: int, contract: VerifierDataContract) -> PreparedRow:
    source = gsm8k_source()
    question = example.get("question")
    answer_text = example.get("answer")
    if not isinstance(question, str):
        raise TypeError("GSM8K row question must be a string.")
    if not isinstance(answer_text, str):
        raise TypeError("GSM8K row answer must be a string.")
    ground_truth = _gsm8k_extract_answer(answer_text)
    instruction = contract.prompt_instruction
    if not instruction:
        raise ValueError(f"{source.name} requires a verifier prompt instruction.")
    normalized = contract.normalize_ground_truth(ground_truth)
    return {
        "data_source": source.dataset_id,
        "prompt": [{"role": "user", "content": question + instruction}],
        "env_class": source.env_id,
        "reward_model": {"ground_truth": normalized},
        "extra_info": {"split": "train", "index": index},
    }


# ---------------------------------------------------------------------------
# Code: verifiable-coding-problems-python and APPS
# ---------------------------------------------------------------------------


def _prepare_verifiable_code(example: Mapping[str, Any], index: int, contract: VerifierDataContract) -> PreparedRow:
    source = verifiable_code_source()
    problem = example.get("problem_statement")
    verification_info = example.get("verification_info")
    if not isinstance(problem, str):
        raise TypeError("verifiable-coding-problems row problem_statement must be a string.")
    if verification_info is None:
        raise ValueError("verifiable-coding-problems row is missing verification_info.")
    if isinstance(verification_info, str):
        try:
            verification_info = ast.literal_eval(verification_info)
        except (ValueError, SyntaxError) as exc:
            raise ValueError("verification_info string could not be parsed.") from exc
    if not contract.prompt_instruction:
        raise ValueError(f"{source.name} requires a verifier prompt instruction.")
    normalized = contract.normalize_ground_truth(verification_info)
    return {
        "data_source": source.dataset_id,
        "prompt": [{"role": "user", "content": problem + contract.prompt_instruction}],
        "env_class": source.env_id,
        "reward_model": {"ground_truth": normalized},
        "extra_info": {"split": "train", "index": index},
    }


def _prepare_apps(example: Mapping[str, Any], index: int, contract: VerifierDataContract) -> PreparedRow:
    source = apps_source()
    problem = example.get("question")
    input_output = example.get("input_output")
    if not isinstance(problem, str):
        raise TypeError("APPS row question must be a string.")
    if input_output is None:
        raise ValueError("APPS row is missing input_output.")
    if isinstance(input_output, str):
        try:
            input_output = json.loads(input_output)
        except json.JSONDecodeError as exc:
            raise ValueError("APPS input_output string is not valid JSON.") from exc
    if not contract.prompt_instruction:
        raise ValueError(f"{source.name} requires a verifier prompt instruction.")
    normalized = contract.normalize_ground_truth(input_output)
    return {
        "data_source": source.dataset_id,
        "prompt": [{"role": "user", "content": problem + contract.prompt_instruction}],
        "env_class": source.env_id,
        "reward_model": {"ground_truth": normalized},
        "extra_info": {"split": "train", "index": index},
    }


# ---------------------------------------------------------------------------
# Science MCQ: GPQA and OpenScience
# ---------------------------------------------------------------------------

import random as _random


def _build_mcq_prompt(question: str, options: list[str], correct_idx: int, seed: int) -> tuple[str, str]:
    """Shuffle options deterministically and return (prompt, correct_letter)."""
    rng = _random.Random(seed)
    order = list(range(len(options)))
    rng.shuffle(order)
    labels = "ABCDEFGHIJKLMNOP"[: len(options)]
    lines = [f"{labels[pos]}) {options[order[pos]]}" for pos in range(len(options))]
    prompt = question.strip() + "\n" + "\n".join(lines)
    correct_letter = labels[order.index(correct_idx)]
    return prompt, correct_letter


def _prepare_gpqa(example: Mapping[str, Any], index: int, contract: VerifierDataContract) -> PreparedRow:
    source = gpqa_source()
    question = example.get("Question")
    correct = example.get("Correct Answer")
    if not isinstance(question, str) or not isinstance(correct, str):
        raise TypeError("GPQA row must have string Question and Correct Answer.")
    options = [correct]
    for key in ("Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"):
        val = example.get(key)
        if isinstance(val, str):
            options.append(val)
    if len(options) < 2:
        raise ValueError("GPQA row needs at least one incorrect answer.")
    prompt_text, correct_letter = _build_mcq_prompt(question, options, 0, index)
    instruction = contract.prompt_instruction
    if not instruction:
        raise ValueError(f"{source.name} requires a verifier prompt instruction.")
    normalized = contract.normalize_ground_truth(correct_letter)
    return {
        "data_source": source.dataset_id,
        "prompt": [{"role": "user", "content": prompt_text + instruction}],
        "env_class": source.env_id,
        "reward_model": {"ground_truth": normalized},
        "extra_info": {"split": "train", "index": index},
    }


def _prepare_openscience(example: Mapping[str, Any], index: int, contract: VerifierDataContract) -> PreparedRow:
    source = openscience_source()
    import re

    prompt_text = example.get("input")
    output = example.get("output")
    if not isinstance(prompt_text, str):
        raise TypeError("OpenScience row input must be a string.")
    if not isinstance(output, str):
        raise TypeError("OpenScience row output must be a string.")
    match = re.search(r"\\boxed\{([A-Da-d])\}", output)
    if not match:
        raise ValueError("OpenScience output missing \\boxed{X} answer letter.")
    ground_truth = match.group(1).upper()
    instruction = contract.prompt_instruction
    if not instruction:
        raise ValueError(f"{source.name} requires a verifier prompt instruction.")
    normalized = contract.normalize_ground_truth(ground_truth)
    return {
        "data_source": source.dataset_id,
        "prompt": [{"role": "user", "content": prompt_text + instruction}],
        "env_class": source.env_id,
        "reward_model": {"ground_truth": normalized},
        "extra_info": {"split": "train", "index": index},
    }


# ---------------------------------------------------------------------------
# Preference: KTO-mix-14k and HH-RLHF
# ---------------------------------------------------------------------------


def _last_assistant_content(messages: list[Mapping[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return str(message["content"])
    raise ValueError("Expected an assistant message.")


def _prepare_kto_mix(example: Mapping[str, Any], index: int, contract: VerifierDataContract) -> PreparedRow:
    source = kto_mix_source()
    prompt = example.get("prompt")
    completion = example.get("completion")
    label = example.get("label")
    if not isinstance(prompt, list) or not isinstance(completion, list):
        raise TypeError("KTO row prompt and completion must be lists.")
    if str(label).lower() != "true":
        raise ValueError("KTO row with label=False has no preferred completion; skipping.")
    chosen_text = _last_assistant_content(completion)
    prompt_text = _last_user_content(prompt)
    normalized = contract.normalize_ground_truth(chosen_text)
    return {
        "data_source": source.dataset_id,
        "prompt": [{"role": "user", "content": prompt_text}],
        "env_class": source.env_id,
        "reward_model": {"ground_truth": normalized},
        "extra_info": {"split": "train", "index": index},
    }


def _parse_hhrlhf(text: str) -> tuple[str, str]:
    """Extract (prompt, response) from an HH-RLHF dialogue string."""
    human_marker = "\n\nHuman:"
    assistant_marker = "\n\nAssistant:"
    h_idx = text.find(human_marker)
    a_idx = text.find(assistant_marker)
    if h_idx < 0 or a_idx < 0:
        raise ValueError("HH-RLHF text missing Human/Assistant markers.")
    prompt = text[h_idx + len(human_marker) : a_idx].strip()
    response = text[a_idx + len(assistant_marker) :].strip()
    return prompt, response


def _prepare_hh_rlhf(example: Mapping[str, Any], index: int, contract: VerifierDataContract) -> PreparedRow:
    source = hh_rlhf_source()
    chosen_raw = example.get("chosen")
    rejected_raw = example.get("rejected")
    if not isinstance(chosen_raw, str) or not isinstance(rejected_raw, str):
        raise TypeError("HH-RLHF chosen and rejected must be strings.")
    prompt_text, chosen_response = _parse_hhrlhf(chosen_raw)
    _, rejected_response = _parse_hhrlhf(rejected_raw)
    if chosen_response == rejected_response:
        raise ValueError("HH-RLHF chosen and rejected responses are identical.")
    normalized = contract.normalize_ground_truth(chosen_response)
    return {
        "data_source": source.dataset_id,
        "prompt": [{"role": "user", "content": prompt_text}],
        "env_class": source.env_id,
        "reward_model": {"ground_truth": normalized},
        "extra_info": {"split": "train", "index": index, "rejected": rejected_response},
    }


# ---------------------------------------------------------------------------
# Source factories
# ---------------------------------------------------------------------------


def deepscaler_source() -> Source:
    return Source("deepscaler", DEEPSCALER_DATASET, "aime", "train", False, "two_sided", _prepare_deepscaler)


def gsm8k_source() -> Source:
    return Source("gsm8k", GSM8K_DATASET, "gsm8k", "train", False, "two_sided", _prepare_gsm8k)


def verifiable_code_source() -> Source:
    return Source(
        "verifiable_code", VERIFIABLE_CODE_DATASET, "lcb", "train", False, "schema_only", _prepare_verifiable_code
    )


def apps_source() -> Source:
    return Source("apps", APPS_DATASET, "lcb", "train", False, "schema_only", _prepare_apps)


def gpqa_source() -> Source:
    return Source("gpqa", GPQA_DATASET, "mcq", "gpqa_diamond", False, "two_sided", _prepare_gpqa)


def openscience_source() -> Source:
    return Source("openscience", OPENSCIENCE_DATASET, "mcq", "train", True, "two_sided", _prepare_openscience)


def kto_mix_source() -> Source:
    return Source("kto_mix", KTO_MIX_DATASET, "preference", "train", False, "schema_only", _prepare_kto_mix)


def hh_rlhf_source() -> Source:
    return Source("hh_rlhf", HH_RLHF_DATASET, "preference", "train", False, "two_sided", _prepare_hh_rlhf)


SOURCES = {
    source.name: source
    for source in (
        rlvr_math_source(),
        dapo_math_source(),
        math500_source(),
        rlvr_ifeval_source(),
        deepscaler_source(),
        gsm8k_source(),
        verifiable_code_source(),
        apps_source(),
        gpqa_source(),
        openscience_source(),
        kto_mix_source(),
        hh_rlhf_source(),
    )
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
