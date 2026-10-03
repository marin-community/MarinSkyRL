"""Final-answer extraction shared by Ultra graders and model transport."""

import re

from skyrl_gym.envs.aime.utils import last_boxed_only_string, remove_boxed
from skyrl_gym.verification import RolloutEvidence

REASONING_DELIMITERS = (
    ("<think>", "</think>"),
    ("<thinking>", "</thinking>"),
    ("<|start_think|>", "<|end_think|>"),
)


def final_answer_text(text: str) -> str:
    """Strip explicit reasoning blocks, preserving unmarked text.

    An unfinished reasoning block has no final answer. A closing marker alone
    ends reasoning whose opening marker was supplied by the chat template.
    Unmarked prose cannot be reliably classified as reasoning.
    """
    for opening, closing in REASONING_DELIMITERS:
        text = re.sub(re.escape(opening) + ".*?" + re.escape(closing), "", text, flags=re.DOTALL)
    last_closing = max(
        (text.rfind(closing) + len(closing) for _, closing in REASONING_DELIMITERS if closing in text), default=0
    )
    text = text[last_closing:]
    if any(opening in text for opening, _ in REASONING_DELIMITERS):
        return ""
    return text.strip().removesuffix("<|eot_id|>").strip()


def has_final_answer(text: str, evidence: RolloutEvidence | None) -> bool:
    """Return whether a completed response carries answer text or a tool call."""
    if evidence is not None and evidence.stop_reason == "length":
        return False
    if final_answer_text(text):
        return True
    message = evidence.metadata.get("assistant_message") if evidence is not None else None
    return bool(message and message.get("tool_calls"))


def last_boxed_answer(text: str) -> str | None:
    boxed = last_boxed_only_string(text)
    return None if boxed is None else remove_boxed(boxed).strip()


def final_verdict(text: str, labels: set[str]) -> str:
    """Accept one exact label on the last nonempty line."""
    cleaned = final_answer_text(text)
    last_line = cleaned.rsplit("\n", 1)[-1].strip()
    if last_line not in labels:
        raise ValueError(f"Invalid final judge verdict: {last_line!r}")
    return last_line
