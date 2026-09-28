"""Final-answer extraction shared by Ultra graders and model transport."""

import re


def final_answer_text(text: str) -> str:
    """Remove complete reasoning blocks; an unfinished block has no answer."""
    text = re.sub(r"<(think|thinking)>.*?</\1>", "", text, flags=re.DOTALL)
    for closing in ("</think>", "</thinking>"):
        if closing in text:
            text = text.rsplit(closing, 1)[-1]
    if re.search(r"<(?:think|thinking)>", text):
        return ""
    return text.strip()


def last_boxed_answer(text: str) -> str | None:
    index = text.rfind(r"\boxed{")
    if index < 0:
        return None
    start = index + len(r"\boxed{")
    depth = 1
    for cursor in range(start, len(text)):
        depth += (text[cursor] == "{") - (text[cursor] == "}")
        if depth == 0:
            return text[start:cursor].strip()
    return None


def final_verdict(text: str, labels: set[str]) -> str:
    """Accept one exact label on the last nonempty line."""
    cleaned = final_answer_text(text)
    last_line = cleaned.rsplit("\n", 1)[-1].strip()
    if last_line not in labels:
        raise ValueError(f"Invalid final judge verdict: {last_line!r}")
    return last_line
