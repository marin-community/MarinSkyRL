import re

_BOXED_LETTER = re.compile(r"\\boxed\{([A-Za-z])\}")


def extract_mcq_answer(response: str) -> str | None:
    """Return the first boxed option letter (A–Z/a–z) uppercased, or None.

    Both the MCQ environment and the ``mcq`` data contract score through this
    helper so runtime rewards and preparation preflights cannot diverge.
    """
    match = _BOXED_LETTER.search(response)
    return match.group(1).upper() if match else None
