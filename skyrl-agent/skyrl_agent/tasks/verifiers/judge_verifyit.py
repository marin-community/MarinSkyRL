"""Grade external judge responses with existing schema and exact primitives."""

from verifyit.grade import InvalidTask, Status
from verifyit.modes.grade_exact import grade_exact_candidate
from verifyit.spec import ExactSpec


def literal_score(candidate: str, reference: str) -> float:
    result = grade_exact_candidate(
        ExactSpec(
            expected=(reference,),
            ignore_case=False,
            ignore_whitespace=False,
            strip_outer_whitespace=False,
        ),
        candidate,
    )
    if result.status is not Status.SCORED:
        raise RuntimeError("Exact judge comparison failed")
    return result.reward


def completion_text(response) -> str:
    choices = response["choices"]
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Judge must return one complete response")
    choice = choices[0]
    message = choice["message"]
    if (
        choice.get("finish_reason") != "stop"
        or message.get("tool_calls")
        or message.get("refusal")
    ):
        raise ValueError("Judge response is incomplete")
    text = message.get("content")
    if not isinstance(text, str):
        raise ValueError("Judge response has no text")
    return text


def require_reference(question, reference):
    if not isinstance(question, str) or not question.strip():
        raise InvalidTask("Judge question must be nonempty text")
    if not isinstance(reference, str) or not reference.strip():
        raise InvalidTask("Judge reference must be nonempty text")
