"""Chat-template error translation and compatibility repairs."""

from copy import deepcopy
from typing import Any

from jinja2 import TemplateError


SINGLE_TOOL_CALL_TEMPLATE_ERROR = "This template expects exactly one tool call per assistant turn."


def template_error_from_exception(error: BaseException) -> TemplateError | None:
    """Recover a Jinja error hidden below vLLM and Ray exception wrappers."""
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, TemplateError):
            return current
        if SINGLE_TOOL_CALL_TEMPLATE_ERROR in str(current):
            return TemplateError(SINGLE_TOOL_CALL_TEMPLATE_ERROR)
        pending.extend(cause for cause in (current.__cause__, current.__context__) if cause is not None)
        as_instanceof_cause = getattr(current, "as_instanceof_cause", None)
        if callable(as_instanceof_cause):
            try:
                remote_cause = as_instanceof_cause()
            except Exception:
                remote_cause = None
            if isinstance(remote_cause, BaseException) and remote_cause is not current:
                pending.append(remote_cause)
    return None


def sequentialize_multi_tool_call_turns(
    messages: list[dict[str, Any]],
    assistant_message_index: int | None = None,
) -> tuple[list[dict[str, Any]], int | None]:
    """Split parallel calls into consecutive single-call assistant messages.

    Tool responses remain after the assistant messages. This representation is
    accepted by strict one-call templates and lets exact continuation splice the
    original sampled token prefix before all tool responses.
    """
    normalized: list[dict[str, Any]] = []
    normalized_assistant_index = assistant_message_index
    for message_index, message in enumerate(messages):
        tool_calls = message.get("tool_calls")
        if message.get("role") != "assistant" or not isinstance(tool_calls, list) or len(tool_calls) <= 1:
            normalized.append(deepcopy(message))
            continue

        first_normalized_index = len(normalized)
        for call_index, tool_call in enumerate(tool_calls):
            single_call_message = deepcopy(message)
            single_call_message["tool_calls"] = [deepcopy(tool_call)]
            if call_index > 0:
                single_call_message["content"] = None
            normalized.append(single_call_message)

        if assistant_message_index is not None:
            if message_index < assistant_message_index:
                normalized_assistant_index += len(tool_calls) - 1
            elif message_index == assistant_message_index:
                normalized_assistant_index = first_normalized_index + len(tool_calls) - 1

    return normalized, normalized_assistant_index
