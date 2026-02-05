from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

POSITIVE_RESPONSE_COLOR = "green"
NEGATIVE_RESPONSE_COLOR = "yellow"
BASE_PROMPT_COLOR = "cyan"


def _color_block_format_and_kwargs(
    text: str,
    color: str,
    field_prefix: str,
) -> tuple[str, dict]:
    """Build a format string and kwargs for a multi-line colored block.

    The format string will look like:
        "<color>{p0}</color>\n<color>{p1}</color>\n..."

    where "p0", "p1", ... are placeholder names starting with `field_prefix`.
    """
    # Ensure at least one line
    lines = text.splitlines() or [""]

    fmt_lines = []
    kwargs: dict[str, str] = {}

    for i, line in enumerate(lines):
        key = f"{field_prefix}{i}"
        # NOTE: double braces {{ }} so that {key} survives into str.format
        fmt_lines.append(f"<{color}>{{{key}}}</{color}>")
        kwargs[key] = line

    fmt = "\n".join(fmt_lines)
    return fmt, kwargs


def log_example(
    logger: Any,
    prompt: List[Dict[str, Any]],
    response: str,
    reward: Optional[Union[float, List[float]]] = None,
) -> None:
    """
    Log a single example prompt and response with formatting and colors.

    Args:
        logger: The logger instance to use (expected to be loguru logger or compatible).
        prompt: The input prompt in OpenAI message format.
        response: The output response string.
        reward: The reward value(s) associated with the response.
    """
    reward_val = 0.0
    reward_str = "N/A"
    try:
        prompt_str = str(prompt)
        response_str = str(response)
        # --- Reward handling ---
        if reward is not None:
            if isinstance(reward, list):
                reward_val = float(sum(reward))
            else:
                reward_val = float(reward)
            reward_str = f"{reward_val:.4f}"

        # --- Color selection ---
        if reward is not None and reward_val > 0:
            response_color = POSITIVE_RESPONSE_COLOR
        else:
            response_color = NEGATIVE_RESPONSE_COLOR

        # --- Build per-line colored blocks in the *format string* ---
        prompt_fmt, prompt_kwargs = _color_block_format_and_kwargs(prompt_str, BASE_PROMPT_COLOR, "p")
        response_fmt, response_kwargs = _color_block_format_and_kwargs(response_str, response_color, "r")

        # Single format string with only our own markup and placeholders
        log_format = "Example:\n" f"  Input: {prompt_fmt}\n" "  Output (Total Reward: {reward}):\n" f"{response_fmt}"

        # Merge all args for str.format
        format_kwargs = {**prompt_kwargs, **response_kwargs, "reward": reward_str}

        # Let Loguru parse tags in log_format and then substitute arguments.
        logger.opt(colors=True).info(log_format, **format_kwargs)
    except Exception as e:
        print(f"Error pretty printing example, debug printing instead: {e}")
        print(f"Example:\n  Input: {prompt}\n  Output (Total Reward: {reward_str}):\n{response}")


@dataclass
class ContentMismatchDiagnostics:
    """Diagnostic information for debugging content mismatch errors.

    Used when vLLM/sglang cannot find expected content after template rendering,
    such as continue_final_message failures or other content alignment issues.
    """
    accumulated_tokens: int
    accumulated_token_ids_count: int
    content_length: int
    content_preview: str
    content_suffix: str
    response_role: Optional[str]
    error_message: str

    @classmethod
    def from_accumulator(
        cls,
        content: str,
        token_count: int,
        token_ids: List[int],
        response_role: Optional[str],
        error_message: str,
        preview_length: int = 500,
        suffix_length: int = 200,
    ) -> "ContentMismatchDiagnostics":
        """Create diagnostics from accumulator state.

        Args:
            content: The accumulated content string
            token_count: Number of completion tokens
            token_ids: List of accumulated token IDs
            response_role: The role of the response (e.g., "assistant")
            error_message: The error message from vLLM/sglang
            preview_length: Max chars to show from content start
            suffix_length: Max chars to show from content end

        Returns:
            ContentMismatchDiagnostics instance
        """
        content_preview = content[:preview_length] if content else "(empty)"
        if len(content) > preview_length:
            content_preview += "..."

        content_suffix = content[-suffix_length:] if len(content) > suffix_length else content

        # Truncate error message if too long
        error_msg_truncated = error_message[:500] if len(error_message) > 500 else error_message

        return cls(
            accumulated_tokens=token_count,
            accumulated_token_ids_count=len(token_ids),
            content_length=len(content),
            content_preview=content_preview,
            content_suffix=content_suffix,
            response_role=response_role,
            error_message=error_msg_truncated,
        )

    def format_log_message(self, header: str = "Content mismatch detected") -> str:
        """Format diagnostics as a multi-line log message.

        Args:
            header: Header text for the log message

        Returns:
            Formatted string for logging
        """
        return (
            f"{header}. Diagnostic info:\n"
            f"  - Accumulated tokens: {self.accumulated_tokens}\n"
            f"  - Accumulated token_ids count: {self.accumulated_token_ids_count}\n"
            f"  - Content length: {self.content_length} chars\n"
            f"  - Content starts with: {self.content_preview!r}\n"
            f"  - Content ends with: {self.content_suffix!r}\n"
            f"  - Response role: {self.response_role}\n"
            f"  - Error: {self.error_message}"
        )
