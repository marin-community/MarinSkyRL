"""The Qwen3 Thinking-2507 accumulating chat template."""

from pathlib import Path

from jinja2 import Environment, StrictUndefined

_TEMPLATE_PATH = Path(__file__).parents[3] / "chat_templates" / "qwen3_thinking_2507_acc.jinja2"
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a command",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
        },
    }
]


def _render(messages: list[dict], *, add_generation_prompt: bool) -> str:
    template = Environment(undefined=StrictUndefined).from_string(_TEMPLATE_PATH.read_text())
    return template.render(messages=messages, tools=_TOOLS, add_generation_prompt=add_generation_prompt)


def test_generation_prompt_opens_the_reasoning_block_like_the_2507_template() -> None:
    rendered = _render([{"role": "user", "content": "List files."}], add_generation_prompt=True)

    assert rendered.endswith("<|im_start|>assistant\n<think>\n")


def test_next_turn_prompt_extends_the_served_prompt_and_sampled_tool_call() -> None:
    user = [{"role": "user", "content": "List files."}]
    sampled = 'Look first.\n</think>\n\n<tool_call>\n{"name": "shell", "arguments": {"command": "ls"}}\n</tool_call>'
    assistant = {
        "role": "assistant",
        "content": "Look first.\n</think>\n\n",
        "tool_calls": [{"type": "function", "function": {"name": "shell", "arguments": '{"command": "ls"}'}}],
    }
    observation = {"role": "tool", "content": "a.txt"}

    served = _render(user, add_generation_prompt=True) + sampled + "<|im_end|>"
    next_prompt = _render([*user, assistant, observation], add_generation_prompt=True)

    assert next_prompt.startswith(served)
    assert next_prompt[len(served) :] == (
        "\n<|im_start|>user\n<tool_response>\na.txt\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n"
    )
