# Context budget contract

## YAML schema

```yaml
context_budget:
  request_window_tokens: <positive integer>
  max_new_tokens_per_turn: <positive integer>
  max_turns: <positive integer>
```

`request_window_tokens` must exceed `max_new_tokens_per_turn`. No other
length-control key may be declared in an Iris RL YAML.

## Public Python API

```python
@dataclass(frozen=True)
class ContextBudget:
    request_window_tokens: int
    max_new_tokens_per_turn: int
    max_turns: int

    @property
    def max_input_tokens(self) -> int:
        """Return the largest input that leaves room for one full response."""

def parse_rl_config(config_path: str, model_override: str | None = None) -> ParsedRLConfig:
    """Parse, validate, and materialize one Iris RL configuration."""

def apply_context_budget_overrides(
    parsed: ParsedRLConfig, overrides: list[str]
) -> tuple[ParsedRLConfig, list[str]]:
    """Resolve high-level context overrides and reject derived-field overrides."""

def write_resolved_context_budget(
    budget: ContextBudget, destination: Path | str, config_path: Path
) -> Path | str:
    """Persist the resolved budget as JSON and return its path."""

def clamp_generation_tokens(
    prompt_tokens: int, request_window_tokens: int, requested_output_tokens: int
) -> int:
    """Return the output allowance that fits this tokenized request, or zero."""
```

`ParsedRLConfig` exposes the resolved `context_budget`. The launcher materializes
the following fields before Hydra receives the config:

| Destination | Value |
| --- | --- |
| `trainer.max_prompt_length` | `max_input_tokens` |
| `generator.max_input_length` | `max_input_tokens` |
| `generator.max_turns` | `max_turns` |
| `terminal_bench.harbor.max_turns` | `max_turns` |
| `generator.sampling_params.max_generate_length` | `max_new_tokens_per_turn` |
| `generator.engine_init_kwargs.max_model_len` | `request_window_tokens` |
| `terminal_bench.model_info.max_input_tokens` | `max_input_tokens` |
| `terminal_bench.model_info.max_output_tokens` | `max_new_tokens_per_turn` |

## Errors

- Missing, non-integer, non-positive, or impossible budget values raise
  `ValueError` during config parsing.
- Any direct YAML declaration or low-level CLI override of a derived field,
  including deprecated `terminal_bench.harbor.max_episodes`,
  raises `ValueError` and names the rejected field.
- A step-wise request with no remaining output budget ends with stop reason
  `length` without calling vLLM.

## Out of scope

- Episode-wide generation budgets.
- Automatic training-memory calibration by model or topology.
- Changes to model architecture, parallelism, Harbor environment, or agent
  configuration unrelated to token limits.
