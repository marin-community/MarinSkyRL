# Single-action rewards

SWE records expose `tool_name`, `nemo`, and `exact` scores. Tool-name matching
checks the called function. NeMo uses the released argument comparison, including
its file-path and list handling. Exact matching compares canonical arguments and
preserves command and list order. Each action retains all three scores; the selected
reward determines training.

Terminal records expose schema, command, and string comparisons, plus `string_or_jev`
when the JEV judge is configured. Passing the released string-similarity threshold short-circuits the judge. Other
valid actions are judged for equivalence of immediate behavior, including command
order, side effects, and completion state. See
[Terminal verifier details](../terminal-pivot-verifier.md) for normalization and
judge behavior.

`require_completed_action` rejects a response that did not terminate normally.
A grading exception is an unavailable reward, not evidence of a policy failure.
Experiment acceptance must check every retained sample for a successful verdict;
loss masking alone does not make an incompletely graded GRPO group valid.
