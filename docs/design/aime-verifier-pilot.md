# AIME verification and reward policy

AIME verification consumes immutable model evidence.
The evidence contains the response, stop reason, exact tokens, and generated-token count.
The verifier returns its native +1/-1 score, an explicit correctness verdict, and diagnostics.

The reward policy separately applies the configured length penalty, truncation penalty, and minimum response length.
It returns an optimization reward and reward components.
The direct answer session returns these values in a TaskSession transition.
The engine retains the sampled tokens without action replacement or a generation-metadata hook.

The final GradeResult preserves the native score and its bounds.
The training projection uses the optimization reward for policy updates and the native verdict for metrics.
Negative scores remain in the failure token-length bucket.

AIME metrics include the fraction of responses above the evaluation token budget and the fraction with a parseable answer within that budget.
Correct and incorrect over-budget fractions remain separate.
Configure the budget and reward policy in `environment.task_sessions.aime`.

See [the AIME grader](../../skyrl-gym/skyrl_gym/answer_tasks.py),
[the native verifier](../../skyrl-gym/skyrl_gym/envs/aime/verifier.py),
and [canonical task rollouts](../../skyrl-train/docs/tutorials/task_rollouts.rst).
