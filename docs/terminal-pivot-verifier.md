# Terminal PivotRL reward

The `string_or_jev` reward requires a normally terminated response, valid Terminus-2 JSON,
and the existing one-sided task-completion check. It then accepts either command-string
similarity at least 0.9 or JEV equivalence probability at least 0.5. Schema validity alone
does not earn reward. Command strings are compared with `SequenceMatcher` over concatenated
keystrokes, matching the released string-only verifier.

JEV receives reference and candidate action JSON and a rubric for equivalence of their
immediate terminal behavior. It receives no model identity or saved reward. This is an
adaptation of the paper's functional verifier, not a reproduction of its judge prompt.
No shell commands execute during grading. Exact commands, string similarity, and
schema/completion remain separate diagnostics; `string_or_jev` is the optimized reward.

```yaml
environment:
  skyrl_gym:
    nemotron_ultra:
      pivot_arm: rl_string_or_jev
      pivot_reward: string_or_jev
      require_completed_action: true
      terminal_judge:
        base_url: https://openrouter.ai/api/alpha/decisions
        model: typesafe/jev-1.13
        expected_model: typesafe/jev-1.13-20260917
        api_key_env: OPENROUTER_API_KEY
        threshold: 0.5
        requests_per_second: 4
        timeout_seconds: 90
```

The launcher forwards `OPENROUTER_API_KEY` from its environment into Iris tasks and Ray
workers. Keep the credential out of YAML and resolved artifacts. Each rollout process
limits uncached judge requests to the configured rate. With four rollout workers in each
of two jobs, four requests/second per process permits 32 requests/second across the pair,
excluding transient retries. Successful identical judgments are cached within each process.
String passes bypass the judge. Judge transport failures produce unavailable verification
and follow the configured trajectory error policy; they must not be reported as negative
semantic judgments. Check graded coverage and retained verification statuses during a run.

Keep training-row selection and evaluation identity explicit when comparing reward variants.
Using a frozen union-selected pool isolates the reward change, but does not repeat pivot
selection under the hybrid verifier.
