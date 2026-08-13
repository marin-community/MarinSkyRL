# RL behavior analysis

`python -m infra.rl_analysis` analyzes local SkyRL rollout artifacts, optional matched
evaluation artifacts, and optional training logs. It writes its output below the
requested directory and uses `INDEX.md` as the completion marker.

```bash
cd /Users/benjaminfeuer/Documents/MarinSkyRL
uv run --group analysis python -m infra.rl_analysis \
  --rollout-dir /path/to/rollouts \
  --baseline-dir /path/to/base-eval \
  --post-dir /path/to/post-rl-eval \
  --training-log-dir /path/to/training_logs \
  --output-dir /path/to/analysis
```

The input directory may contain Harbor `result.json` files recursively, or a
JSONL file with equivalent result mappings. The output includes:

- `Q1_behavioral_delta/comparison.json`: common task and matched-trial counts,
  task- and trial-weighted reward deltas, and comparison validity;
- `Q2_temporal/temporal_summary.json`: time-binned rollout reward, turns, and
  errors;
- `Q2_skyrl_metrics/`: output from the existing SkyRL metric parser when
  `--training-log-dir` is supplied;
- `Q3_temporal_overlay/overlay.json`: rollout reward bins, baseline/post reward
  markers, and comparison statistics;
- `Q4_solve_rate_by_context/`: rollout and evaluation reward/error summaries
  grouped by peak request prompt-token bucket, with missing context reported as
  `unknown`.

Training rollouts and evaluation traces often contain different task sets. A
zero-overlap pair is not a before/after model-quality comparison. The pipeline
records `invalid_for_comparison: true` and `INDEX.md` states the caveat instead
of reporting a delta.

Task dataset inventory and verifier notes are maintained in
[`TaskTrove`](../.agents/projects/rl_data/tasktrove.md).
