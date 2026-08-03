# Debugging log for RL metric log formats

Terminus-2 agentic runs emit JSON `WANDB_MIRROR` records, but the offline parser defaults to an `agentic` mode that only accepts legacy Python-dictionary blocks. The parser should select serialization independently of the rollout harness and must not silently return zero metrics for current agentic logs.

## Initial status

`process_log_file` defaults to `fmt="agentic"`. That branch searches for single-quoted blocks beginning with `async/`; JSON `WANDB_MIRROR` records are parsed only when the caller chooses `--format standard`. The same value also controls trace processing and report selection, despite describing the log serialization in the parser branch.

## Hypothesis 1

Auto-detecting JSON `WANDB_MIRROR` records before falling back to legacy Python-dictionary blocks will
parse current standard and agentic logs without a harness-shaped flag. Trace availability can select
trace-oriented analysis independently of the detected serialization.

## Changes to make

Add a regression test using an agentic Terminus-style JSON training event, then separate serialization selection from trace availability and report behavior. Update the operational call sites to rely on auto-detection.

## Results

The regression test failed before the parser change because the default branch returned an empty metric
list for a valid JSON `WANDB_MIRROR` event. After the change, auto-detection identifies that event as
`wandb-json`, while retained single-quoted logs fall back to `python-dict`.

The CLI regression also passes a `trace_jobs` directory with the JSON log and verifies that the parser
produces the trace-oriented agentic report. This confirms that harness analysis is selected from trace
availability rather than the training-log serialization.

Validation:

```text
uv run --no-project --with pytest --with pandas --with matplotlib --with tabulate pytest infra/tests/ -q
69 passed in 0.48s
```

The review pass also removed unused explicit serialization overrides, consolidated the parallel report
and reward-plot implementations, made report headings neutral to the harness, and made an invalid explicit
`trace_jobs` path fail instead of silently disabling trace analysis. The CLI controller remains responsible
for dispatching its existing outputs; the patch reduces its mode branches and does not move unrelated
output code.

## Future work

- [ ] Remove the legacy Python-dictionary parser after retained logs no longer require it.
