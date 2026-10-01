# Nightly end-to-end gates

The nightly runs GSM8K GRPO on one H100, synchronous OPD on four H100s,
and Grug Megatron training on four H100s.
All policy updates use Megatron and the frozen root environment. The GSM8K run is
scored against a checked-in spec; the other lanes exercise teacher scoring, Grug
training and weight sync. The canonical task rollout gate runs manually through a Marin RL artifact.

| file | role |
| --- | --- |
| `run_h100.sh` | train GSM8K and gate metrics on H100 |
| `run_opd_h100.sh` | run synchronous OPD with separate policy, rollout, and teacher roles |
| `run_grug_megatron.sh` | run Grug parity, training, and serving gates on four H100s |
| `gate.py` | score a training run against its spec |
| `specs/gsm8k-qwen3-0.6b-megatron.json` | GSM8K gate thresholds and provenance |
| `specs/task-rollouts.json` | canonical task completion, token evidence, and optimizer thresholds |

## How the gate sees the run

The trainer mirrors every tracker payload to stdout as a `WANDB_MIRROR` line, so a run's
metrics are recoverable from its log alone — no wandb, no checkpoint, no cluster access:

```
WANDB_MIRROR kind=train step=2 metrics={"policy/policy_loss": 0.41, "reward/avg_raw_reward": 0.25, ...}
```

`gate.py` parses those, counts distinct training steps, and checks the final payload
against the spec's required metrics and bounds. A spec can also require evidence across
the run: finite values at every step, minimum observation counts, first-to-last-window
improvement, and a minimum number of observations above or below a threshold. Training and evaluation
payloads are separate streams. `at_step` selects a numbered step, `first`, or
`last` before checking a series; a required selected observation must exist. Duplicate payloads for one stream and step count once;
conflicting copies fail. The gate exits non-zero with one line per violation.
`tests/cpu/test_marin_nightly_gate.py` covers it.

## Metric series rules

`metric_series` entries name a `kind` (`train` or `eval`) and a metric. Each entry
states `required` and `min_observations`. A missing optional metric is ignored; a
metric that appears is checked for finite values. `finite_every_step` also requires
the metric in every payload of that kind. Sparse per-N metrics omit that field and
set a measured minimum observation count. A `trend` compares the first and last
`window` finite observations; too few observations fail. An `occurrence` requires
`minimum_count` values `above`, `below` or inclusively `at_least` its `threshold`.
`through_step` limits observations to that completed step or earlier.

A learning requirement pairs a trend with a required initial observation:

```json
[
  {"kind": "eval", "metric": "eval/train/avg_score", "required": true,
   "min_observations": 1, "at_step": 0},
  {"kind": "eval", "metric": "eval/train/avg_score", "required": true,
   "min_observations": 2, "trend": {"window": 1, "min_improvement": 0.2}}
]
```

The first row requires the step-0 baseline. The second requires a reward gain
of at least0.2 from that baseline. CatCountCanary's async spec uses sampled
training-prompt evaluations instead: initial reward in[0.10,0.45] atstep0,
then at leastone score >=0.65 throughstep30. The launcher stops at the first
qualifying sampled evaluation. Its spec also requires post-optimizer
`policy/dp_weight_checksum_mismatch`=0 at every training step. Sync is a
manual launcher option outside the canary and CI.

A negative `min_improvement`, such as -0.1, permits a decrease of at most 0.1.
Top-level `finite_metrics` and `bounds` are optional final-step checks;
`metric_series` expresses requirements across the run.

## Two Ray instances cannot share a node

Ray persists session state under a temp directory. Two Ray instances that end up on one node find
each other's and the second dies: "Session name ... does not match persisted value. Perhaps there
was an error connecting to Redis." Observed on 2026-09-10 between two single-GPU jobs submitted six
seconds apart.

This is not specific to this lane and it is not new. `gsm8k-h100` starts Ray through
the standalone SkyRL Hydra entrypoint; `grug-megatron-h100` starts it through `initialize_ray` in
`tests/gpu/test_grug_megatron.py`; both target `cw-rno2a` and both are launched by the same 09:00
cron. Marin-managed launches instead enter through the config-native task runtime, which pins the
ports before starting the same training entrypoint.

No collision has been observed between the scheduled lanes, and Iris placement may well keep them
apart, but nothing here guarantees it. If one of them fails at `ray start` with that message, this
is why. Two manual runs on the same node can cause this failure. Run them serially.

## Running it by hand

Run these commands from `skyrl-train/`. The gate uses the Python standard library and reads a saved run log:

```bash
uv run --frozen python -m ci.marin_nightly.gate \
    --log nightly-run.log \
    --spec ci/marin_nightly/specs/gsm8k-qwen3-0.6b-megatron.json \
    --wall-clock-seconds 900
```

The training run starts from the cluster-configured Iris task image and resolves the
architecture-specific `vllm` wheel from the root `uv.lock`. It takes its knobs from the
environment (`MODEL`, `MAX_STEPS`, `DATA_DIR`). Inside an Iris GPU task:

```bash
MAX_STEPS=2 bash ci/marin_nightly/run_h100.sh
```

The Megatron lane runs `tests/gpu/test_grug_megatron.py` and the two-GPU CP2
FlashAttention forward/backward smoke with the frozen Megatron runtime closure;
see `docs/grug-megatron-training.md` for the Grug tests.

The manual task rollout specification scores a saved canonical Shellbox training log.
It expects eight distinct tasks, one sample per task, and at least two model turns
per task in the final training batch. It is separate from the scheduled workflow.

Launch configuration and log collection follow Marin's
[RL launch reference](https://github.com/marin-community/marin/blob/main/docs/references/rl-launching.md).
The input log must contain the trainer's `WANDB_MIRROR` lines, including the final
optimizer step. From the SkyRL repository root, score that log:

```bash
uv run --frozen python skyrl-train/ci/marin_nightly/gate.py \
    --log task-rollouts.log \
    --spec skyrl-train/ci/marin_nightly/specs/task-rollouts.json \
    --wall-clock-seconds 900
```

Measure elapsed time from artifact submission through terminal export, including allocation and startup.
Supply this value for `--wall-clock-seconds`. The gate requires eight
multi-turn tasks, generated tokens with behavior logprobs, a finite optimizer loss,
no failed trajectories, and a positive mean correction weight no greater than two. The engine rejects changed token
prefixes and token/logprob length mismatches before the buffer receives a rollout.
The metric gate does not inspect checkpoint or export files. Full launch acceptance
also requires a persisted run manifest, a checkpoint at the final optimizer step, and
a Hugging Face export that the model loader can read.

The canonical engine has no OpenCode process or automatic history compaction. The
removed OpenCode stress specifications do not apply to this engine. Engine and worker
tests check context limits, tool output, deadlines, and cleanup. These CPU checks do
not replace live backend acceptance.

The scheduled workflow runs the GSM8K, OPD, and Grug lanes. It does not run the manual
task rollout gate. Trigger it from the repository root:

```bash
gh workflow run marin-nightly.yaml \
  -f max_steps=2 \
  -f target_cluster=cw-rno2a
```

## Tightening the spec

The shipped thresholds are structural: metrics exist, are finite, and `reward/avg_raw_reward`
is inside `[0, 1]` (gsm8k scores each rollout 0 or 1, so a mean outside that range means the
reward path is broken). There is deliberately no reward floor above zero — a 0.6B model can
legitimately score nothing on 16 GSM8K prompts, and a floor would make the nightly flaky
rather than informative. Once enough nightlies have run green, replace it with a floor drawn
from the observed distribution and lower the wall-clock budget to the observed p95.
