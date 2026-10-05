# Nightly end-to-end gates

The nightly runs synchronous OPD, Grug Megatron training and the asynchronous
CatCount learning canary on H100s. Colocated synchronous RL is not a production
mode; CatCount covers learning. OpenCode runs manually through its launcher
script. All policy updates use Megatron and the frozen root environment.

| file | role |
| --- | --- |
| `run_opd_h100.sh` | run synchronous OPD with separate policy, rollout, and teacher roles |
| `run_grug_megatron.sh` | run Grug parity, sampler–trainer agreement, training, and serving gates on four H100s |
| `run_opencode.sh` | submit and gate the federated OpenCode RL canary |
| `run_cat_count_h100.sh` | submit the CatCount coordinator and gate sampled learning on four H100s |
| `gate.py` | score a training run against its spec |
| `specs/cat-count-canary-qwen2.5-0.5b-async.json` | CatCount sampled learning and per-step mechanism requirements |
| `specs/opencode-qwen3-8b.json` | OpenCode continuation and policy update thresholds |

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
payloads are separate streams. A selected observation must exist. Duplicate
payloads for one stream and step count once;
conflicting copies fail. The gate exits non-zero with one line per violation.
`tests/cpu/test_marin_nightly_gate.py` covers it.

## Metric gates

Each `metric_gates` row selects a `kind` (`train` or `eval`) and a metric.
Every selected payload must contain a finite value. `min_observations` counts
payloads, so an evaluation every five steps has fewer observations than the
training stream. `step` selects a numbered step, `first`, or `last`;
`max_step` includes observations through that completed step.

A `bounds` range applies to every observation by default. With `minimum_count`,
it requires that many observations inside the range. Endpoint inclusion is
controlled by `inclusive_minimum` and `inclusive_maximum`. A `trend` compares
the mean of the first and last `window` observations; too few observations fail.
Negative `min_improvement` values permit a bounded decrease.
The gate returns typed failures for missing, nonfinite or out-of-range observations. Top-level `finite_metrics`
and `bounds` check the final training payload.

```json
[
  {"kind": "eval", "metric": "eval/train/avg_score",
   "min_observations": 1, "step": 0},
  {"kind": "eval", "metric": "eval/train/avg_score",
   "min_observations": 2, "trend": {"window": 1, "min_improvement": 0.2}}
]
```

CatCount requires sampled training-prompt reward in [0.10, 0.45] at step 0,
then at least one score ≥ 0.65 by step 30. The launcher evaluates every five
steps and stops at the first qualifying score. It requires
`policy/dp_weight_checksum_mismatch` = 0 after every optimizer step. The
synchronous lane is a manual launcher option outside CI.

## CatCount nightly

`cat-count-h100` submits a 4-CPU, 16-GB-memory, 8-GB-disk coordinator and two
worker tasks with two H100s and 65 CPUs each. It uses seed 17, behavior clipping
and staleness 2, without checkpoints or HF export. The runner clones Marin main,
sets the external MarinSkyRL source to the commit under test, and runs Marin's
`config/update-external.py MarinSkyRL`. That pin supplies both the launcher
package and the GPU runtime. Scheduled and manually dispatched workflows run
all three lanes. The CatCount job summary reports dashboard readiness without
affecting its learning result.

Each attempt has a 20-minute deadline including queue time. A failure with no
native training row is reported as `INFRASTRUCTURE_FAILURE` and retried once.
A failed job after training starts, or a failed metric gate, is `GATE_FAILURE`
and is not retried. The script records each attempt's wall time and exit status;
`OK against ...cat-count-canary-qwen2.5-0.5b-async.json` is the passing gate line.
The workflow uploads the combined native log and cancels its own named jobs
in the shared cleanup step.

## Two Ray instances cannot share a node

Ray persists session state under a temp directory. Two Ray instances that end up on one node find
each other's and the second dies: "Session name ... does not match persisted value. Perhaps there
was an error connecting to Redis." Observed on 2026-09-10 between two single-GPU jobs submitted six
seconds apart.

The Grug tests start Ray through `initialize_ray`; CatCount starts it through
the Marin task runtime. Both target `cw-rno2a`. A Ray startup error with that
message indicates two instances sharing a host; inspect their placement.

## Running it by hand

The gate is pure stdlib and runs anywhere, against any run log:

```bash
python3 skyrl-train/ci/marin_nightly/gate.py \
    --log nightly-run.log \
    --spec skyrl-train/ci/marin_nightly/specs/cat-count-canary-qwen2.5-0.5b-async.json \
    --wall-clock-seconds 900
```

The Grug lane runs `tests/gpu/test_grug_megatron.py`, sampler–trainer agreement in
`tests/gpu/test_grug_sampler_trainer.py`, the Levanter parity oracle and the two-GPU CP2
FlashAttention forward/backward smoke with the frozen Megatron runtime closure;
see `docs/grug-megatron-training.md` for the Grug tests.

The OpenCode lane is deliberately a real federated RL launch rather than a mocked agent
test. It provisions one RNO2A H100x8 node, creates eight air-gapped Daytona sandboxes,
runs the pinned OpenCode 1.18.2 CLI at concurrency eight, captures every served token via
RecordProxy, and completes one policy step. Run it manually with the RL-specific Daytona
credential until the GitHub Iris service account can read the canonical secret. A healthy
run targets about 15 minutes, or roughly 2 H100-hours plus eight short-lived Daytona
sandboxes; its 40-minute hard allowance is a hang backstop, not the expected cost.

To reproduce only this lane from an authenticated checkout:

```bash
LAUNCH_CONFIG=/path/to/resolved-opencode-launch.yaml \
  bash skyrl-train/ci/marin_nightly/run_opencode.sh
```

The launch document is the complete Hydra YAML emitted by the Marin artifact. The script submits
that document synchronously and gates its combined launcher and task log. Its log must contain one finite training
step, eight correlated trials, at least 16 correlated turns, 100% exact behavior-logprob alignment with full token-in/token-out coverage,
a finite positive correction weight no greater than 2, and no fallback, decline, skipped batch, or failed trajectory. A failure before those
metrics should be triaged from the uploaded job log in this order: Iris allocation and
runtime setup, Daytona snapshot/sandbox setup, OpenCode process errors, RecordProxy
correlation, continuation declines, then policy forward/backward and weight sync.

The same launcher has two manually triggered boundary suites. They use the checked-in
`ci/opencode_smoke/tasks/boundary-mix` corpus and are intentionally not daily: each mode
costs another H100x8 allocation and some cases deliberately fail or time out.

```bash
OPENCODE_MODE=compaction-stress \
  LAUNCH_CONFIG=/path/to/resolved-compaction-launch.yaml \
  LOG_PATH=opencode-compaction.log \
  bash skyrl-train/ci/marin_nightly/run_opencode.sh

OPENCODE_MODE=overflow-stress \
  LAUNCH_CONFIG=/path/to/resolved-overflow-launch.yaml \
  LOG_PATH=opencode-overflow.log \
  bash skyrl-train/ci/marin_nightly/run_opencode.sh
```

The first mode requires automatic summarization to preserve an exact post-compaction
training segment while invalid UTF-8 tool bytes, a capped single-turn response, and
agent and verifier timeouts run beside normal trials. The second disables compaction and
requires the same oversized tool result to become exactly one typed
`ContextLengthExceededError`, with no retry storm. These mixed negative-path batches do
not require an optimizer step: timeout and overflow samples without behavior logprobs
are deliberately masked, while the positive-path gate owns the real policy-update
and checkpoint contract. Instead, the stress specs require named log evidence for every
boundary and a clean workflow shutdown. Update a threshold only from a cited real run,
never merely to make a local fixture pass.

To exercise the whole path — provision, train, gate, tear down — trigger the workflow:

```bash
gh workflow run marin-nightly.yaml \
  -f target_cluster=cw-rno2a
```
