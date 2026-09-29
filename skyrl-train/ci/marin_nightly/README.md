# Nightly end-to-end gates

The nightly runs GSM8K GRPO on one H100, synchronous OPD on four H100s,
Grug Megatron training on four H100s, and an OpenCode agentic RL step on eight H100s.
All policy updates use Megatron and the frozen root environment. The GSM8K run is
scored against a checked-in spec; the other lanes exercise teacher scoring, Grug
training and weight sync, and agentic rollout coverage.

| file | role |
| --- | --- |
| `run_h100.sh` | train GSM8K and gate metrics on H100 |
| `run_opd_h100.sh` | run synchronous OPD with separate policy, rollout, and teacher roles |
| `run_grug_megatron.sh` | run Grug parity, training, and serving gates on four H100s |
| `run_opencode.sh` | submit and gate the federated OpenCode RL canary |
| `gate.py` | score the GSM8K run against its spec |
| `specs/gsm8k-qwen3-0.6b-megatron.json` | GSM8K gate thresholds and provenance |
| `specs/opencode-qwen3-8b.json` | OpenCode continuation and policy update thresholds |

## How the gate sees the run

The trainer mirrors every tracker payload to stdout as a `WANDB_MIRROR` line, so a run's
metrics are recoverable from its log alone — no wandb, no checkpoint, no cluster access:

```
WANDB_MIRROR kind=train step=2 metrics={"policy/policy_loss": 0.41, "reward/avg_raw_reward": 0.25, ...}
```

`gate.py` parses those, takes the **final** training step (a run can look healthy for a
step and then degrade into NaN), and checks it against the spec: the step count was
reached, the required metrics are present and finite, the bounded ones are inside their
range, and the run finished inside its wall-clock budget. It exits non-zero with one line
per violation. `tests/cpu/test_marin_nightly_gate.py` covers it.

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
is why. Two runs launched by hand seconds apart will do it: run them serially.

## Running it by hand

The gate is pure stdlib and runs anywhere, against any run log:

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
step, eight correlated trials, at least 16 correlated turns, 100% exact TIS/full-TITO,
and no fallback, decline, skipped batch, or failed trajectory. A failure before those
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
