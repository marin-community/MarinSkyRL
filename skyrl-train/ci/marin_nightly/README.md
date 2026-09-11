# Nightly end-to-end gates

The nightly runs dense Qwen GRPO on one H100, a tiny Grug RL cycle on four GB200s,
the Grug Megatron gates on four H100s, and an OpenCode agentic RL step on eight H100s,
all from the frozen root environment. The
GSM8K run is scored against a checked-in spec; the GB200 run proves the locked Marin
vLLM wheel can load Grug, generate rollouts, train the eager FSDP2 policy, synchronize
mixed-dtype weights, and generate again; the Megatron run checks that the Megatron
port of Grug matches the HF reference, keeps the training forward bit-identical to
the recomputed old log-probs, and completes a rollout/train/broadcast/rollout cycle.
The OpenCode lane runs eight concurrent, three-turn Daytona tasks through the controller
RecordProxy and requires exact full-TITO/TIS coverage before an FSDP2 policy update.
These are integration gates, not model-quality experiments.

| file | role |
| --- | --- |
| `run_h100.sh` | sync the frozen root environment, slice GSM8K, train, and gate on H100 |
| `run_grug_vllm.sh` | run a tiny Grug rollout/train/broadcast/rollout cycle on four GB200s |
| `run_grug_megatron.sh` | run the Grug Megatron parity, training, and serving gates on four H100s |
| `run_opencode.sh` | submit, wait for, and gate the federated RNO2A OpenCode RL canary |
| `gate.py` | reads a run's log and decides whether it was healthy (`python -m ci.marin_nightly.gate`) |
| `specs/gsm8k-qwen3-0.6b.json` | the thresholds, with provenance for why each one is what it is |
| `specs/opencode-qwen3-8b.json` | exact continuation, literal bridge, TIS, and optimizer thresholds |
| `../../../.github/workflows/marin-nightly.yaml` | provisions the GPU gates through Iris and tears them down |

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

## Running it by hand

The gate is pure stdlib and runs anywhere, against any run log:

```bash
uv run --frozen python -m ci.marin_nightly.gate \
    --log nightly-run.log \
    --spec ci/marin_nightly/specs/gsm8k-qwen3-0.6b.json \
    --wall-clock-seconds 900
```

The training run starts from the cluster-configured Iris task image and resolves the
architecture-specific `vllm` wheel from the root `uv.lock`. It takes its knobs from the
environment (`MODEL`, `MAX_STEPS`, `DATA_DIR`). Inside an Iris GPU task:

```bash
MAX_STEPS=2 bash ci/marin_nightly/run_h100.sh
```

The GB200 lane additionally imports `vllm._C` and the cuMem allocator, verifies the
Grug model registry entry, then runs a real rollout, eager FSDP2 policy update,
mixed-dtype weight broadcast, and second rollout. The eager policy path keeps this
gate independent of the optional compiled FlashAttention package.

The Megatron lane runs `tests/gpu/test_grug_megatron.py` with the Megatron runtime
closure; see `docs/grug-megatron-training.md` for what each test guards.

The OpenCode lane is deliberately a real federated RL launch rather than a mocked agent
test. It provisions one RNO2A H100x8 node, creates eight air-gapped Daytona sandboxes,
runs the pinned OpenCode 1.18.2 CLI at concurrency eight, captures every served token via
RecordProxy, and completes one policy step. It runs daily with the other gates. A healthy
run targets about 15 minutes, or roughly 2 H100-hours plus eight short-lived Daytona
sandboxes; its 40-minute hard allowance is a hang backstop, not the expected cost.

To reproduce only this lane from an authenticated checkout:

```bash
JOB_NAME="marinskyrl-opencode-manual-$(date +%s)" \
  bash skyrl-train/ci/marin_nightly/run_opencode.sh
```

The script always cancels its Iris job on exit. Its log must contain one finite training
step, eight correlated trials, at least 24 correlated turns, 100% exact TIS/full-TITO,
and no fallback, decline, skipped batch, or failed trajectory. A failure before those
metrics should be triaged from the uploaded job log in this order: Iris allocation and
runtime setup, Daytona snapshot/sandbox setup, OpenCode process errors, RecordProxy
correlation, continuation declines, then policy forward/backward and weight sync.

The same launcher has two manually triggered boundary suites. They use the checked-in
`ci/opencode_smoke/tasks/boundary-mix` corpus and are intentionally not daily: each mode
costs another H100x8 allocation and some cases deliberately fail or time out.

```bash
OPENCODE_MODE=compaction-stress \
  JOB_NAME="marinskyrl-opencode-compaction-$(date +%s)" \
  LOG_PATH=opencode-compaction.log \
  bash skyrl-train/ci/marin_nightly/run_opencode.sh

OPENCODE_MODE=overflow-stress \
  JOB_NAME="marinskyrl-opencode-overflow-$(date +%s)" \
  LOG_PATH=opencode-overflow.log \
  bash skyrl-train/ci/marin_nightly/run_opencode.sh
```

The first mode requires automatic summarization to preserve an exact post-compaction
training chain while invalid UTF-8 tool bytes, a capped single-turn response, and agent
and verifier timeouts run beside normal trials. The second disables compaction and
requires the same oversized tool result to become exactly one typed
`ContextLengthExceededError`, with no retry storm. Both must still complete a finite
policy step from the usable trajectories. Their checked-in specs are production-smoke
contracts; update a threshold only from a cited real run, never merely to make a local
fixture pass.

To exercise the whole path — provision, train, gate, tear down — trigger the workflow:

```bash
gh workflow run marin-nightly.yaml \
  -f max_steps=2 \
  -f target_cluster=cw-rno2a \
  -f grug_target_cluster=cw-us-east-08a
```

## Tightening the spec

The shipped thresholds are structural: metrics exist, are finite, and `reward/avg_raw_reward`
is inside `[0, 1]` (gsm8k scores each rollout 0 or 1, so a mean outside that range means the
reward path is broken). There is deliberately no reward floor above zero — a 0.6B model can
legitimately score nothing on 16 GSM8K prompts, and a floor would make the nightly flaky
rather than informative. Once enough nightlies have run green, replace it with a floor drawn
from the observed distribution and lower the wall-clock budget to the observed p95.
