# Nightly end-to-end gates

The nightly runs dense Qwen GRPO on one H100 and a tiny Grug RL cycle on four GB200s
from the frozen root environment. The H100 run is scored against a checked-in spec;
the GB200 run proves the locked Marin vLLM wheel can load Grug, generate rollouts,
train the eager FSDP2 policy, synchronize mixed-dtype weights, and generate again.
These are integration gates, not model-quality experiments.

| file | role |
| --- | --- |
| `run_h100.sh` | sync the frozen root environment, slice GSM8K, train, and gate on H100 |
| `run_grug_vllm.sh` | run a tiny Grug rollout/train/broadcast/rollout cycle on four GB200s |
| `gate.py` | reads a run's log and decides whether it was healthy (`python -m ci.marin_nightly.gate`) |
| `specs/gsm8k-qwen3-0.6b.json` | the thresholds, with provenance for why each one is what it is |
| `../../../.github/workflows/marin-nightly.yaml` | provisions both GPU gates through Iris and tears them down |

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
