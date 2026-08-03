# Nightly end-to-end gate

A real GRPO training run on one H100, every night, scored against a checked-in spec. It
exists to answer one question: **does the user-facing training path still work?** Rollouts
generate, rewards score, the policy takes a step, weights sync back into the inference
engine. It says nothing about model quality and is not meant to — two steps of a 0.6B
policy over 16 prompts carries no signal about how good the model is.

| file | role |
| --- | --- |
| `run_h100.sh` | what runs on the GPU: validate the image, slice GSM8K, train, gate |
| `gate.py` | reads a run's log and decides whether it was healthy (`python -m ci.marin_nightly.gate`) |
| `specs/gsm8k-qwen3-0.6b.json` | the thresholds, with provenance for why each one is what it is |
| `../../../.github/workflows/marin-nightly.yaml` | provisions the H100 through Iris and tears it down |

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

The training run needs the digest-pinned GPU-RL task image and takes its knobs from the
environment (`MODEL`, `MAX_STEPS`, `DATA_DIR`). Inside that image:

```bash
NIGHTLY_RL_ENV=/opt/marin/envs/rl MAX_STEPS=2 bash ci/marin_nightly/run_h100.sh
```

To exercise the whole path — provision, train, gate, tear down — trigger the workflow:

```bash
gh workflow run marin-nightly.yaml -f max_steps=2
```

## Tightening the spec

The shipped thresholds are structural: metrics exist, are finite, and `reward/avg_raw_reward`
is inside `[0, 1]` (gsm8k scores each rollout 0 or 1, so a mean outside that range means the
reward path is broken). There is deliberately no reward floor above zero — a 0.6B model can
legitimately score nothing on 16 GSM8K prompts, and a floor would make the nightly flaky
rather than informative. Once enough nightlies have run green, replace it with a floor drawn
from the observed distribution and lower the wall-clock budget to the observed p95.
