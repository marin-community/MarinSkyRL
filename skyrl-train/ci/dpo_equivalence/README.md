# DPO equivalence experiments (issue 969)

Bounded verification of the three stages the BFCL Grug DPO campaign left unchecked: the
distributed backward pass, the optimizer update, and the transfer of trained tensors into
vLLM. These are verification experiments, not bug reports; a pass states exactly which
configuration it covers.

| experiment | test | independent reference |
| --- | --- | --- |
| distributed backward pass | `skyrl-train/tests/gpu/test_dpo_backward_equivalence.py` | eager HF GrugMoE fp32 with a literal TRL-style DPO loss, computed per rank on GPU |
| optimizer update | `skyrl-train/tests/gpu/test_megatron_optimizer_update_equivalence.py` | literal AdamW-with-clipping in float64 on CPU |
| transfer to inference | `skyrl-train/tests/gpu/test_dpo_vllm_transfer_equivalence.py` | trainer snapshot + byte-exact serving readback + trainer-side eval log-probs |

The scalar DPO loss/gradient check on synthetic masked inputs
(`tests/cpu/objective/test_dpo_loss.py`) remains part of the chain and runs in PR CI; the
experiments here do not treat it as sufficient on its own.

## What each experiment covers

**Backward pass.** One fixed preference batch (4 chosen/rejected pairs, left padding,
masked tails) is replayed through the production objective stack
(`MegatronModelWrapper.forward_backward_mini_batch` with the registered DPO loss) on
TP1/CP1 at PP/EP/DP combinations up to PP2/EP4 on eight GPUs, covering micro-batch scaling
(the loss scale that cancels the scheduler's micro-batch division and the DP gradient
average), gradient reduction across pipeline stages and expert shards, and the batch-major
output projection that the training worker installs for DPO. Named gradients are aligned
through the bridge conversion tasks — the same mapping the weight sync uses — by exporting
`param.main_grad` in place of parameter data. Two fixture properties keep the comparison
meaningful and are revalidated on every run: deterministic per-token reference log-prob
offsets separate the frozen reference from the initial policy (without them
delta == 0 and every DPO pair weight, accuracy, and margin is degenerate), and the
checkpoint's router weights are scaled so every biased expert selection clears a minimum
margin, since a near-tie route that flips between independent fp32 implementations would
look exactly like a backward-pass mismatch. Each rank computes the HF reference on its own
GPU, so compared gradients differ only by Megatron-vs-eager kernel arithmetic; a CPU
reference on rank 0 doubles as a fixture-validity guard (routing flips jump ~1e-2, ordinary
device drift stays ~1e-4).

**Optimizer update.** Fixed, named gradients are injected into the production path —
`MegatronStrategy.optimizer_step` over the DistributedOptimizer the trainer builds for DPO
(AdamW, lr 4e-6, betas (0.9, 0.999), weight decay 0, clip 0.5, bf16 parameters with fp32
shards) at PP2/EP2/DP2 on four GPUs. Two consecutive steps compare every owned parameter
slice's value, `exp_avg`, `exp_avg_sq`, the step counter, the returned gradient norm, and
the bf16 copy-back against the literal float64 reference; the owned slices must tile every
parameter exactly once across ranks. Both a clipping-active and a clipping-inactive case
run.

**Transfer to inference.** Exactly one DPO optimizer step runs through the Ray policy
worker at PP2; the updated training tensors are broadcast into disaggregated vLLM engines
(TP1/DP2/EP2) over the NCCL path. Every named serving tensor must match the training
snapshot byte for byte (fp32 router bias and router weight, bf16 parameters, stacked-expert
slices, vocabulary padding), and trainer-vs-vLLM token log-probabilities on fixed inputs
must agree before the update and after the synchronization, with the post-sync values
required to differ from the pre-update ones.

## Tolerances (registered before execution)

| quantity | bound |
| --- | --- |
| gradients, fp32 Megatron vs fp32 HF (per-rank GPU) | rtol 2e-4, atol 3e-4 |
| DPO metric rows (loss, accuracy, margin, rewards) | abs 2e-3 |
| HF reference CPU-vs-GPU log-probs (fixture route guard) | abs 1e-3 |
| minimum biased-router selection margin of the fixture | > 5e-3 |
| optimizer parameter/moments vs float64 reference | rtol 1e-5, atol 1e-8 |
| returned gradient norm vs independent global norm | rel 1e-4 |
| bf16 copy-back into gradient-buffer parameters | exact (rtol 0, atol 0) |
| serving tensor readback vs training snapshot | exact (rtol 0, atol 0) |
| trainer-vs-vLLM log-probs (bf16 kernels) | max abs 2e-1, mean abs 3e-2 |

## Scope limits

The BFCL production topology was TP1/PP15/CP1/EP8 bf16 on 120 H100s. The backward
experiment runs TP1/CP1 (the activation condition of the output projection) at fp32 with
the unfused attention backend so gradient agreement is tight; the production bf16 kernels
are covered separately by the forward parity gates in `tests/gpu/test_grug_megatron.py`.
Pipeline and expert degrees here do not reach 15/8, so production-degree equivalence is
not established by this suite alone.

## Running

On an otherwise idle H100x8 allocation (the backward experiment's largest topology needs
eight GPUs; the other two fit in four):

```bash
bash skyrl-train/ci/dpo_equivalence/run_equivalence.sh
```

`DPO_EQUIVALENCE_TESTS` and `DPO_EQUIVALENCE_PYTEST_ARGS` override the pytest targets and
flags the way `GRUG_MEGATRON_TESTS` does for the nightly Grug gate. The script installs the
frozen root profile (`uv sync --frozen` with the megatron extra), runs through the
`run-grug-gpu-gate` environment contract, writes `dpo-equivalence-junit.xml`, and fails if
any experiment was skipped rather than executed.

## Receipts

Recorded from the validation run on the cw-rno2a cluster; see the job log for the full
per-tensor discrepancy tables (`gradient ...`, `optimizer ...`, `... logprob parity` lines).

- Job: pending first execution.
- GPU/runtime: pending.
- Revisions: pending.
- Input batch sha256: printed by the tests at the start of each run.
- Measured discrepancies: pending.
