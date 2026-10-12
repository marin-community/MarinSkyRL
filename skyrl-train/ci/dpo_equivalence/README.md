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
`param.main_grad` in place of parameter data. Gradient reduction runs through DDP
allreduce (`use_distributed_optimizer=False`) finalized with `finish_grad_sync`, which is
idempotent under allreduce and leaves every rank's buffer the full data-parallel mean the
reference computes; the distributed optimizer's reduce-scatter shard path is verified
element-exactly by the optimizer experiment instead. Two fixture properties keep the
comparison
meaningful and are revalidated on every run: deterministic per-token reference log-prob
offsets separate the frozen reference from the initial policy (without them
delta == 0 and every DPO pair weight, accuracy, and margin is degenerate), and the
checkpoint's router weights are zeroed so biased expert selection is decided by the
constant router bias alone — a data-dependent router score could flip a near-tie route
between independent fp32 implementations and look exactly like a backward-pass mismatch,
and scaling the weights amplifies the score noise as much as the margin. Routing
mechanics stay fully exercised (per-token top-k dispatch, stacked experts, combine
weights, expert-parallel partitions) and the router weight still receives gradients
through the combine-weight softmax. Each rank computes the HF reference on its own
GPU, so compared gradients differ only by Megatron-vs-eager kernel arithmetic; a CPU
reference on rank 0 doubles as a fixture-validity guard (routing flips jump ~1e-2, ordinary
device drift stays ~1e-4). At the routing decisiveness the fixture needs, router-weight
gradients fall below the registered gradient atol, so the comparison is sensitive for
attention, expert, embedding, and projection tensors but not for the router weight itself
(the router bias is a buffer and never carries a gradient).

**Optimizer update.** Fixed, named gradients are injected into the production path —
`MegatronStrategy.optimizer_step` over the DistributedOptimizer the trainer builds for DPO
(AdamW, lr 4e-6, betas (0.9, 0.999), weight decay 0, clip 0.5, bf16 parameters with fp32
shards) at PP2/EP2 on four GPUs (dense data-parallel degree 1; dense tensors replicate
across the two EP ranks of each stage). Two consecutive steps compare every owned parameter
slice's value, `exp_avg`, `exp_avg_sq`, the returned gradient norm, and
the bf16 copy-back against the literal float64 reference; within each data-parallel group
the owned slices must cover one replica exactly once. The reference gradient norm sums the same owned slices production's norm sums —
shared/TP-duplicate filtering mirrored — so replication counts identically on both sides.
Both a clipping-active and a clipping-inactive case run.

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
| minimum biased-router selection margin of the fixture | > 5e-3 (constant bias gaps) |
| optimizer parameter/moments vs float64 reference | rtol 1e-5, atol 2e-7 |
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

## Findings

The verification itself surfaced two results worth recording beyond a pass/fail:

1. **Response-window slicing depends on the array's width convention.** The response
   log-probability window (the log-probs of the final `num_actions` tokens, predicted at
   `num_actions` predictor positions) is `[:, -num_actions-1:-1]` of a sequence-width
   array whose entry k scores the token at column k+1 (the convention used by
   `logprobs_from_logits` and the existing forward-parity helpers), but
   `[:, -num_actions:]` of an array whose last column was already dropped before
   gathering (the trainer's reconstruction and the reference here). This experiment's
   first reference used the wrong convention for its array and imprinted a clean,
   fixture-independent metric shift — tracing it required making routing deterministic
   (below) and dumping the objective's exact inputs. The trainer's own window was
   correct throughout; no production change was needed.
2. **Tiny random MoE fixtures need drift-free routing for cross-implementation
   comparison.** Near-tie biased selections flip between independent fp32 implementations
   (eager HF vs unfused Megatron) and imprint ~1e-2 per-token log-probability noise that
   looks exactly like a backward mismatch. Scaling router weights does not help — the
   selection margins and the score noise scale together. The fixture instead zeroes the
   router weights so biased selection is decided by the constant router bias alone.

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

- Job: full-suite receipt job `dpo-equivalence-suite-f57268174` and reruns on cw-rno2a; the
  diagnostics file (`dpo-diagnostics.log`) printed by the runner carries the full
  per-topology discrepancy tables.
- GPU/runtime: NVIDIA H100 80GB HBM3, torch 2.13.0+cu132, megatron-core 0.18.0
  (`0.0.0.dev20260929+marin.39e62869693c` frozen runtime).
- Revisions: branch `benfeuer/dpo-equivalence-20261011` at `f57268174` (merged with main
  `37ec019b7`); final receipt run at the pushed head of the PR branch.
- Input batch sha256: printed at each run start (fixture is deterministic).
- Backward gradients vs eager HF fp32 reference, worst `max_abs` over all named tensors per
  topology (bound rtol 2e-4 / atol 3e-4): pp1 `3.5e-6`, pp2 `3.5e-6`, pp2_dp2 `6.3e-5`,
  pp2_ep2 `6.3e-5`, pp2_ep4 `9.4e-5`. DPO metric rows agree within `3.6e-7`; eval-window
  policy log-probs within `7.6e-6` on the loss mask; reference CPU-vs-GPU drift
  `max_abs = 1.9e-6` (bound 1e-3).
- Optimizer and transfer receipts: recorded below from the final receipt run.
