# Asynchronous Snowball viability

## Verdict

The JAX-first learner interface can express the asynchronous Snowball workload without moving queueing, rewards,
staleness, or publication ownership into Levanter. A real two-host tiny-model gate completed five optimizer updates,
consumed policy-0 data at ages one through four, overlapped vLLM generation with JAX learning, installed six complete
policies, and generated with final policy version 5.

The exact 67B-A2B M10 gate is **not qualified**. Its last allocation completed the first BF16 gradient and AdamW
update on both eight-H100 learner hosts, but the driver rejected the two results because their host-local timing
measurements differed. It therefore did not publish trained policy 1 or run updates 2-5. There is no final evaluation,
learning comparison, or matched Megatron throughput claim.

## Interface boundary

MSRL still owns generation workers, completed and retry queues, group admission, the age-four limit, rewards,
advantages, and checkpoints. Iris allocates the nodes. Levanter owns the model, optimizer, random key, JAX mesh,
distributed update, native checkpoint, and policy export.

The workload required four narrow interface additions:

- `LearnerBatch` can require finite rollout log probabilities and carry an installed policy version for each selected
  response token. A compact list of contiguous version spans is expanded only at the learner boundary.
- vLLM records the installed version when each response segment produces its first token. An aborted response that
  resumes after publication keeps both real spans; the trainer does not invent one row-wide version.
- The learner distinguishes updated, publishing, installed, and failed policy state. A complete older installed policy
  stays available while learning runs, but generation is paused during the partial installation bracket.
- The async trainer asks the learner for independent old-policy log probabilities, applies the configured admission and
  `regular_mask` rules, and then submits the unchanged scores to the differentiated update. Missing probabilities,
  uncovered version spans, versions newer than the scored learner, and incomplete publication receipts fail closed.

Unsupported Levanter combinations are rejected before Ray actors or inference engines are allocated. The selected path
does not add KL, critic, TIS, sample packing, colocation, or Torch-backend behavior.

## Selected M10 workload

The bounded gate uses the measured regular-policy shape rather than the launcher's `behavior_clip` default.

| Setting | Value |
| --- | --- |
| Model | `marin-community/grug-67b-a2b-sft-s2-thinking-step630` at `6808fe5c219471517bd51df35addefd38ebebf89` |
| Train data | 1,024-row GSM8K parquet, identity `users/ahmad/documents/async-rl-snowball-gsm8k@2026.09.06.13:4c7ebfc9` |
| Evaluation data | 256-row provisional Snowball mechanical battery, content SHA-256 `cb50e2b55a9cc12e9ca95e8e814c47a479d377ba720235e4784c81f267eb343a` |
| Geometry | 32 prompts/update, four responses/prompt, 128 trajectories, five updates |
| Objective | regular clipped GRPO, token mean, clip `0.2/0.2`, group standard-deviation normalization, one update epoch |
| Behavior filter | sampled vLLM probabilities required; `regular_mask` mismatch ratio in `[0.5, 5.0]`, veto below `1e-5`, no renormalization |
| Optimizer | AdamW, LR `1e-6`, betas `0.9/0.999`, epsilon `1e-8`, weight decay `0.01`, max norm `1.0`, constant schedule |
| Async policy | deep pool, abort/retry, maximum age 4, explicit publication after updates 1 and 5 |
| Parsing | one-turn GSM8K, post-thinking native answer, parser-only reward |
| Learner | two nodes / 16 H100s, BF16 parameters and compute, FP32 loss outputs, host-resident Adam state between split executables |
| Serving | one node / eight H100s, vLLM TP1/DP8/EP8, BF16, temperature `1.0` |

The launcher stages the model and both parquet sources from the regional object store, records their immutable
identities, and uses fresh output prefixes. Policy 0 can be adopted without copying weights only when the serving and
learner identities match exactly. Trained policies still require complete ordinary-weight and expert-slice receipts.

## Correctness and async evidence

The stateful fake covers delayed groups, failed and in-progress publication, strict probability requirements,
per-token provenance, and checkpointed queues. CPU numerical tests compare the regular-mask loss and gradient with an
independent PyTorch calculation for unequal response lengths and masks. The Levanter tests also require the old forward
to be independent from the differentiated current-policy calculation.

Iris job `/romain/dev-gpu-asyncsnow-dist2-01a0a404` exercised two physical hosts at commit
`ebeb3b2d48386791dcaf9eae305435c466e08182`. One learner and one vLLM rank ran on each host. The test passed in 127.73
seconds:

- five finite, parameter-changing updates ended at learner, update, and installed version 5;
- six publications completed, including the initial policy, with receiver receipts;
- updates 2-5 consumed genuinely older policy-0 trajectories at ages 1, 2, 3, and 4;
- eight generation intervals of 3.67-7.49 seconds overlapped the first 10.68-second distributed learning phase;
- sampled-token vLLM-versus-JAX absolute log-probability differences were at most `0.00125`-`0.00248`, and mismatch
  ratios stayed within `0.9982`-`1.0025`, so this tiny gate masked no tokens;
- independently recomputed old scores matched the following differentiated forward bit exactly, and every pre-update
  PPO ratio was one; and
- the final vLLM response carried one complete provenance span for installed policy 5.

This proves the exercised distributed interface and serialization boundary. It does not make the tiny model a proxy for
67B memory, publication time, or learning quality.

## Full-checkpoint evidence

The full allocations exposed and reduced three concrete costs:

1. A 64 MiB publication transferred only 127.33 GB in 2,620.24 seconds. Raising the bound to 2 GiB reduced the complete
   checkpoint to 79 chunks. Pipelining host materialization with transfer then reduced total publication from 781.365
   to 443.399 seconds, a 43.3% reduction. The successful install verified 134,157,778,944 bytes and 19,968 expert
   projection slices across all eight serving workers.
2. Replacing a discarded differentiated optimizer program with an independent forward-only old scorer removed its
   101.48 GiB rematerialized peak. The full 128-trajectory score completed in 12.95-15.63 seconds in later runs.
3. Separating gradient calculation from AdamW application and retaining optimizer state in pinned host memory was not
   sufficient while parameters and gradients remained FP32. Native BF16 storage cleared that floor. In
   `/romain/snowball-67b-async-m10-gate-r7-01a0a404`, both learner hosts returned from the first full update in 103.78
   seconds after generating 72,343 response tokens in 15.785 seconds and scoring them in 13.89 seconds.

The r7 failure happened after both actors updated. `_merge_update_results` correctly checks semantic metrics across
replicas, but it also compared the independently measured `gradient_compute_seconds` values. Those wall-clock values
cannot be equal across hosts. The correction reduces all host-local update timings by maximum, which represents the
distributed critical path, while retaining strict comparison for loss and other semantic metrics.

No trained full-checkpoint policy was published in the asynchronous campaign. Earlier synchronous validation of the
same publication receiver remains useful background, but it does not satisfy this async gate, especially after the
expert-scatter optimization.

## Allocation ledger

Every allocation used non-preemptible H100 80GB nodes on `cw-rno2a`. Charges include whole reserved nodes and failed
attempts.

| Allocation | H100-hours | Result |
| ---: | ---: | --- |
| 1 | 0.747267 | Exposed missing thread-local JAX mesh context before update 1 |
| 2 | 0.716044 | Five updates passed; tiny generation drained before learning, so overlap assertion failed |
| 3 | 0.738778 | One-host five-update async contract passed, including final-policy generation |
| 4 | 0.595289 | Two-host holder exceeded an implicit 1.07 GB memory request before the test |
| 5 | 1.354889 | Two-host five-update async contract passed |
| 6 | 41.082667 | Stopped during unusably slow 64 MiB initial publication |
| 7 | 15.018444 | Full initial publication passed; evaluation metadata raised `KeyError: acc` |
| 8 | 7.485067 | Pipelined publication and evaluation passed; differentiated old scorer ran out of memory |
| 9 | 7.669267 | Forward-only old scoring passed; fused FP32 gradient and AdamW update ran out of memory |
| 10 | 6.957400 | Host-offloaded Adam state still re-entered the fused executable and ran out of memory |
| 11 | 5.122333 | Split Adam removed moment coexistence; FP32 gradient executable still ran out of memory |
| 12 | 4.773733 | BF16 update completed on both hosts; driver rejected unequal host timing metrics |
| **Total** | **92.261178** | **3.738822 H100-hours remained** |

The last remainder equals only 560.82 seconds on the required 24-GPU shape. Allocation 12 needed 716.06 seconds to
start and return its first update, so another credible five-update attempt did not fit the 96 H100-hour cap.

## Acceptance and next step

| Requirement | Tiny distributed gate | Exact 67B M10 gate |
| --- | --- | --- |
| Five consecutive real updates | Passed | Failed after update 1 actor execution |
| Generation overlaps learning | Passed | Not established across five updates |
| Genuinely older data, age at most 4 | Passed at ages 1-4 | Not reached |
| Repeated complete publication | Passed | No trained-policy publication |
| Generation from final policy | Passed at version 5 | Not reached |
| Initial/final quality comparison | Not a quality workload | No final result |

The smallest useful next investment is one fresh run of the unchanged five-update r7 workload after the timing merge
fix. It still needs 24 H100s. Allowing 45-50 minutes, or 18-20 H100-hours, covers startup, two measured approximately
7.4-minute full publications, five updates, and terminal evaluation with a cleanup margin. The run should not add a
matched Megatron comparison or longer learning horizon until this exact gate publishes versions 1 and 5, generates
from version 5, and produces its fixed final evaluation.
