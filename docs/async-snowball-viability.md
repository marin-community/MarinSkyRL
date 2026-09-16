# Asynchronous Snowball viability

## Verdict

The JAX-first learner interface is qualified for the selected full-size asynchronous Snowball workload. On five
exclusive H100x8 nodes, the exact 67B-A2B M10 gate used 32 learner GPUs and eight vLLM GPUs to complete five
parameter-changing updates, publish every trained policy, admit overlapping data at ages zero through four, generate
from final policy 5, run fixed initial and final evaluations, and commit its final checkpoint.

A second allocation restored that exact policy-5 checkpoint and continued through policy 20. All 15 resumed updates
published completely and stayed within the age-four limit. Fixed pass@1 rose from `0.16796875` at restored step 5 to
`0.359375` at step 20. The saved iterator lacked epoch UIDs and reset on restore, so the continuation may have reused
the first 160 shuffled rows. It is bounded evidence that the system can learn while operating asynchronously, not a
clean uninterrupted data-consumption comparison.

The largest measured bottleneck is policy publication. Expert-aware scatter reduced a complete 67B publication from
443.399 seconds to about 175 seconds, a 60.5% reduction. Publication still takes roughly four times the usual 43-46
second Levanter optimizer step. No further simple change was identified that would preserve the verified receiver
protocol, so this experiment does not add a more complex publication layer.

A matched five-update Megatron control completed the same 32+8 workload and async contract. Its trained-policy
publication averaged 16.074 seconds, versus 175.144 seconds for Levanter, and its warmed optimizer averaged 6.794
seconds. This localizes the remaining performance gap to the Levanter learner's export and transport path rather than
the shared queue, staleness, or evaluation machinery. Both backends improved on the fixed evaluation, but their native
parameter, optimizer, and pipeline representations differ, so the small quality deltas are not used to rank them.

The semantic boundary is narrow, but the current integration is not modest in code shape. The external learner is
optional inside a trainer built around Torch actors, so construction, publication, checkpointing, export, and shutdown
contain repeated learner-specific branches. Moving the Torch backend behind the same protocol could remove that
branching, but it is a backend retrofit rather than a prerequisite for the demonstrated workload. The generic async
entrypoint now makes one explicit runner choice: HTTP or multi-turn configurations use the text/HTTP runner, while a
direct-engine configuration uses the token-preserving base runner. Mixed settings and missing exact behavior evidence
still fail closed.

## Selected workload

The controlled run keeps Ahmad's M10 semantics and the measured full-size shape.

| Setting | Value |
| --- | --- |
| Model | `marin-community/grug-67b-a2b-sft-s2-thinking-step630` at `6808fe5c219471517bd51df35addefd38ebebf89` |
| Train data | 1,024-row GSM8K parquet, identity `users/ahmad/documents/async-rl-snowball-gsm8k@2026.09.06.13:4c7ebfc9` |
| Evaluation data | 256-row provisional Snowball mechanical battery, content SHA-256 `cb50e2b55a9cc12e9ca95e8e814c47a479d377ba720235e4784c81f267eb343a` |
| Geometry | 32 prompts/update, four responses/prompt, 128 trajectories/update |
| Objective | regular clipped GRPO; effective equal-sequence weighting from one-sequence device microbatches; clip `0.2/0.2`; group standard-deviation normalization; one update epoch |
| Behavior filter | sampled vLLM probabilities required; `regular_mask` mismatch ratio in `[0.5, 5.0]`; veto below `1e-5`; no renormalization |
| Optimizer | AdamW, LR `1e-6`, betas `0.9/0.999`, epsilon `1e-8`, weight decay `0.01`, max norm `1.0`, constant schedule |
| Async policy | 160 generation workers, buffer 32, abort/resume, maximum age 4, publication after every update |
| Parsing | one-turn GSM8K, post-thinking native answer, parser-only reward |
| Learner | four nodes / 32 H100s, FP32 parameters and device-resident Adam state, Levanter PP4/EP8 |
| Serving | one node / eight H100s, vLLM TP1/DP8/EP8, BF16, temperature `1.0` |
| Excluded | KL loss/reward, critic, TIS, packing, colocation, and reward shaping |

The launcher stages the model and both parquet sources from the regional object store, checks their immutable
identities, records the resolved configuration, and uses fresh temporary and durable prefixes. Policy 0 can be adopted
without copying weights only when serving and learner identities match exactly. Every trained policy requires complete
ordinary-weight and expert-slice receipts before it becomes installed.

## Interface boundary

MarinSkyRL owns generation workers, completed and retry queues, group admission, the age-four limit, rewards,
advantages, evaluation, and run lifecycle. Iris allocates the nodes. Levanter owns the model, optimizer, random key,
JAX mesh, distributed update, native checkpoint, and policy export.

The workload required four narrow interface additions:

- `LearnerBatch` can require finite rollout log probabilities and carry an installed policy version for each selected
  response token. A compact list of contiguous version spans is expanded only at the learner boundary.
- vLLM records the installed version when each response segment produces its first token. An aborted response that
  resumes after publication keeps both real spans; the trainer does not invent one row-wide version.
- The learner distinguishes updated, publishing, installed, and failed policy state. A complete older installed policy
  remains available while learning runs, but generation pauses during the partial installation bracket.
- The async trainer asks the learner for independent old-policy log probabilities, applies the configured admission and
  `regular_mask` rules, and submits unchanged scores to the differentiated update. Missing probabilities, uncovered
  version spans, versions newer than the scored learner, and incomplete publication receipts fail closed.

Unsupported Levanter combinations are rejected before Ray actors or inference engines are allocated. Stateful fake
tests cover delayed groups, failed and in-progress publication, strict probability requirements, per-token provenance,
and checkpointed queues. CPU numerical tests compare the regular-mask loss and gradient with an independent PyTorch
calculation for unequal response lengths and masks. The Levanter tests also require the old forward to be independent
from the differentiated current-policy calculation.

## Distributed and full-size evidence

Iris job `/romain/dev-gpu-asyncsnow-dist2-01a0a404` first exercised the interface across two physical hosts. It
completed five finite updates and six publications, consumed policy-0 data at ages one through four, overlapped eight
generation intervals with distributed learning, matched sampled vLLM and JAX log probabilities within
`0.00125`-`0.00248`, and generated a final response whose complete provenance named installed policy 5. This proves
the distributed interface and serialization boundary, but not 67B memory, publication time, or learning quality.

The full-checkpoint campaign then exposed and corrected the limiting costs:

1. Raising the publication chunk bound from 64 MiB to 2 GiB reduced the complete checkpoint to 79 chunks. Pipelining
   host materialization with transfer reduced publication from 781.365 to 443.399 seconds.
2. Replacing a discarded differentiated optimizer program with an independent forward-only old scorer removed its
   101.48 GiB rematerialized peak. Full 128-trajectory scoring later completed in about 13-16 seconds.
3. Splitting gradient calculation from AdamW, keeping the device-resident optimizer state across executables, and using
   a fragmentation-safe JAX allocator made the 32-GPU FP32 M10 learner fit reliably.
4. Sending each expert slice only to its owning vLLM receiver reduced complete publication to about 175 seconds while
   preserving verification of 19,968 expert slices on all eight receivers.

## Qualified five-update gate

Iris job `/romain/snowball-67b-async-m10-gate-r9-01a0a404` ran commit `1fadaf73162f30b7125cab20fd9526a888a82572`
with 32 learner and eight vLLM GPUs. It completed in 2,173.18 seconds and consumed 24.146444 H100-hours.

| Update | Response tokens | Mean/max age | Optimizer seconds | Publication seconds |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 73,372 | 0/0 | 45.442 | 178.64 |
| 2 | 89,562 | 1/1 | 39.997 | 172.20 |
| 3 | 119,891 | 2/2 | 42.803 | 173.73 |
| 4 | 161,524 | 3/3 | 43.382 | 174.79 |
| 5 | 186,123 | 4/4 | 43.344 | 176.36 |

Each optimizer step changed parameters. Each publication installed the matching version on all receivers and verified
all 19,968 expert slices. Generation after every publication used the new installed policy while the buffered groups
used by later updates retained their real older provenance. No group older than four was admitted.

Fixed initial evaluation was pass@1 `0.13671875` and average score `-0.7265625` over all rows; G07 was
`0.1640625/-0.671875` and G08 `0.109375/-0.78125`. Final policy 5 generated before the terminal evaluation. The final
result was `0.1796875/-0.640625` over all rows, G07 `0.234375/-0.53125`, and G08 `0.125/-0.75`.

## Fifteen-update continuation

Iris job `/romain/snowball-67b-async-m10-horizon20-r5-01a0a404` restored the gate's exact step-5 policy and optimizer
state. The restored policy was republished and evaluated before new generation. Updates 6-10 independently repeated a
five-update age sequence of zero through four, and updates 11-20 stayed at or below age four. All 15 updates changed
parameters and published versions 6-20 with complete expert receipts.

The usual optimizer time was 43-46 seconds and publication stayed at 177-180 seconds. Update 19 reported a one-off
13.020-second optimizer time and is not used as a steady-state estimate. The final version-20 policy was installed and
verified before evaluation. The checkpoint saved 1,461 arrays, trainer state, and data-consumption state and committed
the `global_step_20` completion marker.

| Fixed evaluation | Step 5 | Step 20 | Change |
| --- | ---: | ---: | ---: |
| All pass@1 | 0.16796875 | 0.359375 | +0.19140625 |
| All average score | -0.6640625 | -0.28125 | +0.3828125 |
| G07 pass@1 | 0.203125 | 0.453125 | +0.25 |
| G08 pass@1 | 0.1328125 | 0.265625 | +0.1328125 |

The application completed before Iris teardown, but the task-runtime wrapper remained alive with only idle Ray
processes. After more than five minutes it was explicitly completed to avoid wasting the allocation. Iris recorded all
five tasks as successful with exit 0, no failures, retries, or preemptions. The longest task duration was 5,001.64
seconds, or 55.573778 H100-hours.

## Matched Megatron control

Iris job `/romain/snowball-67b-async-m10-megatron-control-r2-01a0a404` used the same model, immutable data,
five-update horizon, 32+8 topology, batch and context geometry, objective, async bounds, publication frequency, and
fixed evaluation. It changed the learner to native Megatron PP2/EP8 and its NCCL publication path. All five Iris tasks
succeeded with exit 0, no failures, retries, or preemptions. The longest task took 1,480.52 seconds and consumed
16.450222 H100-hours.

| Update | Response tokens | Mean/max age | Optimizer seconds | Publication seconds |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 68,771 | 0/0 | 17.078 | 15.866 |
| 2 | 93,284 | 1/1 | 6.271 | 17.625 |
| 3 | 137,058 | 2/2 | 8.105 | 14.774 |
| 4 | 156,821 | 3/3 | 6.397 | 17.280 |
| 5 | 182,559 | 4/4 | 6.401 | 14.823 |

Every optimizer step changed parameters and published its matching policy. The initial publication took 15.289
seconds. Exact sampled-token/logprob alignment passed, no group was rejected as stale, and the five consumed batches
had ages zero through four. Final-policy generation completed before the terminal evaluation. All 32 learner ranks
then uploaded their distributed policy and optimizer shards to the isolated `global_step_5/policy` prefix.

| Fixed evaluation | Policy 0 | Policy 5 | Change |
| --- | ---: | ---: | ---: |
| All pass@1 | 0.15625 | 0.18359375 | +0.02734375 |
| All average score | -0.6875 | -0.6328125 | +0.0546875 |
| G07 pass@1 | 0.2265625 | 0.2578125 | +0.03125 |
| G08 pass@1 | 0.0859375 | 0.109375 | +0.0234375 |

The Levanter and Megatron gates processed 630,472 and 638,493 response tokens, respectively. Mean trained-policy
publication was 175.144 seconds for Levanter and 16.074 seconds for Megatron: **10.90x faster**, or **90.8% lower**.
The usual Levanter optimizer step was 43-46 seconds; after Megatron's first-step warmup, steps 2-5 averaged 6.794
seconds. This is a matched system and workload comparison, not identical learner numerics: Levanter used PP4/EP8 with
FP32 parameters and device-resident Adam state, while Megatron used its native PP2/EP8 parameter and optimizer
representation. The fixed 256-row evaluation improved for both backends but is too small to rank learning quality from
the observed deltas.

## Revision provenance

The branch starts from MarinSkyRL `659cf49f7aec4f109ecffe5d21a90d8205ea5506`. The qualified five-update gate ran
`1fadaf73162f30b7125cab20fd9526a888a82572`. The continuation ran
`64dc1743b4a15b3873cbf91ac1ac59293993ca2a`, which added launch-time
`XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`. The first matched Megatron attempt ran
`81bfac6d76c0164ac11240a05b5423cc0ba224a1` and failed closed because the generic entrypoint selected an HTTP runner
that could not preserve sampled token IDs and log probabilities. The successful control ran
`c96070add26cecfe74facb642749a428a1180fbf`, which selects the direct runner for a direct-engine configuration without
weakening that evidence gate. The selected behavior and launcher were resolved against Marin
`a453edb92b14da9a8406c005f6bc4c76d72bc34b`.

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
| 13 | 10.894667 | Full 32+8 FP32 attempt exposed expert-scatter receiver ownership bug |
| 14 | 1.479267 | Short validation exposed incomplete restored recipe |
| 15 | 24.146444 | Qualified five-update 32+8 gate, final-policy generation/evaluation/checkpoint |
| 16 | 3.716111 | Resume path was parsed as a Hydra override because the TTL path was not quoted |
| 17 | 3.631444 | Second resume-path quoting attempt failed before restore |
| 18 | 11.482222 | Exact restore and initial publication passed; fragmented JAX BFC allocator failed update 6 |
| 19 | 0.229222 | Hydra-only allocator override did not affect pre-JAX process setup; stopped promptly |
| 20 | 55.573778 | Restored continuation completed updates 6-20 and fixed final evaluation |
| 21 | 4.203333 | Matched Megatron preflight rejected an HTTP runner that could not preserve exact behavior evidence |
| 22 | 16.450222 | Matched Megatron five-update control, final evaluation, and distributed checkpoint passed |
| **Total** | **224.067888** | **63.932112 H100-hours remain** |

## Acceptance

| Requirement | Full-size evidence |
| --- | --- |
| Five consecutive real updates | Passed at updates 1-5 and again at 6-10 |
| Generation overlaps learning | Passed; buffered generations reached real ages 0-4 |
| Staleness bounded at four | Passed through update 20; no too-stale admission |
| Publication after every update | Passed for trained policies 1-20 |
| Complete receiver verification | Passed for all eight receivers and 19,968 expert slices per publication |
| Generation from final gate policy | Passed at version 5 before terminal evaluation |
| Fixed initial/final evaluation | Passed for the five-update gate and 15-update continuation |
| Bounded learning evidence | Passed with the iterator-reset caveat described above |
| Matched Megatron comparison | Passed on the same 32+8 shape and five-update contract |

The interface is viable for this selected workload. Broader production cleanup should put Torch training behind the
same learner protocol and collapse repeated lifecycle branches, but that refactor should remain separate from the
attributable numerical and runtime evidence reported here.
