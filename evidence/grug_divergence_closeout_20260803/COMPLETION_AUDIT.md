# Grug divergence closeout completion audit

This is a live requirement-by-requirement audit of
`notes/goals/2026-08-02-grug-eager-grouped-divergence-closeout.md`. A pending
row blocks completion. Evidence work here is disposable; MarinSkyRL PR #276
remains the product branch.

| Requirement | Status | Authoritative evidence |
| --- | --- | --- |
| Confirm usable local space before setup and avoid unrelated deletion | Proven | `df -h` at `2026-08-03T02:34:27Z` reported 5.5 GiB free. No local multiworker job was launched. Root `GROUNDING.md` records the sample. |
| Hold input, parameters, RNG, gradient state, topology, and rank-to-GPU mapping fixed | Proven | Localization artifact `s3://marin-us-east-02a/iris/grug-training-perf-gap/20260803/divergence-closeout-53d420e/localization-s3.json`; gate artifact `s3://marin-us-east-02a/iris/grug-training-perf-gap/20260803/divergence-closeout-2dd905e/preflight-paired-s1.json`; independent gate summary SHA-256 `afaa41ec95b1dc17410fb78562132dfec04e1f9ecb8a8365596db07b39bca38d`. |
| Localize the first differing block and operation | Proven | All ranks first differ in layer 0 after the expert combine. Product routed input, per-expert counts, grouped output, routers, and the isolated selected-expert output and gradients are exact. Repeating only the final BF16 global `index_add_` differs by at most `0.001953125`. Localization canonical result SHA-256 is `d890dc2cb8938b6607b4a98a60cfc9476e187695dbc48f6dba599c8a15cb9782`. |
| Compare with a trusted reference and assign ownership | Proven | The manual grouped reference and direct selected-expert probe are bitwise equal to the product grouped path, including input, gate, up, and down gradients. Only BF16 global accumulation ordering differs. This classifies the divergence as acceptable BF16-sensitive execution, not a grouped-runtime product defect. See `SEMANTIC_GATE.md`. |
| Record one coherent semantic route contract before the confirmatory gate | Proven | `SEMANTIC_GATE.md` was recorded at `2026-08-03T00:33:38Z`, before the gate. It fixes output, CE, route-margin, direct-gradient, full-gradient finiteness, identity, and oracle rules, including the action on failure. |
| Attempt at most one owned substantive correction and one eight-H100 gate | Proven | No product correction was owned after localization. The one gate ran on `/romain/dev-gpu-romain-grug-gate-2dd905e` and passed. No second substantive gate ran. |
| Use the cheap discriminator before any blocked escape | Proven | Same-process block localization, manual grouped comparison, repeated-combine probe, and direct selected-expert forward/backward probes all ran before the confirmatory gate. |
| Keep #276 unchanged unless permanent grouped-runtime logic is wrong | Proven | Live PR #276 head is `0c213586b5491b8046ca7780e965c4b26dc6a2a2`. It is open, non-draft, clean, mergeable, green, and unmerged. Diagnostic commits exist only on the evidence branch. |
| Gate validity under the predeclared contract | Proven | The gate has exact eager oracle behavior; grouped/eager CE and action-log-probability allowance ratios `0.441190` and `0.559959`; 476 finite nonempty gradient tensors and at least 8,384,693,120 gradient elements per rank; 0 unexplained changed-route tokens across 208 rank-layers. See `RUN_LOG.md`. |
| Run exactly one sequential eager/grouped matched-CE forward-plus-backward pair on the same 32 H100s, without optimizer | In progress | Iris job `/romain/grug-paired-eager-grouped-2dd905e-s1-20260803` is running at production priority on four gang-scheduled `cw-rno2a` nodes. Source `2dd905e29597b848f912dd5cdaf2cebdfbf3d0c2`; image digest `sha256:24c655d33ebb6ef78b9f9a5db4053f838c2e9d6c98e3adef338cdb87e1c072a2`. |
| Independently read back correctness, synchronized wall, routed and non-routed time, throughput, peak HBM, GPU-seconds, limits, and source chain | Pending | Frozen result URI is `s3://marin-us-east-02a/iris/grug-training-perf-gap/20260803/divergence-closeout-2dd905e/headline-paired-s1.json`. Independent reader SHA-256 is `78b2133facfa1dbe4d37d931a38f1d0496b496639d33c51cf5e6ca8e35279717`; it will run only after the paired job is terminal. |
| Run ordinary validation, then four High `KIND=GOAL` reviewers and address material findings | Pending | Reviewer dispatch is intentionally held until measurement readback and final validation are complete. |
| Publish final evidence to #7903 and any affected #276 surface | Pending | #7903 is open with no comments. #276 needs no product update unless final evidence changes the classification. |

The source-to-image-to-artifact chain is frozen as source
`2dd905e29597b848f912dd5cdaf2cebdfbf3d0c2`, image
`ghcr.io/marin-community/marinskyrl@sha256:24c655d33ebb6ef78b9f9a5db4053f838c2e9d6c98e3adef338cdb87e1c072a2`,
and the result URI above. Completion remains unproven until every pending row
has terminal evidence.
