# Completion audit

Audit time: `2026-08-05T01:24:32Z`

## Goal identity and immutable evidence

- Controlling goal:
  `notes/goals/2026-08-04-grug-fp32-combine-paired-measurement.md`.
- #276 parent: `0c213586b5491b8046ca7780e965c4b26dc6a2a2`.
- Evidence-only candidate: `fbb1fc8378601e0346d00d186809f10d1ad0360d`.
- Frozen protocol, driver, reader, and preflight commit:
  `9ea3f4b17e9387e8c7ccf350f5a01e26385713a0`.
- Completed raw packet and generated summaries commit:
  `965306a595709540c5c1d993aca84542c5c93c54`.
- Reviewed human report commit:
  `40d365d661f3acc411a78b8801000a4eae964512`.
- Validation and publication-draft commit:
  `8280934139af800b1707f49479fde818046e92c6`.
- Raw packet SHA-256:
  `60e882a908643ae487b6f2f2a1c0c979a6a3d10bb9dccd97c40601c98e02526e`.
- [Immutable reviewed evidence packet](https://github.com/marin-community/MarinSkyRL/tree/40d365d661f3acc411a78b8801000a4eae964512/evidence/grug_fp32_combine_paired_20260805).

## Requirement-by-requirement result

| Requirement | Evidence and result |
| --- | --- |
| Freeze pins, fixture, rules, finite schedule, safety stops, and verdict before live results | `FROZEN_PROTOCOL.json`, `run_pair.py`, `verify_packet.py`, and `PREFLIGHT.md` were committed before the successful run. The verdict was estimate-only; no materiality threshold was invented afterward. |
| Compare exact parent and candidate in one pinned runtime | One image and package inventory loaded the exact parent and candidate module bytes. State, hidden inputs, routes, combine weights, and cotangent were byte-identical. Source and transformation hashes passed the verifier. |
| Preserve BF16 multiplication and isolate accumulation precision | Both arms formed BF16 expert-output/weight products. Their affected operations differed only in running-sum precision. The primary full-block measurement called each hash-pinned product module; the isolated combine boundary reproduced the product's unexposed affected operations in the driver. |
| Independent correctness oracles | Fixed-slot-order FP32 and FP64 accumulators consumed the common BF16 summands and did not call either product combine implementation. Parent fixture discrimination was `4,096 / 524,288` differing elements, maximum `0.015625`; candidate output was exact against FP32 and BF16-cast FP64. |
| Cover eager/grouped outputs and combine-relevant gradients | Candidate eager and grouped outputs were bitwise equal. Combine-weight, gate, up, and down gradients were bitwise equal; hidden-gradient maximum difference was `2.384185791015625e-7`. Every frozen gradient tolerance check passed. |
| Report the FP64 comparison at its supported scope | Returned BF16 output versus the independent uncast FP64 accumulator improved from parent maximum/mean `0.01171875 / 0.000091552734375` to candidate `0.00390625 / 0.000030517578125`. Review found that the frozen fifth gate's “pre-cast” label was too strong because the internal product accumulator was not retained. The human report and #7903 comment withdraw that stronger interpretation; the frozen packet remains unchanged. |
| Real Grug shape and paired H100 design | The fixture was 8,192 tokens, hidden 2,560, intermediate 1,280, 256 experts, top-4, with 128 routed rows per expert. Every physical GPU ran both arms twice in alternating order. Five warmups and twenty retained iterations per process produced 40 samples per arm/GPU. |
| Synchronized full-block and complete-combine timing | CUDA events enclosed forward and backward and synchronized completion; setup and gradient clearing were outside. Full-block F+B candidate-parent median was `+0.537424` ms (`1.038439x`), range `+0.498303` to `+0.545664` ms; all eight GPUs were slower. Complete-combine F+B median was `+0.531640` ms (`1.340920x`). |
| HBM baseline, peak, and increment | Full-block incremental peak was `5,700,675,584` bytes for both arms because another part of the block set the peak. Complete-combine incremental peak rose from `545,390,592` to `796,983,296` bytes: `+251,592,704` bytes (`1.461307x`) on every GPU. |
| Preserve raw paired data and compact readback | `raw_packet.json`, `summary.json`, and `SUMMARY.md` are committed. The standard-library verifier passed 344 checks and regenerated JSON, Markdown, and stdout byte-for-byte. Their SHA-256 values are recorded in `VALIDATION_AND_REVIEW.md`. |
| Bound resource use and release promptly | Iris supplied one `h100-8x` node in `cw-rno2a` at interactive priority. The maximum cgroup sample was `25,553,707,008` bytes, minimum host available memory was `2,063,631,482,880` bytes, and the remote host had no swap. The holder was released after readback. Final `dev_gpu.py status` found no active session, and Iris reports the holder job `killed`, reason `Terminated by user`. |
| Run ordinary validation and three High GOAL reviews | Driver self-test, Python compilation, pinned Ruff, JSON parsing, `git diff --check`, exact candidate CPU regression, packet verification, and byte comparisons passed. Claude and Gemini returned pass verdicts. Codex's material scope finding was addressed in human-facing text, followed by a successful full local revalidation. Review hashes and findings are in `VALIDATION_AND_REVIEW.md`. |
| Publish detailed #7903 evidence, then short #276 note | [#7903 detailed comment](https://github.com/marin-community/marin/issues/7903#issuecomment-5186445223) was posted first. Its exact API body matches `ISSUE_7903_COMMENT.md`: 8,186 bytes, SHA-256 `89bf5f75dfc6d1416f615653708ce9288d336bfac389a7c30f32a81187ea5c70`. [#276 short comment](https://github.com/marin-community/MarinSkyRL/pull/276#issuecomment-5186447968) followed and matches `PR_276_COMMENT.md`: 1,449 bytes, SHA-256 `08f6a7dea24457a4190e81bce5d1711b901270f5394503c0d6f0f7aeb075ee15`. |
| Keep product changes off #276 and preserve the distributed verdict | Final GitHub readback shows #276 open, non-draft, exactly at `0c213586b5491b8046ca7780e965c4b26dc6a2a2`; its four commits do not include `fbb1fc8`. Its exact-head checks are green. Its separate `CONFLICTING` / `DIRTY` state was not touched. #7903 remains open and the failed `1 / 12,288` distributed action-output gate remains controlling. |
| Respect non-goals | No live replay, distributed training run, attention investigation, R3/Levanter expansion, production benchmark switch, or product branch change was made. Old unpaired throughput was not used as causal evidence. |

## Final state

The bounded measurement is complete and reviewable. It supports the candidate's
local fixed-route FP32-combine contract, while showing a consistent time and
combine-HBM cost on this fixture. It does not justify accepting `fbb1fc8`, does
not resolve #7903, and does not alter #276.
