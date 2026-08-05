# Validation and review

## Pre-measurement checks

Before the successful numerical run:

- The exact parent-to-candidate source transformation and both module hashes
  matched `FROZEN_PROTOCOL.json`.
- `run_pair.py self-test`, Python compilation, project-pinned Ruff, JSON
  parsing, and `git diff --check` passed.
- The exact candidate test
  `test_bfloat16_expert_combine_uses_float32_accumulation` passed while importing
  `skyrl_train/models/grug_moe.py` from candidate worktree commit
  `fbb1fc8378601e0346d00d186809f10d1ad0360d`.
- The fixture, seed, numerical oracles and tolerances, finite four-wave schedule,
  safety stops, and estimate-only verdict were fixed before successful live
  results. `PREFLIGHT.md` records the two dependency-preflight failures and the
  narrow runtime-only corrections made before the final inventory freeze.

The frozen evidence sources are commit
`9ea3f4b17e9387e8c7ccf350f5a01e26385713a0`. The completed packet and initial
report are commit `965306a595709540c5c1d993aca84542c5c93c54`; the reviewed report
correction is commit `40d365d661f3acc411a78b8801000a4eae964512`. Both are pushed to
`origin/grug-eager-grouped-divergence-closeout-20260802`.

## Packet copy and independent readback

The local copies matched the remote files byte-for-byte:

| Artifact | SHA-256 |
| --- | --- |
| `raw_packet.json` | `60e882a908643ae487b6f2f2a1c0c979a6a3d10bb9dccd97c40601c98e02526e` |
| `summary.json` | `d2199f2f4bbd95cd0d1db0503192b36db3df47df2fa8130ea24119aee7dfc7dd` |
| `SUMMARY.md` | `1941341978a687f0a627bae57a3af437b75c5629e9812a61bba2fedf93cdebb5` |
| `preflight_import_failure.stderr` | `be9731322997f522f411db752aa38e864cf2d52df65cb7a0271d6a1808f0675f` |
| `preflight_torchtitan_failure.stderr` | `c0d1eb60a074fc2cc0a7f8466678a599ec4d93bcce006a813da71fb2510d33d9` |

The independent standard-library reader passed 344 checks. A fresh local
invocation against the copied raw packet regenerated `summary.json`,
`SUMMARY.md`, and its stdout with SHA-256 values
`d2199f2f4bbd95cd0d1db0503192b36db3df47df2fa8130ea24119aee7dfc7dd`,
`1941341978a687f0a627bae57a3af437b75c5629e9812a61bba2fedf93cdebb5`,
and `d2199f2f4bbd95cd0d1db0503192b36db3df47df2fa8130ea24119aee7dfc7dd`.
Each was byte-identical to the remote-derived counterpart.

After the copy and readback, `dev_gpu.py status` reported no local session,
Iris listed `/romain/dev-gpu-romain-grug-fp32-20260805` as `killed` with reason
`Terminated by user`, Kubernetes returned `NotFound` for the holder pod, and
the local allocate process exited.

## Post-measurement ordinary validation

The completed evidence passed:

```text
python evidence/grug_fp32_combine_paired_20260805/run_pair.py self-test
python -m py_compile <run_pair.py> <verify_packet.py>
skyrl-train/.venv/bin/ruff check <run_pair.py> <verify_packet.py>
jq empty <protocol and packet JSON files>
git diff --check
```

`run_pair.py self-test` printed `self-test passed`; Ruff printed `All checks
passed!`; the remaining commands were silent and exited zero. The exact
candidate CPU regression was rerun with the evidence worktree's pytest
environment from the candidate worktree. It first printed the imported module
path under that candidate worktree, then reported `1 passed, 18 deselected in
5.51s`.

Before publication, GitHub readback showed Marin #7903 open with the
route-aware closeout as its latest comment. MarinSkyRL #276 was open, non-draft,
and still exactly at `0c213586b5491b8046ca7780e965c4b26dc6a2a2`; its exact-head
checks were green, while mergeability was separately `CONFLICTING` / `DIRTY`.
It had no issue comments.

## Required High GOAL reviews

Claude, Codex, and Gemini independently reviewed the completed packet, report,
and both publication drafts under the High-tier `KIND=GOAL` procedure in
`~/llms/call_agents.md`.

| Reviewer | Review SHA-256 | Initial verdict |
| --- | --- | --- |
| Claude | `8f64d759525befc890fbc39e8f29379cc700ca056d781851cb8d05b4b5ecf85a` | Pass; publishable |
| Codex | `875884ca5c018a1f99f30ef99c50d4af9dfc7c862ab1fbb48bdfc531126948741` | Revise before publication |
| Gemini | `30c8c79f144988727b77c36fe93740b10a47544ea6d3e44bd3f49444925366ff` | Pass; approved |

Codex found one material scope error. The fifth frozen gate is labelled
"candidate maximum pre-cast error," but the driver did not retain the product's
internal pre-final-cast FP32 accumulator. Its recorded comparison uses the
returned BF16 product output and an independent uncast FP64 accumulator. The
measurement therefore supports a narrower returned-output accuracy result, not
a direct product pre-cast result.

The human-authored report and #7903 draft now state that distinction and do not
claim that all five frozen gates passed as written. The report also says that
the generated summary's "reduced local accumulation error" phrase must be read
at returned-output scope. The frozen protocol, driver, raw packet, reader, and
generated summaries were left byte-identical. No GPU rerun was needed because
the stronger interpretation was withdrawn; the goal requires a bounded FP64
output/gradient comparison, which the preserved packet contains.

Claude also asked that the report distinguish the hash-pinned full-block call
from the driver's faithful implementation of the otherwise unexposed isolated
combine boundary. That clarification is now explicit. Both Claude and Gemini
confirmed the reported values and source isolation; Claude additionally warned
that unrelated dirty product instrumentation must stay out of evidence commits.
Only named evidence files will be staged.

Review also exposed a timing detail worth stating plainly. Emptying the CUDA
allocator after warmup made the first retained iteration in each process a
visible allocation outlier. All samples remain in the packet. Each 40-sample
arm/GPU pool contains two such samples, neither of which determines its median.
The report and #7903 draft now disclose this.

After these edits, the packet reader again passed 344 checks. Its regenerated
JSON summary, Markdown summary, and stdout were byte-identical to the preserved
files, with SHA-256 values
`d2199f2f4bbd95cd0d1db0503192b36db3df47df2fa8130ea24119aee7dfc7dd`,
`1941341978a687f0a627bae57a3af437b75c5629e9812a61bba2fedf93cdebb5`,
and `d2199f2f4bbd95cd0d1db0503192b36db3df47df2fa8130ea24119aee7dfc7dd`.
The driver self-test, Python compilation, pinned Ruff, JSON parsing, and
`git diff --check` also passed.
