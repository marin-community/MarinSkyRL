# Final validation and review

## Reproduction sources

The exact image-build, gate, headline, monitor, and original independent-readback
sources are preserved under `reproduction/`. Their recorded SHA-256 values match
the pre-run or executed values in `RUN_LOG.md`. The original diagnostic reader is
`read_paired_artifact.py`, SHA-256
`ac941919681b7b4c535cdf684b03640473dda79c8e2ae5f9120ade1e370b4d80`.

The stricter post-review verifier is `verify_headline_artifact.py`, SHA-256
`7c746b80af620575ba8a8bf743a78b0fd7b2777771b0f488822b7be9536f7616`.
It independently asserts the URI, payload and canonical result digests, source,
image, model revision, manifest, logical batch, and runtime-script digest. It
also recomputes all 12,288 action-log-probability checks, the CE check, and all
5,184 representative-gradient checks from the raw arm samples. It requires the
recomputed records to equal the embedded frozen-harness records.

CPU Iris job
`/romain/grug-paired-strict-readback-2dd905e-s1-20260803` succeeded. It confirmed
`embedded_semantic_check_matches_recomputed=true`, structural correctness pass,
and semantic failure with the same 1,995 action-output violations. Its launcher
SHA-256 is
`b13abab67c4d5242fe7bb04c9131991ef761e3dd5ba0ef36880ac67a94744106`.

## Source-to-image binding

Command:

```text
docker buildx imagetools inspect ghcr.io/marin-community/marinskyrl:gpu-rl-2dd905e29597b848f912dd5cdaf2cebdfbf3d0c2
```

Result at `2026-08-03T04:54Z`:

```text
Name:      ghcr.io/marin-community/marinskyrl:gpu-rl-2dd905e29597b848f912dd5cdaf2cebdfbf3d0c2
MediaType: application/vnd.docker.distribution.manifest.v2+json
Digest:    sha256:24c655d33ebb6ef78b9f9a5db4053f838c2e9d6c98e3adef338cdb87e1c072a2
```

The archived build launcher creates the tag from the exact source commit. The
registry readback binds that tag to the digest used by the gate, headline, and
both CPU readbacks.

## Validation

The focused CPU command was:

```text
skyrl-train/.venv/bin/python -m pytest \
  skyrl-train/tests/cpu/models/test_grug_moe.py \
  skyrl-train/tests/cpu/models/test_grug_model_wrapper.py \
  /tmp/grug-closeout/test_gate_contract.py -q
```

The final run collected 38 tests and reported `38 passed in 18.96s`. The first
attempt with the root environment stopped during collection because it lacked
Ray; no test ran in that attempt. The final evidence diff also passed:

```text
uv run python infra/pre-commit.py --changed-files --fix
git diff --check
python -m py_compile <both archived readers>
ruff check <both archived readers>
bash -n <each archived shell launcher>
```

The final post-commit advisory review reported maintainability findings in the
disposable diagnostic harness, but no evidence or measurement error. Its log is
`/tmp/marin-style-lint/grug-eager-grouped-divergence-closeout-20260802/20260803T050416-0a116b9x`.
The harness is not part of PR #276. Pruning or refactoring executed source after
the run would weaken the source-to-artifact chain, so those findings remain
recorded as evidence-branch debt.

## Four High goal reviews

All four reviewers agreed with leaving #276 unchanged, leaving #7903 open, and
refusing to treat the 18.811970x observation as causal. Claude, Gemini, and Kimi
approved the bounded outcome. Codex required provenance and reporting changes
before publication. This evidence archive, strict readback, and the narrowed
wording in `RUN_LOG.md` and the issue draft address the material findings.

| Reviewer | SHA-256 | Outcome |
| --- | --- | --- |
| Claude | `72f9d67fec9809978b2391f7871f7ac961dd41113810fd5c7735eec9250fd40a` | Approve blocked outcome; narrow topology scope and explain route retention. |
| Codex | `4b704ebe5a75267c2be17d88e9728626b1ba11eebf9521d9e2f5d37ec9606603` | Needs changes; preserve launch chain, assert pins, recompute semantics, narrow attribution, label critical ranks, and state the future acceptance condition. |
| Gemini | `6a64ac099e64cddbfe706b553dd9edf84031b32d53aa0aef7e5b63d6e1378477` | Approve blocked outcome. |
| Kimi | `aa8fbcc52277ec1fc616c3ed7491a30183d68c974df1344bf45a78a22d4606ff` | Approve; preserve launchers, image binding, and validation evidence. |

The route-retention and critical-rank clarifications are reporting changes. The
strict reader is CPU-only and does not alter or rerun either accelerator arm.

## Publication

The final evidence branch was pushed through
`0aa40030b4700eb13a2e3bc4223eecb26d9740aa` before publication. The bounded
closeout was then published at
https://github.com/marin-community/marin/issues/7903#issuecomment-5162561004.
Independent GitHub readback confirmed the exact comment, open issue state, and
unchanged clean state of MarinSkyRL #276.
