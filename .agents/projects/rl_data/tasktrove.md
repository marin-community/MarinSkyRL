# TaskTrove — dataset tracker

Comprehensive inventory of **`open-thoughts/TaskTrove`** (HF `repo_type="dataset"`), the OpenThoughts-Agent
task-dataset collection (complement to `open-thoughts/AgentTrove` traces). Verified 2026-07-24.

- **Version: v3.17** · **151 subdirs** · **~2,595,000 tasks** total.
- **Structure:** one subdir per source dataset, named `org__name/` (the source HF repo `org/name` with `/`→`__`), containing **`tasks.parquet`** (columns `path`: str, `task_binary`: gzip-tar bytes). So a subdir maps back to its source repo by replacing the first `__` with `/` (e.g. `laion__exp_rpt_stack-csharp-v5` → `laion/exp_rpt_stack-csharp-v5`).
- **Prior versions** resolvable at git tags `v1` / `v2` / (code-contests fix `v3.1`); v3.2 = swegym v2→v5 swap; **v3.5 (2026-07-19) = +3 verifier-equipped datasets** (superuser + tezos StackExchange computer-use w/ ported nemotron_gym LLM-judge; `exp_rpt_issue-verified` = laion pytest mirror); **v3.6 (2026-07-19) = +5 SFT-source verified datasets** (unix/overflow/codereview/glaive LLM-judge; tulu3-personas-math deterministic exact-match); **v3.7 (2026-07-24) = replacement of the broken 5,000-row Stack-Pytest source with `laion/exp_rpt_stack-pytest-large-v2` (2,552 dependency-complete, deterministic-verifier tasks; one snapshot; 40/40 independent oracle; 200-task public smoke 62.2% positive with no exceptions).** **v3.8 (2026-07-27) = replacement of `laion/tulu3-sft-personas-math-sandboxes-verified` with `laion/tulu3-sft-personas-math-sandboxes-verified-v2` (verifier format fix: `test.sh` now invokes pytest on `test_state.py`, emitting parseable output for the `pass_ratio` reward shaper — graded 0.0/0.5/1.0 instead of silent binary fallback; also fixes 961 tasks silently impossible due to `tr -d ' '` asymmetry).** **v3.9 (2026-07-28) = replacement of `DCAgent/exp_rpt_nemotron-junit` with `laion/exp_rpt_nemotron-junit-v2` (vacuous verifier fix: old `test.sh` returned reward 1 for empty /app due to pipe masking + JUnit 0-test success + EXIT trap default 1; new verifier is default-fail, requires pom.xml, verifies Tests run >0, added `test_state.py` for pass_ratio-parseable output; Docker-validated empty→0).** **v3.10–v3.12 (2026-07-29) = SFT→RL conversion of 27 paper gap-set sources (issue #7418): each had a broken/missing/gamed verifier; all now have default-fail verifiers with pass_ratio-parseable output (`test_state.py`). Stage 1 (19 DET): stack-cpp, manybugs, bugsinpy, nemotron-bash×3, csharp, rust, stack-selfdoc, codeforces, codeelo, nemo-prism-math, quixbugs, bugswarm, stack-rspec, crosscodeeval-py/ts, toolscale, all-puzzles. Stage 2 (3 JUDGE): wizardlm-orca, staqc, qasper. Stage 3 (5 MIXED): magicoder, code-feedback, codeactinstruct, bash-textbook, softwareheritage.** **v3.13 (2026-07-29) = replacement of `SankalpKJ/swesmith-oracle-filtered` with `laion/swesmith-oracle-filtered-v2` (verifier test-file mismatch fix: SweSmith task branches lack test files added by the fix PR → verifier now restores them from the default branch via `git fetch origin && git checkout origin/main -- <test_files>`; Docker two-sided validated buggy→0, gold→1 on 4 diverse tasks).** **v3.14 (2026-07-29) = replacement of `DCAgent/exp_rpt_crosscodeeval-java` with `laion/exp_rpt_crosscodeeval-java-v2` (was Python import of Java code → SyntaxError, reward always 0; now language-aware text-diff verifier: gold→1, empty→0, Python junk→0) AND `DCAgent/r2egym-patched-full-oracle` with `laion/r2egym-patched-full-oracle-v2` (Harbor doesn't mount `setup_files/` → relocated `test_info.json` + `r2e_tests/` into `tests/`; Docker-verified `test_state.py` loads expected tests).** **v3.15 (2026-07-29) = setup_files propagation fix: Harbor NEVER mounts `setup_files/` (dead code — `TaskPaths.setup_files_dir` has zero callers). Scanned all 150+ TaskTrove datasets; found 6 with `/setup_files/` refs. Fixed: agent-facing refs inlined as heredocs (agent creates files at runtime); verifier-facing refs moved to `/tests/` (uploaded before verifier); R2E-Gym test-staging removed from instruction (verifier handles it). Affected: nl2bash (1570), swe_rebench (3787), mix_baseline (333 of 3718), mix_h7 (333 of 3718), openswe (17504), r2egym-v2 (3328 re-fix).** **v3.16 (2026-07-29) = reverted v3.15 heredoc inlining in instruction.md back to original `/setup_files/` references (Harbor patch will mount setup_files/). Verifier-side `/tests/` changes kept. Both `setup_files/` and `tests/setup_files/` present in tar. R2E-Gym test-staging removal kept as-is.** **v3.17 (2026-07-30) = LLM-judge OPENAI_API_KEY propagation fix: 14 datasets with LLM-judge verifiers (litellm/OpenAI) lacked `[verifier.env] OPENAI_API_KEY` in task.toml → verifier AuthenticationError at runtime. Fixed: added `[verifier.env]` to all 14; freelancer's hardcoded `sk-proj-...` credential replaced with `os.environ.get("OPENAI_API_KEY")`. 7 standalone repos also updated.**
- **Usage:** `python -m scripts.datagen.extract_tasks_from_parquet --parquet open-thoughts/TaskTrove --output_dir $SCRATCH/tasks/tasktrove --on_exist overwrite`.

> **Caveats:** (1) `laion__nemotron-gym-agent-workplace-v2` does NOT follow the convention — its data is at `data/train-00000-of-00001.parquet` (297 rows, same schema), so `extract_tasks_from_parquet` may not pick it up like the other 115. (2) `task_binary` is stored as parquet `binary`; the brief calls it "gzip tar bytes".

---

## A. Coding / SWE agentic task sets (64)

The Harbor agentic coding/SWE task sets — the `exp_rpt_*` (repo-PR-test), `exp_rle_*` (reverse-loop-eng),
`exp_flat25_*` styles + the patched-validated SWE sets. Used for agentic RL/eval.

| Subdir | Rows | Notes |
|---|---|---|
| DCAgent2__nl2bash-tasks-cleaned-oracle | 1,570 | v3.16: setup_files propagation fix (instruction.md references `/setup_files/`; verifier-side moved to `/tests/setup_files/`). Standalone repo: `laion/nl2bash-tasks-cleaned-oracle-v2`. |
| DCAgent__code-contests-noblock | 8,728 | competitive coding |
| DCAgent__exp_rle_adversarial | 5,000 | |
| **laion__exp_rpt_crosscodeeval-java-v2** | **2,139** | **v3.14 (2026-07-29). Replaces `DCAgent__exp_rpt_crosscodeeval-java`. Language mismatch fix: old verifier imported `/app/solution.py` as Python — Java code caused `SyntaxError` (reward always 0). Now: instruction asks for `/app/solution.java`, verifier does normalized text-diff against gold Java completion. Docker-validated gold→1, empty→0, Python junk→0.** |
| DCAgent__exp_rpt_curriculum-easy | 514 | |
| DCAgent__exp_rpt_curriculum-hard | 506 | |
| DCAgent__exp_rpt_curriculum-medium | 512 | |
| DCAgent__exp_rpt_e2egit-large | 5,000 | |
| DCAgent__exp_rpt_e2egit-v2 | 500 | |
| DCAgent__exp_rpt_issue | 4,830 | |
| DCAgent__exp_rpt_multifile | 4,907 | |
| DCAgent__exp_rpt_nemotron-cpp | 5,000 | VACUOUS VERIFIER (test embeds reference impl, never #includes agent header → empty /app passes). Regenerated → laion/exp_rpt_nemotron-cpp-v2 (800 tasks, oracle-validated, agent-linked gtest, empty-solution→0). Use v2. |
| laion__exp_rpt_nemotron-junit-v2 | 5,000 | v3.9 (2026-07-28). Replaces `DCAgent__exp_rpt_nemotron-junit` (5,000). **Vacuous verifier fix:** the original `test.sh` returned reward 1.0 for an untouched workspace — three compounding bugs: (1) `javac ... \| tee` without `set -o pipefail` hid javac's failure, (2) JUnit `--scan-class-path` found 0 tests and exited 0, (3) EXIT trap checked `$?=0` → wrote reward 1. New verifier: default-fail (`echo 0 > reward.txt` upfront), requires `pom.xml`, runs `mvn test` without tee (captures real exit code), verifies `Tests run: N>0` in output, added `test_state.py` (asserts `reward.txt=="1"`) invoked via `python3 -m pytest` for pass_ratio-parseable output. Docker-validated: empty /app → reward 0 (confirmed old gave 1). 1 snapshot (`eclipse-temurin:17-jdk` + maven + python3-pytest). Same 5,000 TestSolution.java files unchanged. |
| DCAgent__exp_rpt_pr | 4,793 | |
| DCAgent__exp_rpt_pymethods2test-large | 5,000 | the a3/explore-tis RL family base set |
| DCAgent__exp_rpt_pymethods2test-v3 | 500 | |
| DCAgent__exp_rpt_stack-dockerfile-v2 | 497 | |
| DCAgent__exp_rpt_stack-jest-large | 5,000 | |
| DCAgent__exp_rpt_stack-jest-v2 | 500 | |
| laion__exp_rpt_stack-pytest-large-v2 | 2,552 | v3.7. Replaces the broken DCAgent source: task-level dependencies captured for both agent and verifier; one snapshot; hidden references used only for validation, never shipped. Independent deterministic oracle 40/40 reward=1.0; public 200-task smoke 125/200 positive, 0 exceptions. |
| DCAgent__exp_rpt_stack-pytest-v2 | 500 | |
| DCAgent__exp_rpt_unitsyn-python-large | 5,000 | |
| DCAgent__exp_rpt_unitsyn-python-v3 | 500 | |
| DCAgent__inferredbugs-sandboxes-verifier | 10,000 | |
| DCAgent__llm-verifier-freelancer | 10,000 | LLM-judge verifier |
| **laion__r2egym-patched-full-oracle-v2** | **3,328** | **v3.14→v3.16 (2026-07-29). Replaces `DCAgent__r2egym-patched-full-oracle`. Harbor mount fix: Harbor does not mount `setup_files/` → verifier's `test_state.py` could not find `/setup_files/test_info.json` (reward always 0). Fixed: relocated `test_info.json` + `r2e_tests/` from `setup_files/` into `tests/` (which IS mounted). v3.16 re-fix: test-staging removed from instruction.md (verifier handles it). Standalone repo: `laion/r2egym-patched-full-oracle-v2`. Docker-verified: `test_state.py` loads 29 expected tests.** |
| DCAgent__selfinstruct-naive-sandboxes-2-verified | 9,638 | |
| DCAgent__swe_rebench_patched_oracle | 3,787 | v3.16: setup_files propagation fix. Standalone repo: `laion/swe-rebench-patched-oracle-v2`. |
| DCAgent__swe_rebench_v2_patched_oracle | 18,341 | |
| laion__exp_flat25_pseudocode-v2 | 728 | |
| laion__exp_flat25_speed_bonus-v2 | 764 | |
| laion__exp_flat25_stackoverflow-v2 | 765 | |
| laion__exp_flat25_subtle_debug-v3 | 289 | |
| laion__exp_rle_detailed-v3 | 413 | |
| laion__exp_rle_error_report-v3 | 261 | |
| laion__exp_rle_github_issue-v3 | 264 | |
| laion__exp_rle_heavy_padding-v2 | 784 | |
| laion__exp_rle_minimal_instructions-v3 | 233 | |
| laion__exp_rpt_codenet-python-v2 | 10,000 | **BROKEN — zero-byte test I/O (all reward 0). FIXED → use `laion/exp_rpt_codenet-python-v3`.** |
| laion__exp_rpt_codenet-python-v3 | 10,000 | **v2 fix (2026-06-23).** v1/v2 parquet→tasks extraction dropped the test I/O (empty `tests/inputs`+`tests/outputs` → EOFError, reward always 0); problem_id was also dropped. Recovered authoritative I/O from real CodeNet (`windchimeran/codenet_python`, 2861 problems) via `text-embedding-3-small` NN match (9,756/10k at sim≥0.7); each matched task now ships CodeNet clean stdin/stdout + a reference-code oracle `solution/solve.sh` + restored `problem_id`. Daytona oracle 27/29 (93%), smoke 0%, 1 snapshot. Code: `data/codenet_python_v3/`. |
| laion__exp_rpt_crosscodeeval-csharp-v4 | 1,768 | |
| laion__exp_rpt_defects4j-v3-v4 | 216 | |
| laion__exp_rpt_exercism-python-v2 | 133 | |
| laion__exp_rpt_ghactions-v3 | 9,930 | |
| laion__exp_rpt_issue-verified | 4,830 | v3.5 (2026-07-19). `laion` mirror of `DCAgent/exp_rpt_issue`; keeps its original deterministic **pytest** verifier (`tests/test_issue.py`) unchanged — real tests run against the agent's in-place fix → reward 1/0. 1 snapshot (shared `python:3.10-slim`). |
| laion__exp_rpt_methods2test-large-v2 | 4,472 | |
| laion__exp_rpt_methods2test-large-v3 | 4,472 | |
| laion__exp_rpt_pr-v2 | 4,793 | |
| laion__exp_rpt_scaffold-v2 | 4,861 | |
| laion__exp_rpt_stack-bash-v3 | 9,384 | |
| laion__exp_rpt_stack-bash-withtests-gpt5mini-v2 | 8,922 | |
| laion__exp_rpt_stack-bash-withtests-v2 | 8,922 | |
| laion__exp_rpt_stack-csharp-v5 | 9,485 | |
| laion__exp_rpt_stack-dockerfile-gpt5mini-v3 | 4,137 | |
| laion__exp_rpt_stack-go-v4 | 2,313 | |
| laion__exp_rpt_stack-junit-v6 | 872 | |
| laion__exp_rpt_stack-php-large-v6 | 3,789 | |
| laion__exp_rpt_stack-php-v2-v6 | 438 | |
| laion__exp_rpt_stack-ruby-v2 | 8,627 | |
| laion__exp_rpt_stack-rust-v2 | 9,987 | |
| laion__exp_rpt_taco-v2 | 10,000 | |
| laion__freelancer-projects-sandboxes-ta-rl-gpt-5-mini-v2 | 9,999 | |
| laion__freelancer-projects-sandboxes-ta-rl-gpt-5-nano-v2 | 10,000 | |
| laion__openswe-tasks-patched-v5-oracle-success | 17,504 | v3.16: setup_files propagation fix (instruction.md references `/setup_files/setup.sh`; verifier-side moved to `/tests/setup_files/`). Standalone repo: `laion/openswe-tasks-patched-v5-oracle-success`. |
| **laion__swegym-tasks-patched-validated-v5** | **2,438** | **v3.2 (2026-06-14): replaced `laion__swegym-tasks-patched-validated-v2` (989 tasks)** |
| **laion__tmax15k-tasks-snap-reduced-v3** | **6,751** | **TMax-15K-Harbor VERIFIABLE SUBSET (2026-06-29). 1 snapshot (shared Dockerfile; was 12,839 stale per-task). ~100% oracle-gated real solve.sh; small-sample oracle 40/40=100%, full-dataset oracle ~99.7%. Use for verified RL/datagen.** |
| **laion__tmax15k-tasks-full-v3** | **12,926** | **TMax-15K-Harbor FULL UNFILTERED set (2026-06-29). 1 snapshot (was 12,839 stale; regen via `data/tmax15k/generate.py --from-hub-export`). Verifiable fraction = 5,943/12,926 = 46.0% real solve.sh (rest placeholder, intentionally unfiltered). Small-sample oracle 15/40=37.5%; full-dataset oracle running. Includes unverifiable tasks by design — NO oracle floor.** |

## B. Curriculum mixes (15)

Blended/weighted task mixes for curriculum + reward-shaping ablations (the `mix_h*` hypotheses).

| Subdir | Rows |
|---|---|
| DCAgent__mix_h2_language_proportional | 4,135 |
| DCAgent__mix_h4_binary_easy | 2,010 |
| DCAgent__mix_h6_test_quality_top25 | 2,747 |
| laion__mix_baseline_uniform-v2 | 3,718 | v3.16: setup_files propagation fix (333 affected tasks). Standalone repo: `laion/mix_baseline_uniform-v2`. |
| laion__mix_h1_struggle_zone-v2 | 3,116 |
| laion__mix_h2_language_balanced-v2 | 4,506 |
| laion__mix_h5_skill_diverse-v2 | 3,166 |
| laion__mix_h7_raw_volume_5k-v2 | 3,718 | v3.16: setup_files propagation fix (333 affected tasks). Standalone repo: `laion/mix_h7_raw_volume_5k-v2`. |
| laion__mix_h8_adversarial_tests-v2 | 2,873 |
| laion__mix_h8_original_tests-v2 | 2,862 |
| laion__mix_h10_reward_binary-v2 | 2,862 |
| laion__mix_h10_reward_proportional-v2 | 2,873 |
| laion__mix_h10_reward_staged-v2 | 3,873 |
| laion__mix_h11_compositional_gradient-v2 | 3,873 |
| laion__mix_h11_single_skill_only-v2 | 2,873 |

## C. SankalpKJ oracle-filtered (3)

Large oracle-filtered nemotron/swesmith sets.

| Subdir | Rows |
|---|---|
| SankalpKJ__nemotron-code-oracle-filtered | 15,165 |
| SankalpKJ__nemotron-math-oracle-filtered | 114,280 |
| **laion__swesmith-oracle-filtered-v2** | **12,942** | **v3.13 (2026-07-29). Replaces `SankalpKJ__swesmith-oracle-filtered`. Verifier test-file mismatch fix: SweSmith task branches are created at an older commit where test files added by the fix PR don't exist — the verifier's expected test paths (FAIL_TO_PASS / PASS_TO_PASS) referenced files absent from the checkout, scoring every trial 0.0. Fixed: verifier now `git fetch origin` + restores missing test files from the default branch before running pytest. Docker two-sided validated on 4 diverse tasks (bottlepy/oauthlib/tenacity/pdfminer): buggy→0, gold→1.** |

## D. Nemotron-Gym RLVR conversions (35)

The v3 additions — converted from `nvidia/Nemotron-Post-Training-v3` via the `data.nemotron_gym` framework
(instruction-following, math, science, knowledge, reasoning, multi-turn, safety, single-step agentic pivots).
Self-contained verifiers where a deterministic gold exists; LLM-judge where grading is subjective.

| Subdir | Rows |
|---|---|
| laion__nemotron-gym-agent-calendar | 3,358 |
| laion__nemotron-gym-agent-workplace-v2 | 297 ⚠ |
| laion__nemotron-gym-agentic-conversational-tool-use-pivot-v2 | 96,965 |
| laion__nemotron-gym-agentic-function-calling-pivot-v2 | 9,579 |
| laion__nemotron-gym-agentic-indirect-prompt-injection-v2 | 1,272 |
| laion__nemotron-gym-agentic-swe-pivot-v2 | 3,978 |
| laion__nemotron-gym-arc-agi-python-inductive | 10,000 |
| laion__nemotron-gym-arc-agi-transductive-v2 | 10,000 |
| laion__nemotron-gym-cfbench-v2 | 1,105 |
| laion__nemotron-gym-competitive-coding | 15,713 |
| laion__nemotron-gym-identity-following-v2 | 21,660 |
| laion__nemotron-gym-instruction-following-adversarial-v3 | 1,000 |
| laion__nemotron-gym-instruction-following-calendar | 8,387 |
| laion__nemotron-gym-instruction-following-citation | 9,033 |
| laion__nemotron-gym-instruction-following-freeform | 8,869 |
| laion__nemotron-gym-instruction-following-multiturnchat-v2 | 2,011 |
| laion__nemotron-gym-instruction-following-structured | 9,437 |
| laion__nemotron-gym-instruction-following-v2 | 46,391 |
| laion__nemotron-gym-inverse-ifeval-v2 | 1,000 |
| laion__nemotron-gym-knowledge-mcqa | 616,888 |
| laion__nemotron-gym-knowledge-openqa-v2 | 122,357 |
| laion__nemotron-gym-knowledge-web-search-mcqa | 2,915 |
| laion__nemotron-gym-litmus-bench | 5,232 |
| laion__nemotron-gym-math-advanced-calculations-v3 | 5,291 |
| laion__nemotron-gym-math-openmathreasoning | 112,867 |
| laion__nemotron-gym-math-stack-overflow | 436,307 |
| laion__nemotron-gym-math-v4 | 6,534 |
| laion__nemotron-gym-multichallenge-advanced-v2 | 1,068 |
| laion__nemotron-gym-multichallenge-vanilla-v2 | 1,050 |
| laion__nemotron-gym-qa-abstention-v2 | 3,150 |
| laion__nemotron-gym-reasoning-gym | 14,259 |
| laion__nemotron-gym-safety-v2 | 89,066 |
| laion__nemotron-gym-science-so-openq | 150,644 |
| laion__nemotron-gym-structured-outputs-v4 | 53,870 |
| laion__nemotron-gym-sysbench-v2 | 1,010 |

⚠ `agent-workplace-v2` stores its parquet at `data/train-00000-of-00001.parquet`, not `tasks.parquet`.

## E. StackExchange Q&A / computer-use (LLM-judge verified) (5)

StackExchange-sourced task sets repackaged with the `data.nemotron_gym` **LLM-judge** verifier
(`verifiers/llm_judge.py`) ported in to replace the original Skywork reward-model verifier. Each sandbox's
`tests/test_state.py` now reads `/app/response.txt` + `/tests/verifier_data.json` (task instruction + a
per-dataset adapted rubric), calls litellm/gpt-4o-mini, and writes a 0.0–1.0 reward via `\boxed{score}`.
**Needs `OPENAI_API_KEY` at trial time** (propagated via `task.toml` `[verifier].env`). 1 snapshot each
(shared `ubuntu:24.04`). Rows 1–2 = v3.5; rows 3–5 = v3.6.

| Subdir | Rows | Notes |
|---|---|---|
| laion__stackexchange-superuser-sandboxes-verified | 10,000 | v3.5. Super User (Linux/Windows desktop, hardware, sysadmin) computer-use. Rubric = correctness/completeness/safety/relevance. Source: `DCAgent/stackexchange-superuser-sandboxes-skywork-response`. Live dry-run: good→0.85, bad→0.0. |
| laion__stackexchange-tezos-sandboxes-verified | 10,000 | v3.5. Tezos (blockchain node/baker, crypto) computer-use. Rubric = correctness/completeness/relevance/soundness. Source: `DCAgent/stackexchange-tezos-sandboxes-skywork-response`. |
| laion__stackexchange-unix-sandboxes-verified | 10,000 | v3.6. Unix & Linux (shell, CLI, sysadmin) computer-use. Rubric = correctness/completeness/safety/relevance. Source: `DCAgent/stackexchange-unix-sandboxes-skywork-response`. Live dry-run: good→1.0, bad→0.0. |
| laion__stackexchange-overflow-sandboxes-verified | 10,000 | v3.6. Stack Overflow programming Q&A. Rubric = correctness/completeness/clarity/relevance. Source: `DCAgent/stackexchange-overflow-sandboxes-skywork-response`. |
| laion__stackexchange-codereview-sandboxes-verified | 10,000 | v3.6. Code Review StackExchange (review submitted code). Rubric = correctness/insight/actionability/relevance. Source: `DCAgent/stackexchange-codereview-sandboxes-skywork-response`. |

## F. SFT-source verified additions (v3.6) (2)

Non-StackExchange SFT sources reconciled into verifier-equipped TaskTrove sets (see
`experiments/active/sft-to-tasktrove/TRACKER.md`). 1 snapshot each (shared Dockerfile).

| Subdir | Rows | Notes |
|---|---|---|
| laion__glaive-code-assistant-sandboxes-verified | 10,000 | v3.6. Glaive code-assistant Q&A. Source `DCAgent/glaive-code-assistant-sandboxes` shipped NO verifier → added the nemotron_gym **LLM-judge** (rubric = correctness/completeness/clarity/relevance) + appended `response.txt` submission guidance to instruction.md; reference `solution/` DROPPED to prevent reward-hacking. Needs `OPENAI_API_KEY`. Live dry-run: good→0.9, bad→0.0. |
| laion__tulu3-sft-personas-math-sandboxes-verified-v2 | 9,998 | v3.8 (2026-07-27). Replaces `laion__tulu3-sft-personas-math-sandboxes-verified` (v3.6). **Verifier format fix:** the original `tests/test.sh` emitted `Correct answer: N` / `Incorrect answer: expected N, got M` — unparseable by the `pass_ratio` reward shaper, causing silent fallback to binary reward and RLOO collapse. Fixed: `test.sh` now invokes `python3 -m pytest /tests/test_state.py` (2 tests: file-exists + contents-match), emitting parseable pytest output with **graded signal** (no answer→0.0, wrong answer→0.5, correct→1.0). Also fixes a latent bug: the original `tr -d ' '` stripped spaces from the answer but not the expected, making 961/9998 tasks silently impossible. Docker-validated: correct→2 passed, wrong→1 failed 1 passed, no file→2 failed. 1 snapshot (`ubuntu:24.04` + pytest). No LLM judge, no API cost. |

## G. SFT→RL conversion — paper gap set (v3.10–v3.12) (27)

Converted from the paper's 95 SFT task-gen strategies (issue #7418's 43 gap sources with no usable verifier).
Each had a broken/missing/gamed verifier; all now have default-fail verifiers with pass_ratio-parseable
output (`test_state.py`). Shared infrastructure: `data/rl_converters/` (templates, oracle gate, upload).

### Stage 1 — Deterministic verifiers (19)

| Subdir | Rows | Rank | Notes |
|---|---|---|---|
| laion__exp_rpt_stack-cpp-v2 | 9,943 | 30 | v3.10. Framework-aware (gtest/doctest/boost/catch2 auto-detection + `-I/app`). Old: doctest/gtest mismatch → always-fail. |
| laion__exp_rpt_stack-selfdoc-gpt5mini-v2 | 6,547 | 16 | v3.10. Filtered 3453 heavy-dep tasks (torch/numpy/etc); fixed PYTHONPATH + default-fail. |
| laion__exp_rpt_stack-rspec-v2 | 10,000 | 41 | v3.11. Ruby rspec. Fixed missing python3 + load-path bootstrap. |
| laion__exp_rpt_nemotron-bash-v2 | 5,000 | 63 | v3.10. Fixed embedded-reference + exit-escapes-script. Default-fail + subprocess isolation. |
| laion__exp_rpt_nemotron-bash-withtests-v2 | 10,000 | 58 | v3.10. Same fix as nemotron-bash-v2. |
| laion__exp_rpt_nemotron-bash-withtests-gpt5mini-v2 | 10,000 | 51 | v3.10. Same fix. |
| laion__exp_rpt_nemotron-csharp-v2 | 4,108 | 61 | v3.10. Fixed xunit/MSTest/NUnit framework auto-detection + embedded-reference guard. |
| laion__exp_rpt_nemotron-rust-v2 | 170 | 73 | v3.10. Fixed embedded-ref + exit-escape. Default-fail + crate-name detection. |
| laion__exp_rpt_bugsinpy-v2 | 500 | 40 | v3.10. Fixed malformed test files + broken solution imports + PYTHONPATH. |
| laion__exp_rpt_manybugs-v2 | 164 | 22 | v3.10. Replaced gameable text-diff with exact gold-match. 7 project snapshots. |
| laion__exp_rpt_quixbugs-v2 | 40 | 86 | v3.11. Replaced no-op stub tests with real per-algorithm pytest suites. |
| laion__exp_rpt_bugswarm-v2 | 2 | 54 | v3.11. Fixed undefined deps + made tests self-contained. |
| laion__exp_rpt_crosscodeeval-python-v2 | 500 | 56 | v3.11. Line-completion text-diff verifier (reused CrossCodeEval csharp pattern). |
| laion__exp_rpt_crosscodeeval-typescript-v2 | 3,356 | 65 | v3.11. Same line-completion verifier for TypeScript. |
| laion__toolscale-v2 | 4,035 | 76 | v3.11. Deterministic JSON-schema function-call validation. Was LLM-judge → now no API cost. |
| laion__all-puzzles-v2 | 6,926 | 89 | v3.11. Per-type solver at build → gold.json → normalized compare. 3074 Zebra puzzles dropped (no ground truth). |
| laion__codeforces-v2 | 10,000 | 81 | v3.11. Joined to open-r1/codeforces for test cases; multi-language compile/run/compare + special judge. |
| laion__codeelo-v2 | 500 | 84 | v3.11. Unified compile/run/compare verifier (py/cpp/java). |
| laion__nemo-prism-math-v2 | 10,000 | 83 | v3.11. Normalized exact-match (strip LaTeX/whitespace → compare). Was placeholder (always 1). |

### Stage 2 — LLM-judge verifiers (3)

Needs `OPENAI_API_KEY` at trial time (gpt-4o-mini, temperature=0, `\boxed{score}` → reward).

| Subdir | Rows | Rank | Notes |
|---|---|---|---|
| laion__wizardlm-orca-v2 | 10,000 | 59 | v3.12. Rubric: correctness/completeness/relevance. |
| laion__staqc-v2 | 10,000 | 12 | v3.12. Rubric: correctness/completeness/clarity. |
| laion__qasper-v2 | 10,000 | 69 | v3.12. Rubric: correctness/completeness/groundedness. |

### Stage 3 — MIXED→judge/deterministic (5)

| Subdir | Rows | Rank | Notes |
|---|---|---|---|
| laion__magicoder-v2 | 4,096 | 75 | v3.12. LLM-judge (OSS-Instruct code, no shipped tests). |
| laion__code-feedback-v2 | 10,000 | 9 | v3.12. LLM-judge (multi-turn coding, no tests dir). |
| laion__codeactinstruct-v2 | 10,000 | 70 | v3.12. LLM-judge (CodeAct trajectories, no tests dir). |
| laion__bash-textbook-v2 | 9,078 | 48 | v3.12. LLM-judge (textbook bash snippets). |
| laion__exp_rpt_softwareheritage-v2 | 7 | 29 | v3.12. Deterministic (493/500 source tasks embed reference in test → unsalvageable; 7 fixed). |

---

## Maintaining TaskTrove (add / replace a dataset)

Pattern (see the scripts in `.agents/projects/tasktrove/`, e.g. `_tasktrove_v3_add.py`): stage
`<org__name>/tasks.parquet` in a staging dir, update the root `README.md` version note (dot-bump, e.g.
v3.2→v3.3), then `HfApi().upload_folder(repo_id="open-thoughts/TaskTrove", repo_type="dataset", ...)`.
- **Additive** (new dataset): no `delete_patterns`.
- **Replace** (e.g. the swegym v2→v5 swap): pass `delete_patterns=["<old_subdir>/**"]` scoped to ONLY the old subdir, in the same commit. Always re-list the repo afterward to confirm the new subdir is present, the old is gone, and the count is as expected.
- Secrets: `source secrets.env` for `HF_TOKEN` (env var only). penfever has `write` on `open-thoughts`.
- This is an outward-facing write to a shared org repo — verify before declaring done (supervisor discipline).
