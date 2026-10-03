# NUPA RL training set

Single-turn RL prompts for the Number Cookbook NUPA tasks, graded with the same
representation-sensitive exact-match metric as the `NUPA5K-Loose` policy eval.

## Disjointness from the policy eval

Evalchemy's `NUPA5K-Loose` is the canonical NUPA policy eval: a fixed
5,000-record stratified panel over the pinned `HaotongYang/NUPA_text` test
source. This builder reconstructs that panel's identity manifest from the same
pinned source with the published selection algorithm (round-robin over the 2,391
task/digit strata; within a stratum, unique source texts ordered by SHA-256),
asserts the reconstruction matches Evalchemy's checked-in manifest digest, and
emits one training row for every remaining unique source record. The training
pool is therefore exactly the complement of the eval panel: disjoint by
construction, and any drift between the pinned source revision and the eval
manifest fails the build instead of silently contaminating the eval.

Records are deduplicated by `(task_name, digit, sha256(source_text))` — the same
identity the eval manifest uses — and emitted in sorted identity order, so
rebuilds are byte-stable for a fixed source revision.

## Grading

Reward is the eval's primary metric (`exact_match`): permissive NUPA-Loose
answer extraction, then representation-sensitive component comparison of the
extracted answer against the reference. Rounding ("9.90" vs "9.9"), extra
digits, and unreduced fractions score zero, matching the eval. The `nupa`
skyrl-gym environment implements the reward; the extraction and digit-component
preparation mirror Evalchemy's scorer and correctness is graded through
`verifyit.adapters.evalchemy_nupa.grade_nupa_answer`.

## Build

```bash
cd skyrl-train
uv run --project .. examples/nupa/nupa_dataset.py --output_dir ~/data/nupa_rl
```

The builder downloads the 688 MB pinned source from the Hugging Face Hub
(pass `--source-file` to reuse a local copy), validates one known-good and
known-bad response per task against the runtime verifier, and writes
`train.parquet`. Row schema:

| column | value |
| --- | --- |
| `data_source` | `HaotongYang/NUPA_text` |
| `prompt` | one user message with the eval-identical prompt (`"<problem>  ="`) |
| `env_class` | `nupa` |
| `reward_spec` | `{"method": "rule", "ground_truth": "{\"answer\": ..., \"answer_format\": ...}"}` |
| `extra_info` | `task_name`, `operation`, `answer_format`, `digit`, `length_bucket`, `source_sha256` |

There is no validation split: the held-out measurement for this data is the
`NUPA5K-Loose` eval itself, which must never be trained on. Report eval results
under the `NUPA5K-Loose` name and keep them separate from RL training metrics.

## Train

Point a config at the built parquet, e.g. modeled on `examples/gsm8k` with
`train_batch_files: [{data_path: ~/data/nupa_rl/train.parquet}]` and the
`nupa` env class. The environment needs no environment-specific config.
