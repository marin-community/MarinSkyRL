# GPU fault injection

## Checkpoint generation failure and retry

`checkpoint_failure_retry.py` is an opt-in Megatron test that writes real
checkpoint shards to S3. Run its two cases in separate Python processes, in
order, on one otherwise idle four-H100 node. Run from the repository root with
`skyrl-train/` on `PYTHONPATH`: Ray's uv runtime hook requires the root
`pyproject.toml` to be inside the working directory. Set `CHECKPOINT_TEST_ROOT` to a
fresh, unique east-region `s3://marin-us-east-02a/tmp/ttl=14d/skyrl/users/atqamar/…`
prefix visible to both processes. Do not reuse a prior test root.
The test directory is a Python package so Ray workers can import the test's
worker subclass by its fully qualified module name.

```bash
export PYTHONPATH="$PWD/skyrl-train${PYTHONPATH:+:$PYTHONPATH}"
uv run --frozen --group dev --extra vllm --extra megatron \
  pytest -s skyrl-train/tests/gpu/fault_injection/checkpoint_failure_retry.py \
  -k failed_save_preserves_latest_and_retry_commits
uv run --frozen --group dev --extra vllm --extra megatron \
  pytest -s skyrl-train/tests/gpu/fault_injection/checkpoint_failure_retry.py \
  -k fresh_process_resumes_retry_and_saves_next_step
```

The first process saves step 1, injects an error after the step-2 distributed
save but before trainer publication, verifies that `latest` still selects step
1, and retries step 2 with a fresh attempt ID. The second process resumes the
committed retry through `latest`, performs an optimizer step, and saves step 3.
Both processes must exit zero; an expected injected exception is caught inside
the first test. The S3 root is not deleted by the test and should expire under
the bucket's TTL policy.

## Megatron checkpoint-to-next-step parity

`checkpoint_step_parity.py` is an opt-in, four-H100 Qwen3-0.6B TP2/PP2
checkpoint replay gate. It is **not** evidence of 32-rank Snowball numerical
parity. Run its two tests as separate pytest processes, in order, on one node
with a unique local scratch directory and a unique east-region TTL S3 prefix:

```bash
export CHECKPOINT_PARITY_LOCAL_ROOT=/local/scratch/<unique-case>
export CHECKPOINT_PARITY_S3_ROOT=s3://marin-us-east-02a/tmp/ttl=14d/skyrl/users/atqamar/<unique-case>
export PYTHONPATH="$PWD/skyrl-train${PYTHONPATH:+:$PYTHONPATH}"
uv run --frozen --group dev --extra vllm --extra megatron \
  pytest -s skyrl-train/tests/gpu/fault_injection/checkpoint_step_parity.py \
  -k reference_records_uninterrupted_step
uv run --frozen --group dev --extra vllm --extra megatron \
  pytest -s skyrl-train/tests/gpu/fault_injection/checkpoint_step_parity.py \
  -k fresh_actors_match_next_step
```

Run from the repository root in the pinned GPU runtime; the Iris job should have at
least four H100s and enough local scratch for two copies of all four ranks'
model and optimizer state. The first process performs an optimizer step, saves
the full checkpoint through the real S3 writer, verifies save did not advance
worker RNG, records the state at the checkpoint boundary, then records an
uninterrupted next step. The second process creates fresh actors, restores the
committed checkpoint, replays the exact batch file (SHA256 checked), and
compares every rank's model parameters/buffers, sharded optimizer values,
scheduler, Python/NumPy/Torch/CUDA RNG, and MCore CUDA RNG tracker before and
after the next step. Comparisons are exact; any mismatch is a diagnostic
failure, not an invitation to relax tolerance without understanding it. The
saved local snapshots are trusted pickle inputs; do not reuse an untrusted
artifact directory. Inspect both pytest exit statuses and the manifest before
claiming success.

## Distributed debug artifact acceptance

The opt-in two-node test in `distributed_debug_artifact_contract.py` runs one
healthy gang and one rank non-arrival gang. It checks that both finish within
bounded deadlines and that the declared debug artifacts are present under the
chosen durable root.

Run it only on an otherwise idle two-node allocation in the policy runtime.
See `docs/debug-modes.md` for the command and acceptance criteria. The test is
outside ordinary pytest discovery and does not run in PR CI.
