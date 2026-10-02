# MarinSkyRL testing

Read [the shared Marin testing policy](.agents/marin-style/TESTING-core.md) before writing or reviewing tests.
This file defines MarinSkyRL's package commands and accelerator-test boundaries.

## Package suites

Use the commands in [`AGENTS.md`](AGENTS.md#install-and-test). The root `marinskyrl` project owns launcher and
trainer installs; `skyrl-gym` and `skyrl-tx` retain their independent test environments. The workflow files
under `.github/workflows/` are authoritative for exact CI commands.

## GPU suites

GPU tests are not part of the ordinary CPU PR gate. Read the nearest module documentation before running them
and use an otherwise idle allocation with the required topology.

When the purpose is to validate a legacy built GPU image, run the image's installed Python and pytest directly.
Do not use `uv run --isolated`: it resolves a fresh environment, requires access to every direct URL in the lock,
and may select a different PyTorch/CUDA build from the image under test. Standard Iris tasks instead install the
frozen root profile before running GPU tests. Isolated `uv` runs remain useful on networked development hosts
when dependency resolution itself is part of the test.

The manually dispatched GPU CI suite lives in `skyrl-train/tests/gpu/gpu_ci/`. Expensive, destructive, multi-node, and
fault-injection tests live outside that directory and require an explicit file path. A Python file deliberately
named without the `test_` prefix is opt-in and must remain outside default discovery.

An invocation that names an Apptainer SIF is specific to Jupiter's Slurm runtime. Mark its test or batch file
with `Jupiter-only SIF test` so it cannot be mistaken for an Iris custom-image requirement.

The shared policy prohibits sleeps as readiness checks. An opt-in distributed test may deliberately delay or
withhold a rank when that condition is the test input. A test that expects a collective to hang or a worker to
remain withheld must use isolated worker processes, separate setup and execution deadlines, captured output,
and bounded cleanup. No test may leave a process, process group, Ray actor, or cluster job running.

Do not treat a compact collective smoke test as evidence for a production topology it does not exercise. Record
the GPU type, world size, Megatron parallel dimensions, dependency image or lock revision, command, branch commit, and
complete pass/fail result for on-demand distributed runs.

The two-run debug artifact acceptance contract and its Jupiter command are documented in
[`docs/debug-modes.md`](docs/debug-modes.md#jupiter-acceptance-test).

## Scheduled RL gates

The 06:00 UTC CatCount CPU nightly checks reversed-signal controls on seeds 0
and 1 and async checkpoint resume. PR CI runs the positive CPU canary on seed 0.
Both download the calibrated S3 policy and use cached pretraining when the
object store is unavailable.

At 09:00 UTC, Marin Nightly E2E runs GSM8K learning, synchronous OPD, Grug
Megatron training and the asynchronous CatCount H100 canary. Manual dispatch
can select one lane; other lanes have no job in that run. CatCount uses Marin
main with its external runtime pinned to the MarinSkyRL commit under test.
OpenCode and the GPU CI suite run only on manual dispatch. See
[`skyrl-train/ci/marin_nightly/README.md`](skyrl-train/ci/marin_nightly/README.md)
for gate thresholds and reproduction commands.

## Before a PR

Run `uv run infra/pre-commit.py --changed-files --fix`, commit the clean diff, then run
`uv run infra/pre-commit.py --review` and address every finding.
