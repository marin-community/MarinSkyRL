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

Regular GPU CI tests live in `skyrl-train/tests/gpu/gpu_ci/`. Expensive, destructive, multi-node, and
fault-injection tests live outside that directory and require an explicit file path. A Python file deliberately
named without the `test_` prefix is opt-in and must remain outside default discovery.

The shared policy prohibits sleeps as readiness checks. An opt-in distributed test may deliberately delay or
withhold a rank when that condition is the test input. A test that expects a collective to hang or a worker to
remain withheld must use isolated worker processes, separate setup and execution deadlines, captured output,
and bounded cleanup. No test may leave a process, process group, Ray actor, or cluster job running.

Do not treat a compact collective smoke test as evidence for a production topology it does not exercise. Record
the GPU type, world size, EP/FSDP dimensions, dependency image or lock revision, command, branch commit, and
complete pass/fail result for on-demand distributed runs.

## Before a PR

Run `uv run infra/pre-commit.py --changed-files --fix`, commit the clean diff, then run
`uv run infra/pre-commit.py --review` and address every finding.
