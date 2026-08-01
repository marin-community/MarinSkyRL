# Testing guidelines

MarinSkyRL has four independent Python packages and no root `uv` workspace. Run each package's tests from its
own directory, and use the root only for infrastructure and launcher tests.

## Safe local suites

Run the CPU suites that match the changed code:

```bash
# Trainer
(cd skyrl-train && uv sync --frozen --extra dev && uv run --frozen pytest tests/cpu/)

# Environments
(cd skyrl-gym && uv sync --frozen --extra dev && uv run --frozen pytest tests/)

# JAX engine
(cd skyrl-tx && uv run --extra tinker --extra dev pytest --forked -s tests)

# Root infrastructure and Iris launcher
uv run --no-project --with pytest --with pandas --with matplotlib --with tabulate pytest infra/tests/ -q
uv run --no-project \
  --with pytest --with pyyaml --with fsspec --with huggingface_hub --with universal-pathlib --with boto3 \
  pytest cloud/iris/tests/ -q
```

The workflow files under `.github/workflows/` are authoritative for the exact CI commands. Do not add a root
workspace or run one package through another package's environment.

## Core contract

A test must fail when user-visible or training-relevant behavior is wrong. Prefer assertions on:

- public API results and structured output;
- numerical values and gradients against an independent reference;
- checkpoint, dataset, or wire-format round trips;
- state transitions and real side effects in a temporary directory or in-memory fake;
- distributed values, process-group schedules, and bounded process termination.

Regression tests should reproduce the reported failure before the fix. Keep the setup realistic and small,
then run the narrow test followed by the relevant package suite.

## Tests to reject

Do not check in tests that only assert:

- a symbol exists, a constructor assigned a field, or a constant equals itself;
- a private helper was called a particular number of times;
- a log sentence, comment, or rendered command contains incidental wording;
- production logic agrees with a copy of the same logic in the test;
- code does not raise without observing a result or side effect.

Scratch probes are useful during development but do not belong in pytest. Replace them with a behavior
contract or delete them before review. Exact strings are appropriate only when the string is a public wire
format, machine-readable signal, or promised error message.

## Mocks and fakes

Run real in-process behavior by default. Use mocks at external I/O boundaries such as HTTP, subprocess,
object-store, W&B, Iris, or Kubernetes clients. Prefer an existing in-memory fake, temporary directory, or fake
clock over a mock. Do not mock an internal helper to prove that another internal helper dispatched to it.

## Timing and distributed tests

Do not use `time.sleep()` as a readiness check or assert that work finished within a narrow wall-clock window.
Use explicit readiness signals, bounded subprocess waits, fake clocks, or existing polling helpers.

Fault-injection tests may deliberately delay or withhold a rank when that condition is the behavior under test.
Tests that expect a collective to hang or a worker to remain withheld must use isolated worker processes,
separate setup and execution deadlines, bounded cleanup, and captured worker output on failure. No distributed
test may leave a process, process group, Ray actor, or cluster job running after its controller exits.

## GPU suites

GPU tests are not part of the ordinary CPU PR gate. Read the module documentation before running them and use
an otherwise idle allocation with the required topology.

The regular GPU suite lives in `skyrl-train/tests/gpu/gpu_ci/`. Expensive, destructive, multi-node, and
fault-injection tests live outside that directory and require an explicit file path. A Python file deliberately
named without the `test_` prefix is opt-in and must not be renamed merely to increase default coverage.

Do not treat a compact collective smoke test as evidence for a production topology it does not exercise. Record
the GPU type, world size, EP/FSDP dimensions, dependency image or lock revision, command, branch commit, and
complete pass/fail result for on-demand distributed runs.

## Numerical tests

Use an independent PyTorch, JAX, or straightforward mathematical reference. Cover a small meaningful shape and
dtype grid, including gradients when training uses them. Report pointwise error where useful, not only
`allclose`. Do not weaken a tolerance without user approval.

## Pytest style

- Prefer top-level `test_*` functions and fixtures over test classes.
- Name the subject, scenario, and outcome in the test name.
- Parameterize meaningful behavior variation instead of copying tests.
- Extend an existing file when it already owns the behavior.
- Every test must contain an assertion or `pytest.raises`.
- Remove dead fixtures, unused mocks, permanent skips, and empty test files.

Before opening or updating a PR, run `uv run infra/pre-commit.py --changed-files --fix`, commit the clean diff,
then run `uv run infra/pre-commit.py --review` and address every finding.
