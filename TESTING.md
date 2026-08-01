# MarinSkyRL testing

Read [the shared Marin testing policy](.agents/marin-style/TESTING-core.md) before writing or reviewing tests.
This file defines MarinSkyRL's package commands and accelerator-test boundaries.

## Package suites

MarinSkyRL has four independent Python packages and no root `uv` workspace. Run each package's tests from its
own directory:

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

The workflow files under `.github/workflows/` are authoritative for exact CI commands. Do not add a root
workspace or run one package through another package's environment.

## GPU suites

GPU tests are not part of the ordinary CPU PR gate. Read the nearest module documentation before running them
and use an otherwise idle allocation with the required topology.

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
