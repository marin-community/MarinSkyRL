@.agents/marin-style/AGENTS-core.md

# MarinSkyRL

A hard fork of [NovaSky-AI/SkyRL](https://github.com/NovaSky-AI/SkyRL) maintained for the
Marin project, where it powers agentic RL training (SkyRL + Harbor).

## Fork policy

This is a **hard snapshot**: we own this tree and no upstream sync or merge-back is
planned. The Marin coding standards in `.agents/marin-style/AGENTS-core.md` therefore
apply to the whole repository, not to a Marin-authored subset — reformat, refactor, and
delete dead code freely. There is no upstream diff to keep small.

The one exception is `skyrl-agent/`, which is a dormant snapshot that nothing builds or
tests. It is excluded from lint (see `[tool.marin-style]` in `pyproject.toml`); leave it
alone rather than churning it.

## Repo map

The root `marinskyrl` distribution owns the launcher and trainer dependency graph. It is deliberately a
single project, not a uv workspace. `skyrl-gym/` and `skyrl-tx/` remain independent packages with their own
lockfiles and virtualenvs.

| package | status | what it is |
| --- | --- | --- |
| repository root | **primary** | The `marinskyrl` launcher and trainer distribution. Its frozen lock owns the CPU launcher, vLLM/FSDP2, and Megatron closures. |
| `skyrl-train/` | bundled | Trainer source, examples, and CPU/GPU tests included in the root wheel. |
| `skyrl-gym/` | bundled + independent | Gymnasium-style RL environments included in the root wheel; its standalone package remains independently testable. |
| `skyrl-tx/` | active | A JAX/Flax inference + fine-tuning engine (`tx`), independent of the trainer. Has its own CI. |
| `skyrl-agent/` | dormant | An older agent-harness snapshot. Not built, not tested, not linted. |

## Install and test

Run launcher and trainer commands from the repository root. The base install is CPU-only; select an image-build
extra only when resolving a GPU training environment.

```bash
# Root launcher + skyrl-train CPU tests (what PR CI runs)
uv sync --frozen --group dev --extra cpu --extra telemetry
uv run --frozen pytest cloud/iris/tests/ skyrl-train/tests/cpu/

# FSDP2/vLLM image closure (GPU tests need an 8-GPU node; not run in PR CI)
uv sync --frozen --extra fsdp --extra vllm --group dev
uv run --frozen pytest -s skyrl-train/tests/gpu/gpu_ci -m "not (integrations or megatron)"

# Megatron image closure (select it together with the common training closure)
uv sync --frozen --extra vllm --extra megatron --group dev

# skyrl-gym
cd skyrl-gym
uv sync --frozen --extra dev
uv run --frozen pytest tests/

# skyrl-tx
cd skyrl-tx
uv run --extra tinker --extra dev pytest --forked -s tests
```

Training runs go through Hydra. Commands that need the examples package run with `skyrl-train/` as the working
directory while retaining the root project:

```bash
uv run --project .. examples/gsm8k/gsm8k_dataset.py --output_dir "$HOME/data/gsm8k"
NUM_GPUS=8 LOGGER=console bash examples/gsm8k/run_gsm8k.sh
```

`cpu` and `cuda` are mutually exclusive PyTorch wheel profiles because Python extras cannot replace a base
dependency. GPU-only component extras such as `vllm`, `megatron`, and `deepspeed` imply `cuda`, so callers name
the component rather than its hardware consequence. `fsdp` adds TorchTitan for the expert-parallel FSDP path.
Native vLLM, FlashAttention, TransformerEngine, Mamba, and CUDA artifacts remain Docker image construction
concerns; the extras describe their Python closure.

## Lint

`infra/pre-commit.py` is the single lint entry point, as described in
`.agents/marin-style/AGENTS-core.md`:

```bash
uv run infra/pre-commit.py --all-files --fix
```

It runs `ruff check` + `ruff format` at line length 120 over `skyrl-train/`,
`skyrl-gym/`, and `skyrl-tx/`. `.pre-commit-config.yaml` carries only the gitleaks
secret scan, which marin-style does not cover; `format.sh` runs both.

**Before opening or updating a PR**, run the lint-review pass and fix or answer every
finding it reports:

```bash
uv run infra/pre-commit.py --review
```

This is required, not optional (see `.agents/marin-style/AGENTS-core.md`). The first run
bootstraps `marin-style` from git and can take several minutes; let it finish.

## Agent skill policy

Skills under `.agents/skills/` are durable procedures, not operational state. They must not contain run
identifiers, mutable image or source revisions, current capacity, user-specific paths, campaign status,
credential values, or historical narrative. Resolve those facts at execution time from the selected
configuration, current launcher interface, Marin's Iris configuration, and inspected job state. Describe
required outcomes, evidence, decision gates, and safety constraints rather than preserving a past launch recipe.

## CI

- `cpu_ci.yaml` — lint + `skyrl-train`/`skyrl-gym` CPU tests on every PR.
- `cpu_skyrl_tx.yaml` — `skyrl-tx` tests and tiny real training steps on every PR.
- `marin-nightly.yaml` — the nightly end-to-end gate: a real single-H100 GRPO run on
  GSM8K submitted to Iris, scored against `skyrl-train/ci/marin_nightly/specs/`. See
  `skyrl-train/ci/marin_nightly/README.md`.
