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

Four independent packages, each with its own `pyproject.toml`, lockfile, and virtualenv.
There is deliberately **no root uv workspace** — do not add one.

| package | status | what it is |
| --- | --- | --- |
| `skyrl-train/` | **primary** | The trainer. Ray + FSDP2/DeepSpeed/Megatron policy training with vLLM/SGLang rollouts. Largest test suite (`tests/cpu/`, `tests/gpu/`). Nearly all work happens here. |
| `skyrl-gym/` | active | Gymnasium-style RL environments (gsm8k, aime, …) and their reward functions. A dependency of `skyrl-train`. |
| `skyrl-tx/` | active | A JAX/Flax inference + fine-tuning engine (`tx`), independent of the trainer. Has its own CI. |
| `skyrl-agent/` | dormant | An older agent-harness snapshot. Not built, not tested, not linted. |

## Install and test

Every package uses `uv` and is built and tested from **its own directory**.

```bash
# skyrl-train (CPU tests -- what PR CI runs)
cd skyrl-train
uv sync --frozen --extra dev
uv run --frozen pytest tests/cpu/

# skyrl-train (GPU tests -- needs an 8-GPU node; not run in PR CI)
uv run --isolated --extra dev --extra vllm --extra deepspeed \
    pytest -s tests/gpu/gpu_ci -m "not (sglang or integrations or megatron)"

# skyrl-gym
cd skyrl-gym
uv sync --frozen --extra dev
uv run --frozen pytest tests/

# skyrl-tx
cd skyrl-tx
uv run --extra tinker --extra dev pytest --forked -s tests
```

Training runs go through Hydra, from `skyrl-train/`:

```bash
uv run examples/gsm8k/gsm8k_dataset.py --output_dir "$HOME/data/gsm8k"
NUM_GPUS=8 LOGGER=console bash examples/gsm8k/run_gsm8k.sh
```

`vllm` is `skyrl-train`'s only inference extra. The `sglang`, `mcore`, and `flashrl` extras were
removed: each pinned a `torch` or `transformers` version that the base package's own requirements
exclude, so none of them could be installed and their presence alone made `uv lock` unsolvable.
Production runs none of them.

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
