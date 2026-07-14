set -e

# Lint and format via the shared marin-style checks.
uv run infra/pre-commit.py --all-files --fix

# Secret scan (the only hook left in .pre-commit-config.yaml).
if command -v uv >/dev/null 2>&1; then
    uv pip install -q pre-commit
else
    pip install -q pre-commit
fi

# pre-commit run --all-files always runs from the root directory.
pre-commit run --all-files --config .pre-commit-config.yaml
