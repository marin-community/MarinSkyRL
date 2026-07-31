"""Mechanical completion checks for a downloaded RL model repository."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TRAINING_TRACES_LINK = re.compile(
    r"^## Training Traces\s*$.*?https?://huggingface\.co/datasets/[^\s)]+", re.MULTILINE | re.DOTALL
)


def model_publication_status(repository: Path) -> dict[str, str]:
    """Return required RL publication artifacts as present or absent."""
    training_logs = repository / "training_logs"
    readme = repository / "README.md"
    has_weights = any(repository.glob("*.safetensors"))
    has_tokenizer_configuration = (repository / "config.json").is_file() and any(repository.glob("tokenizer*"))
    has_training_logs = training_logs.is_dir() and any(path.is_file() for path in training_logs.rglob("*"))
    has_trace_link = readme.is_file() and bool(TRAINING_TRACES_LINK.search(readme.read_text(encoding="utf-8")))
    return {
        "weights": "present" if has_weights else "absent",
        "tokenizer_configuration": "present" if has_tokenizer_configuration else "absent",
        "training_logs": "present" if has_training_logs else "absent",
        "training_traces_link": "present" if has_trace_link else "absent",
    }


def require_complete_model_publication(repository: Path) -> dict[str, str]:
    """Raise when a downloaded model repository misses required RL artifacts."""
    status = model_publication_status(repository)
    absent = [artifact for artifact, state in status.items() if state != "present"]
    if absent:
        raise ValueError(f"Incomplete RL model publication: missing {', '.join(absent)}")
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the required artifacts in a downloaded RL model repo.")
    parser.add_argument("repository", type=Path)
    args = parser.parse_args()
    status = require_complete_model_publication(args.repository)
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
