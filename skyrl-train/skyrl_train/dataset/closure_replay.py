"""Write audited structure-completion tasks for the native rollout loader."""

import argparse
import hashlib
import json
from pathlib import Path

from rolloutengine.spec import TaskSessionSpec
from skyrl_gym.closure_dataset import closure_tasks

from skyrl_train.dataset.tasks import write_tasks


def prepare_closure_tasks(tokens: Path, session: TaskSessionSpec, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    failures = {}
    hashes = {}
    counts = {}
    for split in ("train", "validation"):
        conversion = closure_tasks(tokens, split, session)
        failures[split] = list(conversion.failures)
        counts[split] = len(conversion.tasks)
        if not conversion.failures:
            destination = output / f"{split}.parquet"
            write_tasks(destination, conversion.tasks)
            hashes[split] = hashlib.sha256(destination.read_bytes()).hexdigest()
    report = {
        "status": "failed" if any(failures.values()) else "converted",
        "counts": counts,
        "failures": failures,
        "hashes": hashes,
        "token_manifest_sha256": hashlib.sha256((tokens / "manifest.json").read_bytes()).hexdigest(),
        "session": session.model_copy(update={"task_session": "structure_closure", "max_turns": 1}).model_dump(
            mode="json"
        ),
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--session", type=Path, required=True, help="Explicit TaskSessionSpec JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = prepare_closure_tasks(
        args.tokens, TaskSessionSpec.model_validate_json(args.session.read_text()), args.output
    )
    print(json.dumps(report, indent=2))
    raise SystemExit(report["status"] == "failed")


if __name__ == "__main__":
    main()
