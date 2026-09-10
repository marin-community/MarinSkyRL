"""Opt-in two-host K10 group fixture, with no model, learner or dense weights."""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import ray

from skyrl_train.io.io import find_files, read_bytes, write_bytes_atomic
from skyrl_train.weight_sync.shard_group_probe import run_group_probe, tiny_schedule


def require_unmeasured(prefix: str) -> None:
    files = find_files(prefix.rstrip("/") + "/attempts")
    if len(files) > 1000:
        raise ValueError("Unexpected diagnostic namespace size")
    if any(name.endswith("/measurement-started.json") for name in files):
        raise ValueError("K10 diagnostic already contains a measured attempt")


def persist(path: str, row: dict) -> dict:
    payload = json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    if len(payload) > 1024**2:
        raise ValueError("Diagnostic receipt exceeds one MiB")
    write_bytes_atomic(path, payload)
    if read_bytes(path) != payload:
        raise ValueError("Diagnostic durable byte readback differs")
    return {"uri": path, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--durable-prefix", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    uid = os.environ.get("IRIS_ATTEMPT_UID", "")
    if not re.fullmatch("[0-9a-f]{16}", uid) or not re.fullmatch("[0-9a-f]{40}", args.source_commit):
        raise ValueError("Native attempt and immutable source identities are required")
    require_unmeasured(args.durable_prefix)
    prefix = args.durable_prefix.rstrip("/") + f"/attempts/{uid}"
    identity = {
        "attempt_uid": uid,
        "task_id": os.environ.get("IRIS_TASK_ID"),
        "source_commit": args.source_commit,
        "class": "X",
        "scope": "tiny expert-block collective; dense/shared model path and exclusive NIC attribution unqualified",
    }
    persist(prefix + "/entered.json", identity)
    marked = False

    def start(state):
        nonlocal marked
        if marked:
            return
        require_unmeasured(args.durable_prefix)
        persist(prefix + "/measurement-started.json", {**identity, "ready": state["ready"], "groups": state["groups"]})
        marked = True

    ray.init(address="auto")
    receipts = []
    try:
        for receivers in (2, 4):
            result = run_group_probe(
                tiny_schedule(receivers, 32 * 1024**2),
                "nccl",
                str(args.output / f"receivers-{receivers}"),
                start,
                two_hosts=True,
            )
            result.update(identity)
            receipts.append(persist(prefix + f"/receivers-{receivers}.json", result))
            if result["error"] is not None or "cleanup_error" in result:
                raise RuntimeError(result.get("error") or result["cleanup_error"])
        persist(prefix + "/complete.json", {**identity, "receipts": receipts})
        print("K10_TINY_NATIVE_GROUP_PASS senders=2 receivers=2,4 hosts=2", flush=True)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
