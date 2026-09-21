"""Retain every stdout-mirrored training metric from one Iris Ray worker log."""

import argparse
import hashlib
import json
from pathlib import Path
import re


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--arm", required=True)
parser.add_argument("--job", required=True)
parser.add_argument("--runtime-commit", required=True)
parser.add_argument("--trainer-log", type=Path, required=True)
parser.add_argument("--trainer-log-source", required=True)
parser.add_argument("--resolved-config", type=Path, required=True)
parser.add_argument("--resolved-config-source", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

rows = []
pattern = re.compile(r"WANDB_MIRROR kind=(startup|train) step=(\d+) metrics=")
decoder = json.JSONDecoder()
for line in args.trainer_log.read_text(errors="replace").splitlines():
    match = pattern.search(line)
    if match is None:
        continue
    metrics, _ = decoder.raw_decode(line[match.end() :])
    rows.append({"kind": match[1], "step": int(match[2]), "metrics": metrics})

assert sum(row["kind"] == "startup" for row in rows) == 1
train_steps = [row["step"] for row in rows if row["kind"] == "train"]
assert train_steps == list(range(1, len(train_steps) + 1)), train_steps
result = {
    "arm": args.arm,
    "job": args.job,
    "runtime_commit": args.runtime_commit,
    "trainer_log_source": args.trainer_log_source,
    "trainer_log_sha256": hashlib.sha256(args.trainer_log.read_bytes()).hexdigest(),
    "resolved_config_source": args.resolved_config_source,
    "resolved_config_sha256": hashlib.sha256(args.resolved_config.read_bytes()).hexdigest(),
    "rows": rows,
}
args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(f"{args.output}: startup=1 train={len(train_steps)}")
