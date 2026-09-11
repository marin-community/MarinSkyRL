"""Run the three export-decomposition arms as fresh two-rank processes and audit the set."""

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from skyrl_train.weight_sync.export_decomposition_audit import (
    ARMS,
    CONNECTIONS,
    LAYERS,
    PASS_LINE,
    fold_costs,
    validate_receipt_set,
)

ARM_TIMEOUT_SECONDS = 420


def run_fresh_pair(command, environment):
    with subprocess.Popen(command, env=environment, start_new_session=True) as process:
        try:
            return process.wait(timeout=ARM_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return 124


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from skyrl_train.entrypoints.probe_export_decomposition import write_snowball_width_checkpoint

    # One random Snowball-width checkpoint shared by every arm; written before any CUDA context.
    entries = write_snowball_width_checkpoint(args.checkpoint, LAYERS)
    print(
        "K11_EXPORT_DECOMPOSITION_CHECKPOINT "
        + json.dumps({"path": str(args.checkpoint), "entries": len(entries), "layers": LAYERS}),
        flush=True,
    )
    results = []
    cells = []
    for arm in ARMS:
        output = args.output / f"{arm}.json"
        environment = dict(os.environ, CUDA_DEVICE_MAX_CONNECTIONS=str(CONNECTIONS))
        exit_code = run_fresh_pair(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=2",
                "-m",
                "skyrl_train.entrypoints.probe_export_decomposition",
                "--source-commit",
                args.source_commit,
                "--arm",
                arm,
                "--checkpoint",
                str(args.checkpoint),
                "--output",
                str(output),
            ],
            environment,
        )
        results.append({"arm": arm, "exit_code": exit_code})
        if output.exists():
            cells.append(json.loads(output.read_text()))
    # Retain every observed arm before judging the complete set.
    print(
        "K11_EXPORT_DECOMPOSITION_MATRIX_RECEIPT "
        + json.dumps({"source_commit": args.source_commit, "results": results, "cells": cells}),
        flush=True,
    )
    assert all(result["exit_code"] == 0 for result in results)
    validate_receipt_set(cells, args.source_commit)
    summaries = fold_costs(cells)
    print("K11_EXPORT_DECOMPOSITION_OBSERVATIONS " + json.dumps(summaries), flush=True)
    print(PASS_LINE, flush=True)


if __name__ == "__main__":
    main()
