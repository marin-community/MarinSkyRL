"""Run the synthetic expert matrix with local bytes and controller-log receipts."""

import argparse
import os
from pathlib import Path

from iris.client.client import IrisClient
import ray

from cloud.iris.local_probe_receipts import LocalProbeReceipts
from skyrl_train.weight_sync.shard_group_probe import run_group_probe, tiny_schedule


def run_matrix(recorder, output):
    """Keep each native attempt's complete two-case matrix independent."""
    recorder.require_unmeasured()
    receipts = []
    try:
        for receivers in (2, 4):
            result = run_group_probe(
                tiny_schedule(receivers, 32 * 1024**2),
                "nccl",
                str(output / f"receivers-{receivers}"),
                recorder.mark,
                two_hosts=True,
            )
            result.update(recorder.identity)
            if not result["groups"] or any(
                row.get("readiness", {}).get("phase") != "groups-ready" for row in result["groups"]
            ):
                result["error"] = result["error"] or "Native group warmup ACK is missing"
            receipts.append(recorder.persist(f"receivers-{receivers}", result))
            if result["error"] or result.get("cleanup_error") or result.get("rendezvous_cleanup_error"):
                raise RuntimeError(result["error"] or result.get("cleanup_error") or result["rendezvous_cleanup_error"])
        recorder.persist(
            "complete",
            {
                **recorder.identity,
                "receipts": receipts,
                "scope": "synthetic expert collectives; no full model replay or NIC attribution",
            },
        )
        recorder.event("complete")
        print("K10_TINY_NATIVE_GROUP_PASS senders=2 receivers=2,4 hosts=2 warmup=true reserved_store=true", flush=True)
    except BaseException as error:
        recorder.event("measurement-failed" if recorder.marked else "startup-failed", error=type(error).__name__)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    client = IrisClient.in_cluster(os.environ["IRIS_CONTROLLER_ADDRESS"], timeout_ms=5000)
    try:
        recorder = LocalProbeReceipts(
            client, os.environ["IRIS_TASK_ID"], os.environ["IRIS_ATTEMPT_UID"], args.source_commit, args.output
        )
        ray.init(address="auto")
        try:
            run_matrix(recorder, args.output)
        finally:
            ray.shutdown()
    finally:
        client.shutdown()


if __name__ == "__main__":
    main()
