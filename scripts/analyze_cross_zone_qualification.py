"""Summarize checked paired transport runs and two-lane cadence replays."""

import argparse
import json
import statistics
from pathlib import Path


def summary(values: list[float]) -> dict:
    return {
        "count": len(values),
        "median_seconds": statistics.median(values),
        "minimum_seconds": min(values),
        "maximum_seconds": max(values),
    }


def paired(path: Path) -> dict:
    sender = json.loads((path / "sender-result.json").read_text())
    receiver = json.loads((path / "receiver-result.json").read_text())
    received = {row["sample"]: row for row in receiver["samples"]}
    by_repeat: dict[int, dict] = {}
    for row in sender["samples"]:
        matching = received[row["sample"]]
        assert row["sha256"] == row["receiver"]["sha256"] == matching["sha256"]
        block = by_repeat.setdefault(row["repeat"], {})
        assert row["method"] not in block
        block[row["method"]] = row
    assert len(received) == len(sender["samples"])
    assert all(set(block) == {"tcp", "object"} for block in by_repeat.values())
    assert all(block["tcp"]["sha256"] == block["object"]["sha256"] for block in by_repeat.values())
    pairs = [by_repeat[index] for index in sorted(by_repeat)]
    return {
        "sender": sender["facts"]["hostname"],
        "receiver": receiver["facts"]["hostname"],
        "bytes_per_version": pairs[0]["tcp"]["bytes"],
        "versions": len(pairs),
        "tcp": summary([block["tcp"]["delivery_seconds"] for block in pairs]),
        "object": summary([block["object"]["delivery_seconds"] for block in pairs]),
        "tcp_minus_object": summary(
            [block["tcp"]["delivery_seconds"] - block["object"]["delivery_seconds"] for block in pairs]
        ),
        "tcp_faster_count": sum(
            block["tcp"]["delivery_seconds"] < block["object"]["delivery_seconds"] for block in pairs
        ),
        "sender_retransmissions": sorted(
            {leg["tcp_info"]["total_retransmissions"] for block in pairs for leg in block["tcp"]["legs"]}
        ),
    }


def fanout(path: Path, method: str) -> dict:
    lanes = []
    for lane in (0, 1):
        sender = json.loads((path / f"sender-result-{method}-{lane}.json").read_text())
        receiver = json.loads((path / f"receiver-result-{method}-{lane}.json").read_text())
        received = {row["sample"]: row for row in receiver["samples"]}
        assert len(sender["samples"]) == len(receiver["samples"])
        for row in sender["samples"]:
            assert row["method"] == method
            assert row["sha256"] == row["receiver"]["sha256"] == received[row["sample"]]["sha256"]
        lanes.append(sender["samples"])
    assert len(lanes[0]) == len(lanes[1])
    versions = []
    for first, second in zip(*lanes, strict=True):
        assert first["repeat"] == second["repeat"]
        versions.append(
            {
                "version": first["repeat"],
                "slowest_ack_after_schedule_seconds": max(
                    first["ack_epoch"] - first["scheduled_epoch"],
                    second["ack_epoch"] - second["scheduled_epoch"],
                ),
                "maximum_start_queue_seconds": max(first["queue_seconds"], second["queue_seconds"]),
                "lane_delivery_seconds": [first["delivery_seconds"], second["delivery_seconds"]],
            }
        )
    return {"method": method, "lanes": 2, "versions": versions}


def one_to_two(path: Path, method: str) -> dict:
    sender = json.loads((path / f"sender-one-to-two-{method}.json").read_text())
    receivers = [json.loads((path / f"receiver-result-{method}-{lane}.json").read_text()) for lane in (0, 1)]
    received = [{row["sample"]: row for row in receiver["samples"]} for receiver in receivers]
    versions = []
    for row in sender["samples"]:
        assert row["method"] == method and len(row["receiver_acks"]) == 2
        for lane in (0, 1):
            assert row["sha256"] == row["receiver_acks"][lane]["sha256"]
            assert row["sha256"] == received[lane][row["sample"]]["sha256"]
        versions.append(
            {
                "version": row["repeat"],
                "slowest_ack_after_schedule_seconds": row["ack_epoch"] - row["scheduled_epoch"],
                "start_queue_seconds": row["queue_seconds"],
                "slowest_ack_after_start_seconds": row["slowest_ack_seconds"],
            }
        )
    assert all(len(receiver["samples"]) == len(versions) for receiver in receivers)
    return {"method": method, "receivers": 2, "bytes_per_version": sender["samples"][0]["bytes"], "versions": versions}


def nccl(path: Path, receivers: int) -> dict:
    sender_name = "sender-2-result.json" if receivers == 2 else "sender-result.json"
    sender = json.loads((path / sender_name).read_text())
    receiver_rows = [
        json.loads((path / (f"receiver-{rank}-result.json" if receivers == 2 else "receiver-result.json")).read_text())
        for rank in range(receivers)
    ]
    assert sender["success"] and all(row["success"] for row in receiver_rows)
    assert all(len(row["samples"]) == len(sender["samples"]) for row in receiver_rows)
    for index, sample in enumerate(sender["samples"]):
        assert all(sample["sha256"] == row["samples"][index]["sha256"] for row in receiver_rows)
    timings = [row["acknowledged_seconds"] for row in sender["samples"]]
    return {
        "sender": sender["hostname"],
        "receivers": [row["hostname"] for row in receiver_rows],
        "bytes_per_version": sender["bytes"],
        "chunks": len(sender.get("sizes", [sender["bytes"]])),
        "versions": len(timings),
        "first_seconds": timings[0],
        "warm": summary(timings[1:]) if len(timings) > 1 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    args = parser.parse_args()
    result = {"paired": {}, "fanout": {}, "one_to_two": {}, "nccl": {}}
    for path in sorted(args.raw.glob("paired-*/sender-result.json")):
        result["paired"][path.parent.name] = paired(path.parent)
    for method in ("tcp", "object"):
        path = args.raw / "fanout-two-lane"
        if path.exists():
            result["fanout"][method] = fanout(path, method)
        path = args.raw / "one-to-two"
        if path.exists():
            result["one_to_two"][method] = one_to_two(path, method)
    for name, receivers in (("nccl-socket", 1), ("nccl-chunks", 1), ("nccl-chunks-02", 1), ("nccl-fanout", 2)):
        path = args.raw / name
        if path.exists():
            result["nccl"][name] = nccl(path, receivers)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
