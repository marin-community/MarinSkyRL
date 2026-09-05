"""Summarize retained inference receipts and counters without time bucketing.

Usage: python benchmarks/inference/analyze.py RUN_DIRECTORY
Writes analysis.json, intervals.csv, and engine-intervals.csv beside events.jsonl.
Only full poll intervals inside a treatment contribute to interval rates. Refill
steady state starts ten seconds after submission and ends at the last submission,
before its finite input queue drains. Profiled arms are explicitly marked.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def distribution(values):
    values = list(values)
    return {
        "mean": statistics.mean(values) if values else None,
        **{f"p{q}": quantile(values, q / 100) for q in (50, 90, 99)},
        "max": max(values) if values else None,
    }


def histogram(snapshot, name):
    return next(h for h in snapshot["histograms"] if h["name"] == name)


def midpoint(reading, clock):
    return (reading[f"poll_started_{clock}"] + reading[f"poll_finished_{clock}"]) / 2


def counters(event):
    totals = Counter()
    for reading in event["readings"]:
        s = reading["snapshot"]
        totals.update({k: s["cumulative"][k] for k in ("generation_tokens", "prompt_tokens", "preemptions")})
        totals.update({f"finish_{k}": v for k, v in s["cumulative"]["finished_by_reason"].items()})
        h = histogram(s, "inter_token_latency_seconds")
        totals.update(itl_count=h["count"], itl_sum=h["total"])
    return totals


def write_csv(path, rows):
    with path.open("w") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def analyze(directory):
    events = [json.loads(line) for line in (directory / "events.jsonl").open()]
    snapshots = [e for e in events if e["event"] == "snapshot"]
    expected = {r["snapshot"]["engine_id"] for r in snapshots[0]["readings"]}
    audit = {
        "engines": len(expected),
        "snapshots": len(snapshots),
        "identity_errors": 0,
        "histogram_errors": 0,
        "counter_resets": 0,
        "collection_errors": sum(e["event"] == "collection_error" for e in events),
        "request_errors": sum(e["event"] == "request_error" for e in events),
    }
    for event in snapshots:
        readings = event["readings"]
        ids = [r["snapshot"]["engine_id"] for r in readings]
        audit["identity_errors"] += int(len(ids) != len(expected) or set(ids) != expected)
        for reading in readings:
            for h in reading["snapshot"]["histograms"]:
                bucket_counts = [b[1] for b in h["buckets"]]
                if bucket_counts != sorted(bucket_counts) or bucket_counts[-1] != h["count"]:
                    audit["histogram_errors"] += 1

    commands = {
        e["command"]["name"]: e["command"] for e in events if e["event"] == "command" and "name" in e["command"]
    }
    starts = {e["treatment"]: e for e in events if e["event"] == "treatment_start"}
    ends = {e["treatment"]: e for e in events if e["event"] == "treatment_end"}
    audit["incomplete_treatments"] = sorted(starts.keys() - ends.keys())
    audit["errors_by_treatment"] = dict(
        Counter(
            e.get("treatment") or "outside_treatment"
            for e in events
            if e["event"] in ("collection_error", "request_error")
        )
    )
    receipts = defaultdict(list)
    for event in events:
        if event["event"] == "completion":
            receipts[event["treatment"]].append(event)

    engine_rows, interval_rows = [], []
    periodic = [e for e in snapshots if e["boundary"] == "periodic"]
    for left, right in pairwise(periodic):
        name = right["treatment"]
        if name is None or name != left["treatment"] or name not in ends:
            continue
        by_id = {r["snapshot"]["engine_id"]: r for r in left["readings"]}
        rows = []
        for r in right["readings"]:
            engine = r["snapshot"]["engine_id"]
            if engine not in by_id:
                continue
            l = by_id[engine]
            a, b = l["snapshot"], r["snapshot"]
            elapsed = midpoint(r, "monotonic") - midpoint(l, "monotonic")
            delta = b["cumulative"]["generation_tokens"] - a["cumulative"]["generation_tokens"]
            ha, hb = histogram(a, "inter_token_latency_seconds"), histogram(b, "inter_token_latency_seconds")
            count, total = hb["count"] - ha["count"], hb["total"] - ha["total"]
            if elapsed <= 0 or min(delta, count, total) < 0:
                audit["counter_resets"] += 1
                continue
            completed = sum(b["cumulative"]["finished_by_reason"].values()) - sum(
                a["cumulative"]["finished_by_reason"].values()
            )
            rows.append(
                {
                    "treatment": name,
                    "engine_id": engine,
                    "engine_index": r["engine_index"],
                    "start": midpoint(l, "at"),
                    "end": midpoint(r, "at"),
                    "seconds": elapsed,
                    "tokens": delta,
                    "tokens_per_second": delta / elapsed,
                    "completions": completed,
                    "itl_count": count,
                    "itl_sum": total,
                    "itl_seconds": total / count if count else None,
                    "running": b["current"]["running_requests"],
                    "waiting": b["current"]["waiting_capacity"] + b["current"]["waiting_deferred"],
                    "kv_usage": b["current"]["kv_cache_usage"],
                    "preemptions": b["cumulative"]["preemptions"] - a["cumulative"]["preemptions"],
                }
            )
        if len(rows) != len(expected):
            continue
        engine_rows.extend(rows)
        count = sum(r["itl_count"] for r in rows)
        start, end = min(r["start"] for r in rows), max(r["end"] for r in rows)
        last_submit = max(r["submitted_at"] for r in receipts[name])
        steady = commands[name]["mode"] == "refill" and start >= starts[name]["at"] + 10 and end <= last_submit
        interval_rows.append(
            {
                "treatment": name,
                "start": start,
                "end": end,
                "seconds": statistics.mean(r["seconds"] for r in rows),
                "steady": steady,
                "profiled": bool(commands[name].get("profiles")),
                "tokens": sum(r["tokens"] for r in rows),
                "tokens_per_second": sum(r["tokens_per_second"] for r in rows),
                "completions": sum(r["completions"] for r in rows),
                "itl_count": count,
                "itl_sum": sum(r["itl_sum"] for r in rows),
                "itl_seconds": sum(r["itl_sum"] for r in rows) / count if count else None,
                "running": sum(r["running"] for r in rows),
                "waiting": sum(r["waiting"] for r in rows),
                "kv_peak": max(r["kv_usage"] for r in rows),
                "preemptions": sum(r["preemptions"] for r in rows),
            }
        )

    arms = {}
    for name, end in ends.items():
        reqs = receipts[name]
        before = next(e for e in snapshots if e["treatment"] == name and e["boundary"] == "before")
        after = next(e for e in snapshots if e["treatment"] == name and e["boundary"] == "after")
        ca, cb = counters(before), counters(after)
        delta = {k: cb[k] - ca[k] for k in ca}
        rows = [r for r in interval_rows if r["treatment"] == name]
        steady_rows = [r for r in rows if r["steady"]]
        steady_seconds = sum(r["seconds"] for r in steady_rows)
        generated = sum(r["output_tokens"] for r in reqs)
        outcomes = Counter(r["finish_reason"] for r in reqs)
        arms[name] = {
            "command": commands[name],
            "start": starts[name]["at"],
            "end": end["at"],
            "seconds": end["seconds"],
            "requests": len(reqs),
            "tokens": generated,
            "whole_tokens_per_second": generated / end["seconds"],
            "whole_completions_per_second": len(reqs) / end["seconds"],
            "output_tokens": distribution(r["output_tokens"] for r in reqs),
            "latency_seconds": distribution(r["latency_seconds"] for r in reqs),
            "ttft_seconds": distribution(
                r["attempts"][0]["first_token_at"] - r["submitted_at"]
                for r in reqs
                if r["attempts"] and r["attempts"][0]["first_token_at"] is not None
            ),
            "outcomes": outcomes,
            "length_fraction": outcomes["length"] / len(reqs),
            "counter_delta": delta,
            "counter_token_receipt_difference": delta["generation_tokens"] - generated,
            "counter_finish_receipt_difference": sum(v for k, v in delta.items() if k.startswith("finish_"))
            - len(reqs),
            "missing_request_timings": sum(not r["attempts"] for r in reqs),
            "unknown_request_engines": sum(a["engine_id"] not in expected for r in reqs for a in r["attempts"]),
            "itl_seconds": delta["itl_sum"] / delta["itl_count"] if delta["itl_count"] else None,
            "peak_running": max((r["running"] for r in rows), default=None),
            "peak_waiting": max((r["waiting"] for r in rows), default=None),
            "peak_kv_usage": max((r["kv_peak"] for r in rows), default=None),
            "steady_seconds": steady_seconds,
            "steady_tokens_per_second": sum(r["tokens"] for r in steady_rows) / steady_seconds
            if steady_seconds
            else None,
            "steady_completions_per_second": sum(r["completions"] for r in steady_rows) / steady_seconds
            if steady_seconds
            else None,
            "wave_seconds": [e["seconds"] for e in events if e["event"] == "wave_end" and e["treatment"] == name],
        }
    audit.update(
        raw_bytes=(directory / "events.jsonl").stat().st_size,
        poll_collection_seconds=distribution(e["collection_seconds"] for e in periodic),
        poll_publication_seconds=distribution(e["publication_seconds"] for e in periodic),
        poll_spacing_seconds=distribution(b["poll_started_at"] - a["poll_started_at"] for a, b in pairwise(periodic)),
    )
    result = {"audit": audit, "arms": arms}
    (directory / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    write_csv(directory / "intervals.csv", interval_rows)
    write_csv(directory / "engine-intervals.csv", engine_rows)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    result = analyze(parser.parse_args().directory)
    print(json.dumps(result, indent=2))
