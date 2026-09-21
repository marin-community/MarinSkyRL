"""Recompute matched same-region S3 relay results from retained raw JSON."""

import argparse
import json
from pathlib import Path
import statistics


ARMS = ("matched-dense", "matched-index", "matched-xor", "repeat-xor", "repeat-index")
EXPECTED_MODES = {
    "matched-dense": "dense",
    "matched-index": "gpu_index",
    "matched-xor": "cpu_xor",
    "repeat-xor": "cpu_xor",
    "repeat-index": "gpu_index",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def analyze(root: Path) -> dict:
    result = {}
    reference_hashes = None
    for arm in ARMS:
        sender = load(root / f"relay-{arm}-sender.json")
        receiver = load(root / f"relay-{arm}-receiver.json")
        for role, payload in (("sender", sender), ("receiver", receiver)):
            if payload["role"] != role or payload["mode"] != EXPECTED_MODES[arm]:
                raise ValueError(f"Wrong role or mode in {arm} {role}")
            if (payload["values"], payload["density"], payload["pattern"]) != (67_108_864, 0.019, "random_xor"):
                raise ValueError(f"Wrong input recipe in {arm} {role}")
            if len(payload["rows"]) != 6 or [row["version"] for row in payload["rows"]] != list(range(1, 7)):
                raise ValueError(f"Expected six consecutive updates in {arm} {role}")
        hashes = [row["target_sha256"] for row in sender["rows"]]
        if hashes != [row["target_sha256"] for row in receiver["rows"]]:
            raise ValueError(f"Sender and receiver disagree on target bytes in {arm}")
        if not all(row["exact"] for row in receiver["rows"]):
            raise ValueError(f"Receiver did not confirm exact bytes in {arm}")
        if reference_hashes is None:
            reference_hashes = hashes
        elif hashes != reference_hashes:
            raise ValueError(f"Input target bytes differ from other arms in {arm}")

        sends = sender["rows"][1:]
        receives = receiver["rows"][1:]

        def median(rows: list[dict], key: str) -> float:
            return statistics.median(row[key] for row in rows)

        publication = [
            sum(
                row[key]
                for key in (
                    "d2h_seconds",
                    "detect_seconds",
                    "pack_seconds",
                    "compress_seconds",
                    "put_seconds",
                    "ack_wait_seconds",
                    "commit_seconds",
                )
            )
            for row in sends
        ]
        result[arm] = {
            "postwarmup_updates": 5,
            "encoded_bytes_median": median(sends, "encoded_bytes"),
            "publication_seconds_median": statistics.median(publication),
            "publication_seconds_samples": publication,
            "put_seconds_median": median(sends, "put_seconds"),
            "wait_get_seconds_median": median(receives, "wait_get_seconds"),
            "get_seconds_median": median(receives, "get_seconds") if "get_seconds" in receives[0] else None,
            "receiver_apply_seconds_median": median(receives, "apply_seconds"),
            "source_local_seconds_median": statistics.median(
                sum(row[key] for key in ("d2h_seconds", "detect_seconds", "pack_seconds", "compress_seconds"))
                for row in sends
            ),
        }
    return {"target_sha256_by_update": reference_hashes, "arms": result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.evidence), indent=2))


if __name__ == "__main__":
    main()
