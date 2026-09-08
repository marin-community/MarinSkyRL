"""Byte-bounded transport for exact native request-state receipts."""

import hashlib
import json


RECEIPT_CHUNK_BYTES = 3072
MAX_RECEIPT_BYTES = 1 << 20


def publication_receipt_fields(state: dict) -> list[dict[str, str | int]]:
    """Encode one state without exceeding telemetry's 4096-byte string limit.

    ASCII JSON makes character and byte boundaries identical. The digest covers
    the whole receipt, so an auditor can reject missing or conflicting parts.
    Queue/export acknowledgement remains the telemetry transport's responsibility;
    consumers must check complete reassembly and the terminal loss counter.
    """
    receipt = json.dumps(state, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True)
    encoded = receipt.encode("ascii")
    if len(encoded) > MAX_RECEIPT_BYTES:
        raise ValueError("weight-sync request receipt exceeds the 1 MiB transport bound")
    digest = hashlib.sha256(encoded).hexdigest()
    chunks = [receipt[offset : offset + RECEIPT_CHUNK_BYTES] for offset in range(0, len(receipt), RECEIPT_CHUNK_BYTES)]
    return [
        {
            "receipt_json": chunk,
            "receipt_sha256": digest,
            "part_index": index,
            "part_count": len(chunks),
            "receipt_bytes": len(encoded),
        }
        for index, chunk in enumerate(chunks)
    ]
