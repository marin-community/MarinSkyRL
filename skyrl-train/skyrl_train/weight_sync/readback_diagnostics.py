"""Bounded, read-only diagnostics for an opt-in initial weight sync."""

import hashlib
import itertools
import json
import os
import socket
import time
from pathlib import Path

import torch
from skyrl_train.io.io import read_bytes, stat_object, write_bytes_atomic


HASH_CHUNK_BYTES = 1 << 20
RECEIPT_CHUNK_BYTES = 3072
# One receipt in this many is still re-read in full, so the cheap verify is
# audited against the expensive one it replaces rather than trusted outright.
FULL_READBACK_EVERY = 8
_RECEIPTS_WRITTEN = itertools.count()
ENVIRONMENT_KEYS = (
    "NCCL_DEBUG",
    "NCCL_DEBUG_SUBSYS",
    "NCCL_DEBUG_FILE",
    "NCCL_IB_HCA",
    "NCCL_SOCKET_IFNAME",
    "NCCL_NET",
    "NCCL_NET_GDR_LEVEL",
    "NCCL_IB_DISABLE",
    "NCCL_PROTO",
    "NCCL_ALGO",
    "NCCL_MIN_NCHANNELS",
    "NCCL_MAX_NCHANNELS",
    "NCCL_NTHREADS",
    "NCCL_P2P_NET_DISABLE",
    "NCCL_LAUNCH_MODE",
    "NCCL_COLLNET_ENABLE",
    "NCCL_NVLS_ENABLE",
    "VLLM_BATCH_INVARIANT",
    "VLLM_ALLREDUCE_USE_SYMM_MEM",
    "CUBLAS_WORKSPACE_CONFIG",
)


def environment_readback() -> dict:
    return {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "observed_monotonic": time.monotonic(),
        "values": {key: os.environ.get(key) for key in ENVIRONMENT_KEYS},
        "network_log": network_log_readback(),
    }


def network_log_readback() -> dict:
    """Retain a bounded local NCCL network tail, with truncation and absence explicit."""
    pattern = os.environ.get("NCCL_DEBUG_FILE")
    if not pattern:
        return {"available": False, "reason": "NCCL_DEBUG_FILE is unset"}
    path = Path(pattern.replace("%h", socket.gethostname()).replace("%p", str(os.getpid())))
    if not path.is_file():
        return {"available": False, "path": str(path), "reason": "native file not present"}
    size = path.stat().st_size
    offset = max(0, size - 262144)
    with path.open("rb") as stream:
        stream.seek(offset)
        raw = stream.read(262144)
    lines = [
        line
        for line in raw.decode("utf-8", errors="replace").splitlines()
        if any(marker in line for marker in ("NET/IB", "NET/Socket", "Using network", "Bootstrap"))
    ]
    return {
        "available": True,
        "path": str(path),
        "file_bytes": size,
        "read_offset": offset,
        "tail_sha256": hashlib.sha256(raw).hexdigest(),
        "tail_bytes": len(raw),
        "truncated": offset > 0 or len(lines) > 128 or any(len(line) > 1024 for line in lines),
        "network_lines": [line[:1024] for line in lines[-128:]],
    }


def _verify_written_object(uri: str, payload: bytes) -> dict:
    """Establish that the stored object is exactly the payload, preferring not to move it.

    A single-part S3 object's ETag is the MD5 of exactly the bytes stored, so comparing
    it against a locally computed MD5 proves what re-reading the object proves, for the
    cost of a HEAD. These receipts are megabytes -- 6.13 MB median at P32/I8 -- and at
    that size the re-read costs about three times the write and sits on the trainer's
    critical path after resume.

    The cheap path can only ever accelerate, never weaken, the claim. Anything that makes
    the ETag unusable -- a local or non-S3 path, a multipart or server-side-encrypted
    object whose ETag is not a payload digest, a missing ETag, a size disagreement, or a
    digest that simply does not match -- falls through to the full re-read, which remains
    the authority. A sampled fraction is re-read in full regardless, so the cheap verify
    is audited against the expensive one rather than trusted outright.
    """
    sampled = next(_RECEIPTS_WRITTEN) % FULL_READBACK_EVERY == 0
    verify = "full-readback-sampled" if sampled else "full-readback"
    if not sampled and uri.startswith("s3://"):
        digest = hashlib.md5(payload, usedforsecurity=False).hexdigest()
        try:
            detail = stat_object(uri)
        except Exception as error:  # metadata is an optimisation; never fail the write on it
            detail = {"error": f"{type(error).__name__}: {error}"}
        etag = str(detail.get("ETag", "")).strip('"')
        if etag and "-" not in etag and detail.get("size") == len(payload) and etag == digest:
            return {
                "verify": "etag-md5",
                "etag": etag,
                "md5": digest,
                "full_readback_every": FULL_READBACK_EVERY,
                "verify_scope": "stored-object digest equals written payload; bytes not transferred",
            }
        # A mismatch here is not yet a failure: the ETag may simply not be a payload
        # digest. Record why the cheap path was abandoned so a systematic cause is
        # visible in the receipts instead of silently costing a full re-read forever.
        verify = "full-readback-after-etag-unusable"
        etag_detail = {"etag": etag or None, "stored_bytes": detail.get("size"), "md5": digest}
    else:
        etag_detail = {}
    if read_bytes(uri) != payload:
        raise ValueError("Durable native readback differs from the original receipt")
    return {
        "verify": verify,
        "full_readback_every": FULL_READBACK_EVERY,
        "verify_scope": "stored object re-read in full and compared byte for byte",
        **etag_detail,
    }


def persist_readback(output_uri: str, stage: str, receipt: dict) -> dict:
    """Write and verify native evidence before advancing to the next phase."""
    payload = json.dumps(receipt, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    uri = f"{output_uri.rstrip('/')}/{stage}-{socket.gethostname()}-{os.getpid()}.json"
    write_bytes_atomic(uri, payload)
    verification = _verify_written_object(uri, payload)
    return {
        "uri": uri,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        **verification,
    }


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash exact contiguous storage bytes using at most one MiB of host transfer scratch.

    This diagnostic hash is not the K14 every-byte comparison proof. No tensor
    conversion, full-sized contiguous copy, or device scratch allocation occurs.
    """
    if not tensor.is_contiguous():
        raise ValueError("Diagnostic hashing requires contiguous tensor storage")
    raw = tensor.detach().reshape(-1).view(torch.uint8)
    digest = hashlib.sha256()
    for offset in range(0, raw.numel(), HASH_CHUNK_BYTES):
        chunk = raw[offset : offset + HASH_CHUNK_BYTES].cpu()
        digest.update(memoryview(chunk.numpy()))
    return digest.hexdigest()


def parameter_digests(model_chunks) -> dict:
    """Keep dense and expert-shard identities separate, preserving names and shapes."""
    digests = {kind: hashlib.sha256() for kind in ("dense", "expert")}
    counts = {kind: {"parameters": 0, "bytes": 0} for kind in digests}
    samples = []
    for chunk_index, chunk in enumerate(model_chunks):
        for name, parameter in chunk.named_parameters():
            kind = "dense" if getattr(parameter, "allreduce", True) else "expert"
            receipt = {
                "chunk": chunk_index,
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "sha256": tensor_sha256(parameter),
            }
            digests[kind].update(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode())
            counts[kind]["parameters"] += 1
            counts[kind]["bytes"] += parameter.numel() * parameter.element_size()
            if ".experts." in name and name.endswith(("linear_fc1.weight0", "linear_fc2.weight0")):
                receipt["stride"] = list(parameter.stride())
                receipt["contiguous"] = parameter.is_contiguous()
                if name.endswith("linear_fc1.weight0"):
                    if parameter.shape[0] % 2:
                        raise ValueError("Expert fc1 cannot be split into equal gate/up halves")
                    halves = parameter.chunk(2, dim=0)
                    receipt["half_sha256"] = [tensor_sha256(half) for half in halves]
                samples.append(receipt)
    return {
        "digests": {kind: {**counts[kind], "sha256": value.hexdigest()} for kind, value in digests.items()},
        "expert_samples": samples,
    }


def validate_replica_digests(rows: list[dict]) -> list[dict]:
    """Compare only ranks in actual Megatron replica groups, never across expert shards."""
    by_rank = {row["rank"]: row for row in rows}
    if len(by_rank) != len(rows):
        raise ValueError("Duplicate policy rank")
    comparisons = []
    for kind in ("dense", "expert"):
        groups = {tuple(row["replica_ranks"][kind]) for row in rows}
        for ranks in sorted(groups):
            if not ranks or len(set(ranks)) != len(ranks) or any(rank not in by_rank for rank in ranks):
                raise ValueError("Incomplete replica group coverage")
            group = [by_rank[rank] for rank in ranks]
            if any(tuple(row["replica_ranks"][kind]) != ranks for row in group):
                raise ValueError("Asymmetric replica group membership")
            first = group[0]["digests"][kind]
            if any(row["digests"][kind] != first for row in group):
                raise ValueError(f"{kind} parameter digest mismatch on replica ranks {ranks}")
            comparisons.append({"kind": kind, "ranks": list(ranks), **first})
    return comparisons


def receipt_chunks(receipt: dict) -> list[dict]:
    """Encode bounded ASCII chunks; JSON escaping preserves arbitrary Unicode exactly."""
    encoded = json.dumps(receipt, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    parts = [encoded[offset : offset + RECEIPT_CHUNK_BYTES] for offset in range(0, len(encoded), RECEIPT_CHUNK_BYTES)]
    return [
        {"sha256": digest, "part": index, "parts": len(parts), "bytes": len(encoded), "payload": part}
        for index, part in enumerate(parts)
    ]


def reassemble_receipt(chunks: list[dict]) -> dict:
    if not chunks:
        raise ValueError("Missing diagnostic receipt")
    first = chunks[0]
    if len(chunks) != first["parts"] or {chunk["part"] for chunk in chunks} != set(range(first["parts"])):
        raise ValueError("Incomplete diagnostic receipt")
    if any(any(chunk[key] != first[key] for key in ("sha256", "parts", "bytes")) for chunk in chunks):
        raise ValueError("Mixed diagnostic receipts")
    encoded = "".join(chunk["payload"] for chunk in sorted(chunks, key=lambda chunk: chunk["part"]))
    if len(encoded.encode()) != first["bytes"] or hashlib.sha256(encoded.encode()).hexdigest() != first["sha256"]:
        raise ValueError("Diagnostic receipt hash mismatch")
    return json.loads(encoded)
