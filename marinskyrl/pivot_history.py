"""Recover frozen profiling samples from committed rollout archives."""

from dataclasses import dataclass, field
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from itertools import batched
import gzip
import json
from io import BytesIO
from typing import Any
from zipfile import ZipFile

from marinskyrl.pivot import is_context_exclusion
from marinskyrl.remote_io import filesystem_and_path


def validate_profile_retention(retention: Mapping[str, Any]) -> None:
    """Require complete profiling archives before allocating inference engines."""
    if not (
        retention.get("enabled", False)
        and retention.get("required", False)
        and retention.get("sample_fraction") == 1.0
        and retention.get("max_bytes_per_run") is None
        and retention.get("max_bytes_per_step") is None
        and "eval" in retention.get("phases", ())
        and retention.get("output_path")
    ):
        raise ValueError("Profiling requires complete, unbounded, durable evaluation retention")


@dataclass
class ProfileHistory:
    samples: dict[tuple[str, int, int], dict[str, Any]] = field(default_factory=dict)
    discarded_record_ids: list[str] = field(default_factory=list)
    archives: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    response_tokens: int = 0
    attempts: int = 0


def profile_record_summary(record: dict[str, Any]) -> dict[str, Any]:
    """Keep the filtering inputs and costs without copying token arrays into the archive index."""
    trajectory = record["trajectory"]
    extra = trajectory["environment_extras"]["extra_info"]
    summary = {
        key: record[key]
        for key in ("record_id", "provenance", "phase", "global_step", "verification_result", "disposition", "reward")
    }
    summary["trajectory"] = {
        "repetition_id": trajectory["repetition_id"],
        "environment_extras": {
            "extra_info": {"source_id": extra["source_id"], "profiling_attempt": extra.get("profiling_attempt", 0)}
        },
    }
    summary["prompt_tokens"] = len(record["prompt"]["token_ids"])
    summary["response_tokens"] = len(record["response"]["token_ids"])
    return summary


def profile_sample_outcome(record: dict[str, Any]) -> tuple[float | None, str | None]:
    """Distinguish binary model outcomes, context exclusions, and infrastructure errors."""
    verdict = record["verification_result"]
    if verdict is not None and is_context_exclusion(verdict):
        return None, "context_window"
    if record["disposition"]["exception_type"] is not None or verdict is None or verdict["status"] != "verified":
        return None, "infrastructure_error"
    score = verdict["score"]
    if score not in (0, 1) or record["reward"]["outcome"] != score or record["reward"]["shaped"] != score:
        raise ValueError("Resumed profiling requires unshaped binary verifier outcomes")
    return float(score), None


def _read_archive(filesystem, path: str) -> list[dict[str, Any]]:
    with filesystem.open(path, "rb", block_size=65536, cache_type="none") as source, ZipFile(source) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    summaries = {entry["record_id"]: entry.get("profiling") for entry in manifest["records"]}
    if any(record is None for record in summaries.values()):
        # Read legacy archives once rather than fetching every gzip member separately.
        with filesystem.open(path, "rb") as source:
            payload = source.read()
        with ZipFile(BytesIO(payload)) as archive:
            for entry in manifest["records"]:
                if summaries[entry["record_id"]] is None:
                    summaries[entry["record_id"]] = profile_record_summary(
                        json.loads(gzip.decompress(archive.read(entry["entry"])))
                    )
    records = []
    for entry in manifest["records"]:
        record = summaries[entry["record_id"]]
        if record["record_id"] != entry["record_id"]:
            raise ValueError("Profiling index does not match its retained record")
        records.append(record)
    return records


def load_profile_history(uri: str, model_path: str, model_revision: str, read_concurrency: int = 8) -> ProfileHistory:
    """Keep the first committed draw for each slot; discard later preemption replays.

    New archives carry compact profiling entries. Existing archives are read once
    to recover the same fields. Both paths validate the frozen policy identity.
    """
    if not isinstance(read_concurrency, int) or read_concurrency < 1:
        raise ValueError("Profiling history read concurrency must be positive")
    history = ProfileHistory()
    filesystem, root = filesystem_and_path(uri)
    if not filesystem.exists(root):
        return history
    files = filesystem.find(root, detail=True)

    def archive_order(path: str) -> tuple[float, str]:
        info = files[path]
        modified = info.get("LastModified", info.get("mtime"))
        if modified is None:
            raise ValueError(f"Profiling archive has no commit timestamp: {path}")
        timestamp = modified.timestamp() if isinstance(modified, datetime) else float(modified)
        return timestamp, path

    archives = sorted((path for path in files if path.endswith(".zip")), key=archive_order)
    # Bound both concurrent reads and buffered results. map yields in submission
    # order, preserving the first committed draw even if later reads finish first.
    with ThreadPoolExecutor(max_workers=read_concurrency) as executor:
        for paths in batched(archives, read_concurrency):
            records_by_archive = executor.map(lambda path: _read_archive(filesystem, path), paths)
            for path, records in zip(paths, records_by_archive, strict=True):
                history.archives.append(path)
                for record in records:
                    provenance = record["provenance"]
                    if (
                        provenance["model_path"] != model_path
                        or provenance["model_source_identity"] != model_revision
                        or provenance["model_version_step"] != 0
                        or provenance.get("resume_path") is not None
                        or record["phase"] != "eval"
                        or record["global_step"] != 0
                    ):
                        raise ValueError("Profiling resume requires the same frozen initial policy")
                    profile_sample_outcome(record)
                    extra = record["trajectory"]["environment_extras"]["extra_info"]
                    attempt = extra["profiling_attempt"]
                    if not isinstance(attempt, int) or attempt < 0:
                        raise ValueError("Invalid retained profiling attempt")
                    key = (extra["source_id"], record["trajectory"]["repetition_id"], attempt)
                    history.attempts += 1
                    history.prompt_tokens += record["prompt_tokens"]
                    history.response_tokens += record["response_tokens"]
                    if key in history.samples:
                        if profile_sample_outcome(history.samples[key])[1] == "infrastructure_error":
                            raise ValueError(
                                "Ambiguous repeated infrastructure-error attempt; cannot resume automatically"
                            )
                        history.discarded_record_ids.append(record["record_id"])
                        continue
                    history.samples[key] = record
    return history
