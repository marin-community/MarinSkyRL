"""Queryable grade sidecars tied one-to-one to immutable trajectory archives."""

import gzip
from io import BytesIO
import json
from zipfile import ZipFile

import pyarrow as pa
import pyarrow.parquet as pq

GRADE_SCHEMA = pa.schema(
    [
        ("record_id", pa.string()),
        ("archive", pa.string()),
        ("entry", pa.string()),
        ("run_id", pa.string()),
        ("phase", pa.string()),
        ("step", pa.int64()),
        ("source_id", pa.string()),
        ("task_id", pa.string()),
        ("group_id", pa.string()),
        ("repetition_id", pa.int64()),
        ("record_kind", pa.string()),
        ("status", pa.string()),
        ("training_arm", pa.string()),
        ("training_reward", pa.string()),
        ("status_reason", pa.string()),
        ("tool_name", pa.float64()),
        ("nemo", pa.float64()),
        ("exact", pa.float64()),
        ("diagnostics_json", pa.string()),
        ("response_tokens", pa.int64()),
        ("loss_tokens_before_budget", pa.int64()),
        ("model_path", pa.string()),
        ("model_revision", pa.string()),
        ("model_step", pa.int64()),
        ("stop_reason", pa.string()),
        ("exception_type", pa.string()),
    ]
)


def grade_table_path(archive: str) -> str:
    return archive.replace("archives/", "grades/", 1).removesuffix(".zip") + ".parquet"


def archive_grade_table(payload: bytes, archive_uri: str) -> bytes:
    """Build exactly one grade row per raw record, including ungraded failures."""
    rows = []
    seen = set()
    with ZipFile(BytesIO(payload)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        for entry in manifest["records"]:
            record = json.loads(gzip.decompress(archive.read(entry["entry"])))
            if record["record_id"] != entry["record_id"] or record["record_id"] in seen:
                raise ValueError("Archive record identity mismatch")
            seen.add(record["record_id"])
            trajectory, provenance = record["trajectory"], record["provenance"]
            extra = trajectory["environment_extras"]["extra_info"]
            source = json.loads(extra["nemotron_ultra"]["record_json"])
            verdict = record["verification_result"] or {}
            diagnostic = verdict.get("diagnostics", {})
            grades = diagnostic.get("pivot", {})
            scores = grades.get("scores", {})
            rows.append(
                dict(
                    record_id=record["record_id"],
                    archive=archive_uri,
                    entry=entry["entry"],
                    run_id=record["run_id"],
                    phase=record["phase"],
                    step=record["global_step"],
                    source_id=extra["source_id"],
                    task_id=source["metadata"]["instance_id"],
                    group_id=f"{record['run_id']}:{record['phase']}:{record['global_step']}:{trajectory['instance_id']}",
                    repetition_id=trajectory["repetition_id"],
                record_kind=diagnostic.get("record_kind", "sampled"),
                status=verdict.get("status", "unavailable"),
                training_arm=grades.get("training_arm"),
                training_reward=grades.get("training_reward"),
                status_reason=verdict.get("reason"),
                    tool_name=scores.get("tool_name"),
                    nemo=scores.get("nemo"),
                    exact=scores.get("exact"),
                    diagnostics_json=json.dumps(diagnostic, sort_keys=True),
                    response_tokens=len(record["response"]["token_ids"]),
                    loss_tokens_before_budget=sum(record["response"]["loss_mask"]),
                    model_path=provenance["model_path"],
                    model_revision=provenance["model_source_identity"],
                    model_step=provenance["model_version_step"],
                    stop_reason=record["response"]["stop_reason"],
                    exception_type=record["disposition"]["exception_type"],
                )
            )
    output = BytesIO()
    pq.write_table(pa.Table.from_pylist(rows, schema=GRADE_SCHEMA), output, compression="zstd")
    return output.getvalue()


def validate_archive_grade_table(payload: bytes, table_payload: bytes) -> None:
    """Check sidecar record identities against the archive manifest after publish or resume."""
    with ZipFile(BytesIO(payload)) as archive:
        expected = [row["record_id"] for row in json.loads(archive.read("manifest.json"))["records"]]
    table = pq.read_table(BytesIO(table_payload), columns=["record_id"])
    actual = table.column("record_id").to_pylist()
    if len(actual) != len(set(actual)) or set(actual) != set(expected) or len(actual) != len(expected):
        raise ValueError("Parquet grade table does not match its raw trajectory archive")
