"""Durable FineStore writes for frozen probes and their scorings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from skyrl_train.config.mismatch_probe import GENERATION_SCORING


def mismatch_schema():
    # The schema lives in Marin FineStore so the producer and report reader
    # validate against precisely the same row contract.
    try:
        from finestore import mismatch
    except ImportError as error:
        raise RuntimeError(
            "The mismatch probe requires a marin-finestore build containing finestore.mismatch"
        ) from error
    return mismatch


@dataclass(frozen=True)
class FrozenProbeSource:
    """Completed source archive rows needed to reuse frozen tokens."""

    manifest: Any
    probes: list[Any]
    generations: dict[str, Any]


class MismatchArchive:
    """One archive writer; each completed score group is a transaction."""

    def __init__(self, uri: str, *, writer_id: str):
        from finestore.store import DataStore

        self.schema = mismatch_schema()
        self.uri = uri
        self.store = DataStore.open(uri, writer_id=writer_id)
        self.schema.register_mismatch_tables(self.store)

    def write(
        self,
        *,
        probes: Sequence | None = None,
        scores: Sequence | None = None,
        layers: Sequence | None = None,
        manifest=None,
    ) -> None:
        with self.store.transaction() as transaction:
            for table, rows in (
                (self.schema.PROBE_TABLE, probes),
                (self.schema.SCORES_TABLE, scores),
                (self.schema.LAYERS_TABLE, layers),
            ):
                for row in rows or ():
                    transaction.table(table).add(row.model_dump())
            if manifest is not None:
                transaction.table(self.schema.MANIFEST_TABLE).add(manifest.model_dump())

    def close(self) -> None:
        self.store.close()


def read_frozen_probe(uri: str) -> FrozenProbeSource:
    """Read completed source rows, including their generation scores."""
    from finestore.reader import ReadView

    schema = mismatch_schema()
    view = ReadView(uri)
    manifests = [schema.ManifestRow.model_validate(row) for row in view.scan(schema.MANIFEST_TABLE).to_pylist()]
    if len(manifests) != 1 or manifests[0].status != "complete":
        raise ValueError(f"reuse_probe source {uri} is not a single complete mismatch archive")
    probes = [schema.ProbeRow.model_validate(row) for row in view.scan(schema.PROBE_TABLE).to_pylist()]
    scores = [schema.ScoreRow.model_validate(row) for row in view.scan(schema.SCORES_TABLE).to_pylist()]
    generations = {row.sample_id: row for row in scores if row.scoring == GENERATION_SCORING}
    if not probes or len(generations) != len(probes):
        raise ValueError("reuse_probe source has incomplete generation-time scores")
    probes.sort(key=lambda row: row.batch_position)
    if any(row.probe_hash != manifests[0].probe_hash for row in probes):
        raise ValueError("reuse_probe source probe hashes do not match the manifest")
    return FrozenProbeSource(manifests[0], probes, generations)
