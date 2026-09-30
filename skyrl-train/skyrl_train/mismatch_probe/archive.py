"""Durable FineStore writes for frozen probes and their scorings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from finestore import mismatch_probe as mismatch
from finestore.reader import ReadView
from finestore.store import DataStore, TransactionTooLarge

from skyrl_train.config.mismatch_probe import CACHE_OFF, FROZEN_RESCORE_SCORER, GENERATION_SCORER, RESCORE_SCORER

COMPLETE_STATUS = mismatch.ArchiveStatus.COMPLETE
BUILDING_STATUS = mismatch.ArchiveStatus.BUILDING


@dataclass(frozen=True)
class FrozenProbeSource:
    """Completed source archive rows needed to reuse frozen tokens."""

    manifest: mismatch.ManifestRow
    probes: list[mismatch.ProbeRow]
    generations: dict[str, mismatch.ScoreRow]
    # The source's prefill reference: its own frozen copy when it reused a probe, else its cache-off re-read.
    rereads: dict[str, mismatch.ScoreRow]


class MismatchArchive:
    """Write each supplied group of archive rows in one transaction."""

    def __init__(self, uri: str, *, writer_id: str, **store_options):
        self.uri = uri
        self.store = DataStore.open(uri, writer_id=writer_id, **store_options)
        mismatch.register_mismatch_tables(self.store)

    def write(
        self,
        *,
        probes: Sequence[mismatch.ProbeRow] | None = None,
        scores: Sequence[mismatch.ScoreRow] | None = None,
        manifest: mismatch.ManifestRow | None = None,
    ) -> None:
        """Commit the rows in as few transactions as FineStore's size limit allows, manifest last.

        Readers accept only an archive whose manifest is complete, so a write split across
        transactions is never read half-finished.
        """
        pending = [
            *((mismatch.PROBE_TABLE, row) for row in probes or ()),
            *((mismatch.SCORES_TABLE, row) for row in scores or ()),
            *(((mismatch.MANIFEST_TABLE, manifest),) if manifest is not None else ()),
        ]
        while pending:
            added = 0
            with self.store.transaction() as transaction:
                for table, row in pending:
                    try:
                        transaction.table(table).add(row.model_dump())
                    except TransactionTooLarge:
                        if not added:
                            raise
                        break
                    added += 1
            pending = pending[added:]

    def close(self) -> None:
        self.store.close()


def read_frozen_probe(uri: str) -> FrozenProbeSource:
    """Read completed source rows, including their generation scores."""
    view = ReadView(uri)
    manifests = [mismatch.ManifestRow.model_validate(row) for row in view.scan(mismatch.MANIFEST_TABLE).to_pylist()]
    if len(manifests) != 1 or manifests[0].status != COMPLETE_STATUS:
        raise ValueError(f"reuse_probe source {uri} is not a single complete mismatch archive")
    probes = [mismatch.ProbeRow.model_validate(row) for row in view.scan(mismatch.PROBE_TABLE).to_pylist()]
    scores = [mismatch.ScoreRow.model_validate(row) for row in view.scan(mismatch.SCORES_TABLE).to_pylist()]
    generations = {row.sample_id: row for row in scores if row.scorer == GENERATION_SCORER and row.update == 0}
    if not probes or len(generations) != len(probes):
        raise ValueError("reuse_probe source has incomplete generation-time scores")
    probes.sort(key=lambda row: row.batch_position)
    if any(row.probe_hash != manifests[0].probe_hash for row in probes):
        raise ValueError("reuse_probe source probe hashes do not match the manifest")
    rereads = {}
    for scorer in (RESCORE_SCORER, FROZEN_RESCORE_SCORER):
        rows = {
            row.sample_id: row
            for row in scores
            if row.scorer == scorer and row.update == 0 and row.cache_mode == CACHE_OFF
        }
        if len(rows) == len(probes):
            rereads = rows
    return FrozenProbeSource(manifests[0], probes, generations, rereads)
