"""Keep consumed token versions separate from rollout lease ages."""

from __future__ import annotations

from collections import Counter
from itertools import groupby

import numpy as np


def consumed_policy_versions(batch, uids: list[str], consuming_version: int) -> tuple[dict, dict[str, float]]:
    """Archive exact version spans and counts after trajectory selection.

    Versions name completed trainer steps. The applied optimizer-update ledger
    converts their differences to optimizer ages; skipped updates must count zero.
    """
    versions = batch.get("token_policy_versions")
    if versions is None:
        raise ValueError("consumed rollout batch lacks measured token policy versions")
    rows = []
    counts = Counter()
    mixed = 0
    for uid, tokens, mask, measured in zip(uids, batch["response_ids"], batch["loss_masks"], versions, strict=True):
        if len(tokens) != len(mask) or len(measured) != len(tokens):
            raise ValueError("consumed token policy versions do not match response IDs and loss masks")
        spans = []
        position = 0
        for version, values in groupby(measured):
            length = sum(1 for _ in values)
            spans.append({"start": position, "end": position + length, "version": version})
            position += length
        active = []
        for version, keep in zip(measured, mask, strict=True):
            if keep:
                if isinstance(version, bool) or not isinstance(version, int) or not 0 <= version <= consuming_version:
                    raise ValueError("consumed loss token has an unknown or future generating policy")
                counts[consuming_version - version] += 1
                active.append(version)
        mixed += len(set(active)) > 1
        rows.append({"uid": uid, "response_token_ids": tokens, "loss_mask": mask, "policy_version_spans": spans})
    ages = np.asarray(list(counts), dtype=np.int64)
    weights = np.asarray(list(counts.values()), dtype=np.int64)
    total = int(weights.sum())
    metrics = {
        "async/mixed_policy_response_fraction": mixed / len(rows) if rows else 0.0,
        "async/measured_policy_loss_tokens": float(total),
    }
    if counts:
        metrics.update(
            {
                "async/token_publication_gap_mean": float(np.dot(ages, weights) / total),
                "async/token_publication_gap_max": float(ages.max()),
                "async/token_publication_gap_min": float(ages.min()),
            }
        )
    record = {
        "schema_version": 1,
        "consuming_policy_version": consuming_version,
        "version_unit": "completed trainer steps; convert using the applied optimizer-update ledger",
        "loss_token_publication_gap_counts": {str(age): count for age, count in sorted(counts.items())},
        "rows": rows,
    }
    return record, metrics
