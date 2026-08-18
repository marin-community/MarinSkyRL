"""Deterministic composition of verifier-bound RL data sources."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml
from skyrl_gym import get_data_contract

from infra.rl_data.contracts import VerifierDataContract
from infra.rl_data.preparation import (
    PREPARATION_VERSION,
    PROVENANCE_SCHEMA_VERSION,
    PreparationOptions,
    PreparedArtifact,
    TokenCount,
    prepare_artifact,
)
from infra.rl_data.sources import (
    TEST_ONLY_SOURCE_LABELS,
    TEST_ONLY_SOURCE_NAMES,
    PreparedRow,
    Source,
    load_source_rows,
    source_by_name,
)


@dataclass(frozen=True)
class MixtureSlice:
    """One reproducible source slice in a train or validation mixture."""

    source: str
    revision: str
    cap: int | None = None
    minimum_unique_rows: int = 1
    parameters: Mapping[str, Any] = field(default_factory=dict)
    split: str | None = None


@dataclass(frozen=True)
class MixtureSpec:
    """Training and validation slices published as one artifact bundle."""

    train: tuple[MixtureSlice, ...]
    validation: tuple[MixtureSlice, ...]


SourceLookup = Callable[[str], Source]
ContractLookup = Callable[[str], VerifierDataContract]
RowLoader = Callable[[Source, str, Mapping[str, Any]], Iterable[Mapping[str, Any]]]


def _reject_unknown_fields(raw: Mapping[str, Any], supported: set[str], location: str) -> None:
    unknown = set(raw) - supported
    if unknown:
        raise ValueError(f"{location} has unsupported fields: {sorted(unknown)}.")


def _parse_slice(raw: Any, location: str) -> MixtureSlice:
    if not isinstance(raw, Mapping):
        raise TypeError(f"{location} must be a mapping.")
    supported = {"source", "revision", "cap", "minimum_unique_rows", "parameters", "split"}
    _reject_unknown_fields(raw, supported, location)
    source = raw.get("source")
    revision = raw.get("revision")
    if not isinstance(source, str) or not source:
        raise ValueError(f"{location}.source must be a non-empty string.")
    if not isinstance(revision, str) or not revision:
        raise ValueError(f"{location}.revision must be a non-empty immutable revision.")
    cap = raw.get("cap")
    if cap is not None and (type(cap) is not int or cap <= 0):
        raise ValueError(f"{location}.cap must be a positive integer.")
    minimum_unique_rows = raw.get("minimum_unique_rows", 1)
    if type(minimum_unique_rows) is not int or minimum_unique_rows <= 0:
        raise ValueError(f"{location}.minimum_unique_rows must be a positive integer.")
    parameters = raw.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise TypeError(f"{location}.parameters must be a mapping.")
    split = raw.get("split")
    if split is not None and (not isinstance(split, str) or not split):
        raise ValueError(f"{location}.split must be a non-empty string when specified.")
    return MixtureSlice(source, revision, cap, minimum_unique_rows, dict(parameters), split)


def load_mixture_spec(path: Path) -> MixtureSpec:
    """Load and validate a mixture YAML file."""
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, Mapping):
        raise TypeError("Mixture YAML must contain a mapping.")
    _reject_unknown_fields(raw, {"train", "validation"}, "Mixture YAML")

    def parse_split(name: str) -> tuple[MixtureSlice, ...]:
        entries = raw.get(name)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Mixture YAML {name!r} must be a non-empty list.")
        return tuple(_parse_slice(entry, f"{name}[{index}]") for index, entry in enumerate(entries))

    return MixtureSpec(train=parse_split("train"), validation=parse_split("validation"))


def _compose(artifacts: list[PreparedArtifact], seed: int, split: str) -> PreparedArtifact:
    rows: list[PreparedRow] = []
    source_provenance: list[dict[str, Any]] = []
    total_rows = sum(len(artifact.rows) for artifact in artifacts)
    for artifact in artifacts:
        for row in artifact.rows:
            row["extra_info"]["source_index"] = row["extra_info"]["index"]
            rows.append(row)
        provenance = dict(artifact.provenance)
        provenance["share"] = len(artifact.rows) / total_rows
        source_provenance.append(provenance)

    random.Random(seed).shuffle(rows)
    for index, row in enumerate(rows):
        row["extra_info"]["index"] = index

    return PreparedArtifact(
        rows=rows,
        provenance={
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "preparation_version": PREPARATION_VERSION,
            "mixture": True,
            "split": split,
            "seed": seed,
            "counts": {"emitted_rows": total_rows, "sources": len(artifacts)},
            "sources": source_provenance,
        },
    )


def prepare_mixture(
    spec: MixtureSpec,
    token_count: TokenCount,
    max_prompt_tokens: int,
    seed: int,
    *,
    allow_train_on_test: bool = False,
    source_lookup: SourceLookup = source_by_name,
    contract_lookup: ContractLookup = get_data_contract,
    row_loader: RowLoader = load_source_rows,
) -> tuple[PreparedArtifact, PreparedArtifact]:
    """Prepare independently validated source slices, then compose each split."""

    def prepare_split(slices: tuple[MixtureSlice, ...], split: str) -> PreparedArtifact:
        artifacts: list[PreparedArtifact] = []
        for source_slice in slices:
            source = source_lookup(source_slice.source)
            if split == "train" and source.name in TEST_ONLY_SOURCE_NAMES and not allow_train_on_test:
                label = TEST_ONLY_SOURCE_LABELS[source.name]
                raise ValueError(f"{label} is test-only; enable allow_train_on_test to train on it.")
            if source_slice.split is not None:
                source = replace(source, split=source_slice.split)
            contract = contract_lookup(source.env_id)
            artifact = prepare_artifact(
                source,
                row_loader(source, source_slice.revision, source_slice.parameters),
                contract,
                token_count,
                PreparationOptions(
                    source_revision=source_slice.revision,
                    max_prompt_tokens=max_prompt_tokens,
                    minimum_unique_rows=source_slice.minimum_unique_rows,
                    artifact_split=split,
                    seed=seed,
                    subsample_n=source_slice.cap,
                ),
            )
            artifact = replace(
                artifact,
                provenance={
                    **artifact.provenance,
                    "mixture_slice": {
                        "cap": source_slice.cap,
                        "parameters": dict(source_slice.parameters),
                    },
                },
            )
            artifacts.append(artifact)
        return _compose(artifacts, seed, split)

    return prepare_split(spec.train, "train"), prepare_split(spec.validation, "validation")
