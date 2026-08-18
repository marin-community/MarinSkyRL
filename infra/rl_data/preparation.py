"""Validation, provenance, and atomic publication for RLVR dataset artifacts."""

from __future__ import annotations

import json
import math
import random
import shutil
import tempfile
import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from infra.rl_data.contracts import VerifierDataContract
from infra.rl_data.sources import PreparedRow, Source

TokenCount = Callable[[str], int]
ParquetWriter = Callable[[list[PreparedRow], Path], None]
PREPARATION_VERSION = 2
PROVENANCE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PreparationOptions:
    """Explicit reproducibility and quality gates for one source artifact."""

    source_revision: str
    max_prompt_tokens: int
    minimum_unique_rows: int
    artifact_split: str = "train"
    seed: int = 42
    subsample_n: int | None = None
    unique_cap: int | None = None
    minimum_yield_fraction: float | None = None


@dataclass(frozen=True)
class PreparedArtifact:
    """Validated rows and the provenance required to reproduce them."""

    rows: list[PreparedRow]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class ConversionFailure:
    index: int
    error: str

    def provenance(self) -> dict[str, int | str]:
        return {"index": self.index, "error": self.error}


@dataclass(frozen=True)
class ConversionResult:
    raw_rows: int
    converted_rows: int
    unique_rows: list[PreparedRow]
    failures: list[ConversionFailure]

    @property
    def yield_fraction(self) -> float:
        return self.converted_rows / self.raw_rows if self.raw_rows else 0.0


def _prompt_content(row: PreparedRow) -> str:
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or len(prompt) != 1:
        raise ValueError("Prepared row must contain exactly one chat prompt.")
    content = prompt[0].get("content") if isinstance(prompt[0], Mapping) else None
    if not isinstance(content, str) or not content:
        raise ValueError("Prepared row must contain non-empty user content.")
    return content


def _validate_row(row: PreparedRow, source: Source) -> None:
    required_fields = {"data_source", "prompt", "env_class", "reward_model", "extra_info"}
    missing_fields = required_fields - row.keys()
    if missing_fields:
        raise ValueError(f"Prepared row is missing required fields: {sorted(missing_fields)}.")
    if row["data_source"] != source.dataset_id:
        raise ValueError(f"Prepared row data_source does not match {source.dataset_id!r}.")
    if row["env_class"] != source.env_id:
        raise ValueError(f"Prepared row env_class does not match {source.env_id!r}.")
    reward_model = row["reward_model"]
    if not isinstance(reward_model, Mapping) or not isinstance(reward_model.get("ground_truth"), str):
        raise TypeError("Prepared row must contain a string reward_model.ground_truth.")
    if not reward_model["ground_truth"]:
        raise ValueError("Prepared row ground_truth must be non-empty.")
    if not isinstance(row["extra_info"], Mapping):
        raise TypeError("Prepared row extra_info must be a mapping.")
    _prompt_content(row)


def _percentile(values: list[int], percentile: float) -> int:
    rank = max(0, math.ceil(percentile * len(values)) - 1)
    return sorted(values)[rank]


def _subsample(rows: list[PreparedRow], options: PreparationOptions) -> list[PreparedRow]:
    if options.subsample_n is None or options.subsample_n >= len(rows):
        return rows
    selected = sorted(random.Random(options.seed).sample(range(len(rows)), options.subsample_n))
    return [rows[index] for index in selected]


def _convert_rows(
    source: Source,
    examples: Iterable[Mapping[str, Any]],
    contract: VerifierDataContract,
    options: PreparationOptions,
) -> ConversionResult:
    raw_rows = 0
    converted_rows = 0
    failures: list[ConversionFailure] = []
    unique_rows: list[PreparedRow] = []
    seen_prompts: set[str] = set()
    for index, example in enumerate(examples):
        raw_rows += 1
        try:
            row = source.prepare_row(example, index, contract)
        except (TypeError, ValueError) as error:
            failures.append(ConversionFailure(index, f"{type(error).__name__}: {error}"))
            continue

        row["extra_info"]["split"] = options.artifact_split
        _validate_row(row, source)
        converted_rows += 1
        prompt = _prompt_content(row)
        if prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)
        unique_rows.append(row)
        if options.unique_cap is not None and len(unique_rows) >= options.unique_cap:
            break
    return ConversionResult(raw_rows, converted_rows, unique_rows, failures)


def _check_conversion_yield(
    source: Source,
    conversion: ConversionResult,
    minimum_yield_fraction: float | None,
) -> None:
    if conversion.failures and conversion.unique_rows:
        indices = [failure.index for failure in conversion.failures]
        displayed_indices = indices[:20]
        suffix = f" (+{len(indices) - len(displayed_indices)} more)" if len(indices) > len(displayed_indices) else ""
        warnings.warn(
            f"{source.name} skipped {len(conversion.failures)} of {conversion.raw_rows} rows after conversion "
            f"failures; indices={displayed_indices}{suffix}.",
            UserWarning,
            stacklevel=3,
        )
    if minimum_yield_fraction is not None and conversion.yield_fraction < minimum_yield_fraction:
        raise ValueError(
            f"{source.name} conversion yield {conversion.converted_rows}/{conversion.raw_rows} "
            f"({conversion.yield_fraction:.1%}) is below minimum_yield_fraction={minimum_yield_fraction}; "
            f"failed indices={[failure.index for failure in conversion.failures]}."
        )


def prepare_artifact(
    source: Source,
    examples: Iterable[Mapping[str, Any]],
    contract: VerifierDataContract,
    token_count: TokenCount,
    options: PreparationOptions,
) -> PreparedArtifact:
    """Transform and validate source examples before any publishable artifact exists."""
    if source.env_id != contract.env_id:
        raise ValueError(f"Source {source.name!r} requires {source.env_id!r}, got {contract.env_id!r}.")
    if options.max_prompt_tokens <= 0:
        raise ValueError("max_prompt_tokens must be positive.")
    if options.minimum_unique_rows <= 0:
        raise ValueError("minimum_unique_rows must be positive.")
    if options.minimum_yield_fraction is not None and not 0 <= options.minimum_yield_fraction <= 1:
        raise ValueError("minimum_yield_fraction must be between 0 and 1.")
    if options.artifact_split not in {"train", "validation"}:
        raise ValueError("artifact_split must be 'train' or 'validation'.")
    if options.unique_cap is not None and options.unique_cap < options.minimum_unique_rows:
        raise ValueError("unique_cap cannot be smaller than minimum_unique_rows.")

    conversion = _convert_rows(source, examples, contract, options)
    _check_conversion_yield(source, conversion, options.minimum_yield_fraction)
    unique_rows = conversion.unique_rows

    if not unique_rows:
        if conversion.failures:
            first = conversion.failures[0]
            raise ValueError(
                f"{source.name} produced no unique rows; first conversion failure at row "
                f"{first.index}: {first.error}."
            )
        raise ValueError(f"{source.name} produced no unique rows.")
    if len(unique_rows) < options.minimum_unique_rows:
        raise ValueError(
            f"{source.name} produced {len(unique_rows)} unique rows, below minimum_unique_rows="
            f"{options.minimum_unique_rows}."
        )

    prompt_tokens = [token_count(_prompt_content(row)) for row in unique_rows]
    if min(prompt_tokens) < 0:
        raise ValueError("token_count must return non-negative values.")
    maximum = max(prompt_tokens)
    token_summary = {
        "min": min(prompt_tokens),
        "p50": median(prompt_tokens),
        "p95": _percentile(prompt_tokens, 0.95),
        "max": maximum,
    }
    if maximum > options.max_prompt_tokens:
        raise ValueError(
            f"{source.name} prompt tokens {token_summary} exceed max_prompt_tokens={options.max_prompt_tokens}."
        )

    rows = _subsample(unique_rows, options)
    for index, row in enumerate(rows):
        row["extra_info"]["index"] = index

    provenance = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "preparation_version": PREPARATION_VERSION,
        "source": {
            "name": source.name,
            "dataset_id": source.dataset_id,
            "revision": options.source_revision,
            "split": source.split,
        },
        "verifier": {"env_class": source.env_id, "prompt_instruction": contract.prompt_instruction},
        "preparation": {
            "seed": options.seed,
            "subsample_n": options.subsample_n,
            "max_prompt_tokens": options.max_prompt_tokens,
            "minimum_unique_rows": options.minimum_unique_rows,
            "minimum_yield_fraction": options.minimum_yield_fraction,
            "unique_cap": options.unique_cap,
            "artifact_split": options.artifact_split,
        },
        "counts": {
            "raw_rows": conversion.raw_rows,
            "converted_rows": conversion.converted_rows,
            "malformed_rows_skipped": len(conversion.failures),
            "unique_rows": len(unique_rows),
            "emitted_rows": len(rows),
        },
        "conversion_failures": [failure.provenance() for failure in conversion.failures],
        "prompt_tokens": token_summary,
        "verification": source.verification,
    }
    return PreparedArtifact(rows=rows, provenance=provenance)


def _write_parquet(rows: list[PreparedRow], path: Path) -> None:
    import datasets

    datasets.Dataset.from_list(rows).to_parquet(str(path))


def write_artifact(
    artifact: PreparedArtifact,
    output_dir: Path,
    parquet_writer: ParquetWriter = _write_parquet,
) -> None:
    """Publish one train artifact atomically, leaving no target after a failed write."""
    _write_bundle({"train": artifact}, output_dir, parquet_writer)


def write_bundle(
    train: PreparedArtifact,
    validation: PreparedArtifact,
    output_dir: Path,
    parquet_writer: ParquetWriter = _write_parquet,
) -> None:
    """Publish paired train and validation parquet files with combined provenance."""
    _write_bundle({"train": train, "validation": validation}, output_dir, parquet_writer)


def _write_bundle(
    artifacts: Mapping[str, PreparedArtifact],
    output_dir: Path,
    parquet_writer: ParquetWriter,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {output_dir}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        for name, artifact in artifacts.items():
            parquet_writer(artifact.rows, staging_dir / f"{name}.parquet")
        provenance = {name: artifact.provenance for name, artifact in artifacts.items()}
        (staging_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
        staging_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
