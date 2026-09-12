"""Command-line entry point for reproducible RLVR dataset preparation."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from skyrl_gym import get_data_contract

from infra.rl_data.mixtures import load_mixture_spec, prepare_mixture
from infra.rl_data.nemotron_ultra_swe import prepare_swe_task_artifact
from infra.rl_data.preparation import (
    PreparationOptions,
    PreparedArtifact,
    TokenCount,
    ordered_tail_holdout,
    prepare_artifact,
    write_bundle,
)
from infra.rl_data.sources import (
    SOURCES,
    TEST_ONLY_SOURCE_LABELS,
    TEST_ONLY_SOURCE_NAMES,
    load_source_rows,
    source_by_name,
)


class _PreparationArgumentError(ValueError):
    pass


def _token_counter(tokenizer_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return lambda text: len(tokenizer(text, add_special_tokens=False).input_ids)


def _options(args: argparse.Namespace, source_name: str, revision: str) -> PreparationOptions:
    minimum_unique_rows = args.minimum_unique_rows
    if minimum_unique_rows is None:
        minimum_unique_rows = 1000 if source_name == "dapo_math" else 1
    unique_cap = args.unique_cap
    if unique_cap is None and source_name == "dapo_math":
        unique_cap = 20_000
    return PreparationOptions(
        source_revision=revision,
        max_prompt_tokens=args.max_prompt_tokens,
        minimum_unique_rows=minimum_unique_rows,
        seed=args.seed,
        subsample_n=args.subsample_n,
        unique_cap=unique_cap,
        minimum_yield_fraction=args.minimum_yield_fraction,
    )


def _prepare_source_artifact(
    source_name: str,
    revision: str,
    token_count: TokenCount,
    options: PreparationOptions,
    *,
    allow_train_on_test: bool,
) -> PreparedArtifact:
    source = source_by_name(source_name)
    if options.artifact_split == "train" and source.name in TEST_ONLY_SOURCE_NAMES and not allow_train_on_test:
        label = TEST_ONLY_SOURCE_LABELS[source.name]
        raise _PreparationArgumentError(
            f"{label} is test-only; pass --allow-train-on-test to use it as a training source."
        )
    return prepare_artifact(
        source,
        load_source_rows(source, revision),
        get_data_contract(source.env_id),
        token_count,
        options,
    )


def _prepare_ordered_tail(args: argparse.Namespace, token_count: TokenCount) -> None:
    source_name = str(args.source)
    revision = str(args.revision)
    artifact = _prepare_source_artifact(
        source_name,
        revision,
        token_count,
        _options(args, source_name, revision),
        allow_train_on_test=args.allow_train_on_test,
    )
    train, validation = ordered_tail_holdout(artifact, args.validation_tail_rows)
    write_bundle(train, validation, args.output_dir)


def _prepare_dual_source(args: argparse.Namespace, token_count: TokenCount) -> None:
    train_source_name = str(args.source)
    train_revision = str(args.revision)
    validation_source_name = str(args.validation_source)
    validation_revision = str(args.validation_revision)
    train_options = _options(args, train_source_name, train_revision)
    validation_options = replace(
        _options(args, validation_source_name, validation_revision),
        artifact_split="validation",
        subsample_n=None,
        unique_cap=None,
    )
    train = _prepare_source_artifact(
        train_source_name,
        train_revision,
        token_count,
        train_options,
        allow_train_on_test=args.allow_train_on_test,
    )
    validation = _prepare_source_artifact(
        validation_source_name,
        validation_revision,
        token_count,
        validation_options,
        allow_train_on_test=args.allow_train_on_test,
    )
    write_bundle(train, validation, args.output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nemotron-ultra-swe-tasks",
        action="store_true",
        help="Build the complete Harbor SWE sidechannel for the released RLVR blends.",
    )
    parser.add_argument("--mixture", type=Path, help="YAML file declaring train and validation source slices.")
    parser.add_argument("--source", choices=sorted(SOURCES))
    parser.add_argument("--revision", help="Immutable Hugging Face revision for the training source.")
    parser.add_argument("--validation-source", choices=sorted(SOURCES))
    parser.add_argument("--validation-revision", help="Immutable Hugging Face revision for validation.")
    parser.add_argument(
        "--validation-tail-rows",
        type=int,
        help="Reserve this many rows from the ordered tail of the prepared training source for validation.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New local directory for train/validation parquet."
    )
    parser.add_argument("--tokenizer", help="Tokenizer used by the planned training run.")
    parser.add_argument("--max-prompt-tokens", type=int)
    parser.add_argument("--minimum-unique-rows", type=int)
    parser.add_argument(
        "--minimum-yield-fraction",
        type=float,
        help="Optional minimum fraction of source rows that must convert successfully (0 to 1).",
    )
    parser.add_argument("--unique-cap", type=int, help="Stop streaming after this many unique rows.")
    parser.add_argument("--subsample-n", type=int, help="Optional deterministic train-row cap recorded in provenance.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-train-on-test", action="store_true")
    args = parser.parse_args()

    if args.nemotron_ultra_swe_tasks:
        conflicting = (
            args.mixture,
            args.source,
            args.revision,
            args.validation_source,
            args.validation_revision,
            args.validation_tail_rows,
            args.tokenizer,
            args.max_prompt_tokens,
        )
        if any(value is not None for value in conflicting):
            parser.error("--nemotron-ultra-swe-tasks cannot be combined with parquet preparation options.")
        provenance = prepare_swe_task_artifact(args.output_dir)
        print(provenance)
        return

    if args.tokenizer is None or args.max_prompt_tokens is None:
        parser.error("parquet preparation requires --tokenizer and --max-prompt-tokens.")

    counter = _token_counter(args.tokenizer)
    if args.mixture is not None:
        single_source_args = (
            args.source,
            args.revision,
            args.validation_source,
            args.validation_revision,
            args.validation_tail_rows,
        )
        if any(value is not None for value in single_source_args):
            parser.error("--mixture cannot be combined with single-source arguments.")
        train, validation = prepare_mixture(
            load_mixture_spec(args.mixture),
            counter,
            args.max_prompt_tokens,
            args.seed,
            allow_train_on_test=args.allow_train_on_test,
        )
        write_bundle(train, validation, args.output_dir)
        return

    if args.validation_tail_rows is not None:
        if args.validation_tail_rows <= 0:
            parser.error("--validation-tail-rows must be positive.")
        incompatible = {
            "--validation-source": args.validation_source,
            "--validation-revision": args.validation_revision,
            "--subsample-n": args.subsample_n,
            "--unique-cap": args.unique_cap,
        }
        selected = [name for name, value in incompatible.items() if value is not None]
        if selected:
            parser.error(f"--validation-tail-rows cannot be combined with {', '.join(selected)}.")
        missing = [name for name, value in (("--source", args.source), ("--revision", args.revision)) if value is None]
        if missing:
            parser.error(f"--validation-tail-rows requires {', '.join(missing)}.")

        try:
            _prepare_ordered_tail(args, counter)
        except _PreparationArgumentError as error:
            parser.error(str(error))
        return

    missing = [
        name
        for name, value in (
            ("--source", args.source),
            ("--revision", args.revision),
            ("--validation-source", args.validation_source),
            ("--validation-revision", args.validation_revision),
        )
        if value is None
    ]
    if missing:
        parser.error(f"single-source mode requires {', '.join(missing)}; otherwise pass --mixture.")

    try:
        _prepare_dual_source(args, counter)
    except _PreparationArgumentError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
