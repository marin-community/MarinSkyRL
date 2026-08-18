"""Command-line entry point for reproducible RLVR dataset preparation."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from infra.rl_data.mixtures import load_mixture_spec, prepare_mixture
from infra.rl_data.preparation import PreparationOptions, prepare_artifact, write_bundle
from infra.rl_data.sources import SOURCES, load_source_rows, source_by_name


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixture", type=Path, help="YAML file declaring train and validation source slices.")
    parser.add_argument("--source", choices=sorted(SOURCES))
    parser.add_argument("--revision", help="Immutable Hugging Face revision for the training source.")
    parser.add_argument("--validation-source", choices=sorted(SOURCES))
    parser.add_argument("--validation-revision", help="Immutable Hugging Face revision for validation.")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New local directory for train/validation parquet."
    )
    parser.add_argument("--tokenizer", required=True, help="Tokenizer used by the planned training run.")
    parser.add_argument("--max-prompt-tokens", type=int, required=True)
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

    counter = _token_counter(args.tokenizer)
    if args.mixture is not None:
        single_source_args = (args.source, args.revision, args.validation_source, args.validation_revision)
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

    train_source = source_by_name(args.source)
    validation_source = source_by_name(args.validation_source)
    if train_source.name == "math500" and not args.allow_train_on_test:
        parser.error("MATH-500 is test-only; pass --allow-train-on-test to use it as a training source.")

    from skyrl_gym import get_data_contract

    train_contract = get_data_contract(train_source.env_id)
    validation_contract = get_data_contract(validation_source.env_id)
    train_options = _options(args, train_source.name, args.revision)
    validation_options = replace(
        _options(args, validation_source.name, args.validation_revision),
        artifact_split="validation",
        subsample_n=None,
        unique_cap=None,
    )
    train = prepare_artifact(
        train_source,
        load_source_rows(train_source, args.revision),
        train_contract,
        counter,
        train_options,
    )
    validation = prepare_artifact(
        validation_source,
        load_source_rows(validation_source, args.validation_revision),
        validation_contract,
        counter,
        validation_options,
    )
    write_bundle(train, validation, args.output_dir)


if __name__ == "__main__":
    main()
