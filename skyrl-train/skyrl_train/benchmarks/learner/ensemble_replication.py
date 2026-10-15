"""Build larger learner-benchmark updates by replicating a routed ensemble's sequences.

A small ensemble (e.g. ``v0/short``: 128 matched-bank sequences routed by a Hero checkpoint) yields updates too
small to amortize the fixed optimizer step. This module draws each larger update from the source ensemble's
sequences, replicated and shuffled with a fixed seed, and carries every copy's routes along so no new routing
pass is needed. Rewards and advantages are redrawn per batch as in ``matched_bank``.

Update sizes share one profile and are told apart by batch index: copy ``c`` of an update of ``n`` sequences is
``batch_<n * BATCH_INDEX_STRIDE + c>`` (e.g. ``batch_1280``, ``batch_1281``, ``batch_5120``), and its manifest
``source["update_sequences"]`` records ``n``.

Known limitation: duplicated sequences share prompts, responses and routing patterns, so expert load across a
replicated update is less diverse than in a real batch of the same size.
"""

import argparse
import dataclasses
import json
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from skyrl_train.benchmarks.learner.batch_artifact import (
    EnsembleBatchManifest,
    LengthProfile,
    batch_manifest,
    ensemble_batch_path,
    list_ensemble,
    read_ensemble_batch,
    write_ensemble_batch,
    write_routes,
)
from skyrl_train.benchmarks.learner.matched_bank import BankSample, bank_batch, synthetic_rewards
from skyrl_train.benchmarks.learner.route_pass import route_requests
from skyrl_train.dataset.routed_expert_batch import RoutedExpertRows
from skyrl_train.training_batch import TrainingInputBatch

BATCH_INDEX_STRIDE = 10
REPLICATION_LIMITATION = "duplicated sequences share prompts, responses and routing patterns"


@dataclass(frozen=True)
class RoutedSample:
    sample: BankSample
    routes: np.ndarray


def replicated_batch_index(update_sequences: int, copy: int) -> int:
    if not 0 <= copy < BATCH_INDEX_STRIDE:
        raise ValueError(f"copy must lie in [0, {BATCH_INDEX_STRIDE}), got {copy}")
    return update_sequences * BATCH_INDEX_STRIDE + copy


def routed_samples(
    root: str, ensemble_version: str, profile: LengthProfile, routes_checkpoint: str
) -> tuple[list[RoutedSample], int, str]:
    """Read every batch of one profile with its routes; return its sequences, expert count and tokenizer."""
    samples: list[RoutedSample] = []
    tokenizers = set()
    num_experts = None
    for path in list_ensemble(root, ensemble_version):
        batch, manifest = read_ensemble_batch(path, routes_checkpoint=routes_checkpoint)
        if manifest.profile != profile:
            continue
        routes = batch.routed_expert_rows
        num_experts = routes.num_experts
        tokenizers.add(manifest.tokenizer)
        row_ids = manifest.source["row_ids"]
        for request, row_id, rows in zip(route_requests(batch), row_ids, routes.rows, strict=True):
            prompt = request.token_ids[: request.prompt_length]
            response = request.token_ids[request.prompt_length :]
            samples.append(RoutedSample(BankSample(row_id, prompt, response), rows))
    if not samples or len(tokenizers) != 1:
        raise ValueError(f"{root}/{ensemble_version}/{profile} needs routed batches with one tokenizer")
    return samples, int(num_experts), tokenizers.pop()


def replicated_draws(num_samples: int, update_sequences: int, copies: int, seed: int) -> list[list[int]]:
    """Split ``copies`` updates' worth of replicated sample indices, shuffled together, into updates.

    Every sample appears ``ceil(update_sequences * copies / num_samples)`` times in the shuffled pool, so each
    sample is used as evenly as the sizes allow.
    """
    total = update_sequences * copies
    repeats = -(-total // num_samples)
    pool = np.random.default_rng(seed).permutation(np.tile(np.arange(num_samples), repeats))[:total]
    return [pool[copy * update_sequences : (copy + 1) * update_sequences].tolist() for copy in range(copies)]


def replicated_batch(
    samples: Sequence[RoutedSample], indices: Sequence[int], *, pad_token_id: int, num_experts: int, reward_seed: int
) -> TrainingInputBatch:
    """Collate the drawn samples and attach each copy's routes."""
    drawn = [samples[index] for index in indices]
    batch = bank_batch(
        [entry.sample for entry in drawn],
        pad_token_id=pad_token_id,
        rewards=synthetic_rewards(len(drawn), reward_seed),
    )
    batch.routed_expert_rows = RoutedExpertRows(
        tuple(entry.routes for entry in drawn), int(batch.metadata["response_length"]), num_experts
    )
    return batch


def build_replicated_ensemble(
    samples: Sequence[RoutedSample],
    *,
    out_root: str,
    ensemble_version: str,
    profile: LengthProfile,
    update_sizes: Sequence[int],
    copies: int,
    seed: int,
    pad_token_id: int,
    num_experts: int,
    tokenizer: str,
    routes_checkpoint: str,
    source: dict,
) -> list[tuple[str, EnsembleBatchManifest]]:
    """Write ``copies`` replicated batches per update size, each with its routes file."""
    written = []
    for size_index, update_sequences in enumerate(update_sizes):
        draw_seed = seed + size_index
        for copy, indices in enumerate(replicated_draws(len(samples), update_sequences, copies, draw_seed)):
            batch_index = replicated_batch_index(update_sequences, copy)
            batch = replicated_batch(
                samples, indices, pad_token_id=pad_token_id, num_experts=num_experts, reward_seed=seed + batch_index
            )
            routes = batch.routed_expert_rows
            batch.routed_expert_rows = None
            manifest = batch_manifest(
                batch,
                ensemble_version=ensemble_version,
                profile=profile,
                batch_index=batch_index,
                source={
                    **source,
                    "update_sequences": update_sequences,
                    "replication_seed": draw_seed,
                    "reward_seed": seed + batch_index,
                    "duplicated": True,
                    "limitation": REPLICATION_LIMITATION,
                    "routes_checkpoint": routes_checkpoint,
                    "row_ids": [samples[index].sample.row_id for index in indices],
                },
                tokenizer=tokenizer,
            )
            path = ensemble_batch_path(out_root, ensemble_version, profile, batch_index)
            write_ensemble_batch(path, batch, manifest)
            write_routes(path, routes_checkpoint, routes)
            written.append((path, manifest))
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m skyrl_train.benchmarks.learner.ensemble_replication")
    parser.add_argument("--source-root", required=True, help="routed source ensemble root path or URI")
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--ensemble-version", required=True)
    parser.add_argument("--profile", required=True, choices=[profile.value for profile in LengthProfile])
    parser.add_argument("--routes-checkpoint", required=True)
    parser.add_argument("--update-sizes", required=True, type=int, nargs="+")
    parser.add_argument("--copies", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--pad-token-id", required=True, type=int, help="policy tokenizer pad token id")
    args = parser.parse_args(argv)

    profile = LengthProfile(args.profile)
    samples, num_experts, tokenizer = routed_samples(
        args.source_root, args.source_version, profile, args.routes_checkpoint
    )
    written = build_replicated_ensemble(
        samples,
        out_root=args.out_root,
        ensemble_version=args.ensemble_version,
        profile=profile,
        update_sizes=args.update_sizes,
        copies=args.copies,
        seed=args.seed,
        pad_token_id=args.pad_token_id,
        num_experts=num_experts,
        tokenizer=tokenizer,
        routes_checkpoint=args.routes_checkpoint,
        source={"replicated_from": f"{args.source_root.rstrip('/')}/{args.source_version}/{profile}"},
    )
    for path, manifest in written:
        summary = {key: value for key, value in dataclasses.asdict(manifest).items() if key != "source"}
        print(json.dumps({"path": path, **summary}))


if __name__ == "__main__":
    main()
