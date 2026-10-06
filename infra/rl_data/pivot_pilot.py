"""Prepare and freeze the task-disjoint SWE behavior pilot."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from infra.rl_data.pivot import file_sha256, heldout_trajectories, pivot_selected
from skyrl_gym.envs.nemotron_ultra.tool_call import PIVOT_VERIFIER_IDENTITIES
from infra.rl_data.pivot_recipe import recipe, STUDENTS

SEED = 42
ARMS = ("sft_all", "sft_random", "sft_selected", "rl_tool_name", "rl_nemo", "rl_exact")
TEMPLATE = Path(__file__).resolve().parents[2] / "cloud/iris/configs/snowball_pivotrl_96gpu.yaml"


def seeded_hash(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def row_identity(row: dict) -> tuple[str, str]:
    extra = row["extra_info"]
    record = json.loads(extra["nemotron_ultra"]["record_json"])
    return extra["source_id"], record["metadata"]["instance_id"]


def identity_hash(rows: list[dict]) -> str:
    return hashlib.sha256("\n".join(sorted(row_identity(row)[0] for row in rows)).encode()).hexdigest()


def prepare(prepared: Path, output: Path) -> dict:
    """Split entire tasks and record precisely which candidates need new profiling."""
    release = pq.read_table(prepared / "release.parquet")
    rows = release.to_pylist()
    if len({row_identity(row)[0] for row in rows}) != len(rows):
        raise ValueError("Release contains duplicate source identities")
    counts = Counter(row_identity(row)[1] for row in rows)
    heldout = heldout_trajectories(counts, 256, SEED)
    train = [row for row in rows if row_identity(row)[1] not in heldout]
    validation = [row for row in rows if row_identity(row)[1] in heldout]
    tasks = sorted(heldout, key=seeded_hash)[:64]
    quick = [
        min(
            (row for row in validation if row_identity(row)[1] == task),
            key=lambda row: seeded_hash(row_identity(row)[0]),
        )
        for task in tasks
    ]
    old = {row_identity(row)[0] for row in pq.read_table(prepared / "candidates.parquet").to_pylist()}
    missing = [row for row in train if row_identity(row)[0] not in old]
    if (len(train), len(validation), len(heldout), len(quick), len(missing)) != (3722, 256, 154, 64, 240):
        raise ValueError("Input release does not reproduce the approved seed-42 split")
    output.mkdir(parents=True, exist_ok=False)
    groups = dict(candidates=train, validation=validation, quick=quick, missing_profile=missing)
    for name, group in groups.items():
        for row in group:
            row["extra_info"]["split"] = "validation" if name in {"validation", "quick"} else "train"
        pq.write_table(pa.Table.from_pylist(group, schema=release.schema), output / f"{name}.parquet")
    manifest = {
        "parent": json.loads((prepared / "manifest.json").read_text()),
        "split_seed": SEED,
        "split_unit": "metadata.instance_id",
        "validation_tasks": sorted(heldout),
        "hash_order": "sha256(utf8('42:' + identity))",
        "quick_tasks": tasks,
        "quick_source_ids": [row_identity(row)[0] for row in quick],
        "counts": {name: len(group) for name, group in groups.items()},
        "identity_hashes": {name: identity_hash(group) for name, group in groups.items()},
        "artifacts": {name: {"sha256": file_sha256(output / f"{name}.parquet")} for name in groups},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def freeze(
    split: Path, rescored: list[Path], output: Path, student: str, exclusions: dict[str, str] | None = None
) -> dict:
    """Freeze tool-name selection only after complete, terminal K=8 coverage."""
    table = pq.read_table(split / "candidates.parquet")
    rows = table.to_pylist()
    candidates = {row_identity(row)[0] for row in rows}
    exclusions = exclusions or {}
    if exclusions.keys() - candidates or any(not reason for reason in exclusions.values()):
        raise ValueError("Exclusions must identify training candidates and state their reasons")
    groups = defaultdict(dict)
    record_ids = set()
    for path in rescored:
        for line in path.open():
            record = json.loads(line)
            if record["source_id"] not in candidates:
                continue
            if record["record_id"] in record_ids:
                raise ValueError(f"Duplicate profile record: {record['record_id']}")
            record_ids.add(record["record_id"])
            key = (record["repetition_id"], record["profiling_attempt"])
            if key in groups[record["source_id"]]:
                raise ValueError("Duplicate profiling repetition/attempt")
            groups[record["source_id"]][key] = record
    eligible, selected, statistics = [], [], []
    signal_groups = defaultdict(list)
    for row in rows:
        source_id, task = row_identity(row)
        attempts = groups[source_id]
        if {key[0] for key in attempts} != set(range(8)):
            raise ValueError(f"Incomplete K=8 coverage for {source_id}")
        terminal = []
        for repetition in range(8):
            history = sorted((a, r) for (rep, a), r in attempts.items() if rep == repetition)
            if [a for a, _ in history] != list(range(len(history))):
                raise ValueError("Noncontiguous profiling retry history")
            if any(r["status"] != "infrastructure_error" for _, r in history[:-1]):
                raise ValueError("A successful or excluded profiling attempt was resampled")
            terminal.append(history[-1][1])
        usable = all(record["status"] == "verified" for record in terminal) and source_id not in exclusions
        passed = None
        if usable:
            scores = [record["scores"]["tool_name"] for record in terminal]
            if any(score not in (0, 1) for score in scores):
                raise ValueError("Profiling rewards must be binary")
            passed = int(sum(scores))
            eligible.append(row)
            for verifier in ("tool_name", "nemo", "exact"):
                values = [record["scores"][verifier] for record in terminal]
                if any(value not in (0, 1) for value in values):
                    raise ValueError("Profiling verifier grades must be binary")
                signal_groups[verifier].append(values)
            if pivot_selected(passed, 8, 0.5):
                selected.append(row)
        statistics.append(
            dict(source_id=source_id, task=task, usable=usable, successes=passed, exclusion=exclusions.get(source_id))
        )
    if not selected:
        raise ValueError("No pivots selected")
    control = random.Random(SEED).sample(sorted(eligible, key=lambda row: row_identity(row)[0]), len(selected))
    output.mkdir(parents=True, exist_ok=False)
    datasets = dict(all_train=eligible, train=selected, random_train=control)
    for name, group in datasets.items():
        pq.write_table(pa.Table.from_pylist(group, schema=table.schema), output / f"{name}.parquet")
    group_signal = {}
    for verifier, groups in signal_groups.items():
        for size in (2, 4, 8):
            mixed = 0
            for index, values in enumerate(groups):
                sample = random.Random(int(seeded_hash(f"{verifier}:{index}")[:16], 16)).sample(values, size)
                mixed += 0 < sum(sample) < size
            group_signal[f"{verifier}_g{size}_mixed_fraction"] = mixed / len(groups)
    manifest = dict(
        parent=json.loads((split / "manifest.json").read_text()),
        profile_policy=STUDENTS[student][0],
        profile_revision=STUDENTS[student][1],
        profiling_samples=8,
        difficulty_threshold=0.5,
        selection_reward="tool_name",
        seed=SEED,
        profiles=[dict(path=str(path), sha256=file_sha256(path)) for path in rescored],
        counts={name: len(group) for name, group in datasets.items()},
        excluded=len(rows) - len(eligible),
        explicit_exclusions=exclusions,
        signal_availability=group_signal,
        identity_hashes={name: identity_hash(group) for name, group in datasets.items()},
        artifacts={name: {"sha256": file_sha256(output / f"{name}.parquet")} for name in datasets},
    )
    (output / "statistics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in statistics))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def pilot_recipe(
    student: str, arm: str, train_uri: str, validation_uri: str, split_manifest: dict, baseline_cache: str
) -> dict:
    """Compile one of six arms with explicit stopping and evaluation schedules."""
    if arm not in ARMS:
        raise ValueError(arm)
    is_rl = arm.startswith("rl_")
    raw = recipe(student, "pivotrl" if is_rl else "sft", train_uri, [validation_uri], 0.001, TEMPLATE)
    trainer, generator = raw["trainer"], raw["generator"]
    trainer.update(
        max_steps=20 if is_rl else 100000,
        epochs=100000,
        eval_interval=-1,
        eval_before_train=True,
        eval_loss_token_interval=None,
        loss_token_budget=None if is_rl else 1000000,
        ckpt_interval=5 if is_rl else -1,
        dump_data_batch=True,
    )
    trainer["pivot_pilot"] = dict(
        arm=arm,
        quick_source_ids=split_manifest["quick_source_ids"],
        split_hash=split_manifest["identity_hashes"]["validation"],
        quick_hash=split_manifest["identity_hashes"]["quick"],
        baseline_cache=baseline_cache,
        archive_root="${generator.trajectory_retention.output_path}",
        grade_table_root="${generator.trajectory_retention.output_path}/schema_v5/grades",
    )
    generator["n_samples_per_prompt"] = 8
    generator["trajectory_retention"].update(
        enabled=True,
        required=True,
        sample_fraction=1.0,
        max_bytes_per_step=None,
        max_bytes_per_run=None,
        grade_table=True,
    )
    raw["environment"]["skyrl_gym"]["nemotron_ultra"]["pivot_reward"] = arm[3:] if is_rl else "tool_name"
    raw["environment"]["skyrl_gym"]["nemotron_ultra"]["pivot_arm"] = arm
    raw["pivot"].update(
        verifier_identities=dict(PIVOT_VERIFIER_IDENTITIES),
        train_data_uri=train_uri,
        evaluation_rows=256,
        quick_rows=64,
    )
    return raw


def write_pilot_recipes(
    frozen: Path, split: Path, output: Path, student: str, artifact_root: str, validation_uri: str, baseline_cache: str
) -> list[Path]:
    """Write the six immutable-data arm recipes for one student."""
    manifest = json.loads((frozen / "manifest.json").read_text())
    split_manifest = json.loads((split / "manifest.json").read_text())
    uris = {name: f"{artifact_root.rstrip('/')}/{name}.parquet" for name in ("all_train", "train", "random_train")}
    selected_uri, selected = uris["train"], manifest["identity_hashes"]["train"]
    recipes = {}
    for arm in ARMS:
        train = uris["all_train"] if arm == "sft_all" else uris["random_train"] if arm == "sft_random" else selected_uri
        raw = pilot_recipe(student, arm, train, validation_uri, split_manifest, baseline_cache)
        raw["pivot"].update(selected_data_hash=selected, selected_data_uri=selected_uri)
        raw["trainer"]["pivot_pilot"].update(
            train_data_sha256=manifest["artifacts"][Path(train).name.removesuffix(".parquet")]["sha256"],
            validation_data_sha256=split_manifest["artifacts"]["validation"]["sha256"],
            quick_data_sha256=split_manifest["artifacts"]["quick"]["sha256"],
            selected_data_uri=selected_uri,
            selected_data_sha256=manifest["artifacts"]["train"]["sha256"],
            verifier_tool_name_id=PIVOT_VERIFIER_IDENTITIES["tool_name"],
            verifier_nemo_id=PIVOT_VERIFIER_IDENTITIES["nemo"],
            verifier_exact_id=PIVOT_VERIFIER_IDENTITIES["exact"],
        )
        recipes[arm] = raw
    output.mkdir(parents=True, exist_ok=False)
    paths = []
    for arm, raw in recipes.items():
        path = output / f"{student}-{arm}.yaml"
        path.write_text(yaml.safe_dump(raw, sort_keys=False))
        paths.append(path)
    manifest["split_hash"] = split_manifest["identity_hashes"]["validation"]
    manifest["quick_hash"] = split_manifest["identity_hashes"]["quick"]
    manifest["recipe_hashes"] = {path.name: file_sha256(path) for path in paths}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return paths


def write_profile_recipe(split: Path, output: Path, student: str, missing_uri: str, validation_uri: str) -> Path:
    """Write the bounded K=8 recipe for rows absent from the completed profile."""
    raw = recipe(student, "profile", missing_uri, [validation_uri], 0.001, TEMPLATE)
    raw["pivot"].update(
        split_hash=json.loads((split / "manifest.json").read_text())["identity_hashes"]["candidates"],
        expected_prefixes=240,
        profiling_samples=8,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(raw, sort_keys=False))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--prepared", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    frozen = commands.add_parser("freeze")
    frozen.add_argument("--split", type=Path, required=True)
    frozen.add_argument("--rescored", type=Path, nargs="+", required=True)
    frozen.add_argument("--output", type=Path, required=True)
    frozen.add_argument("--student", choices=STUDENTS, required=True)
    frozen.add_argument("--exclusions", type=Path, help="JSON source-ID to reason mapping for unusable rows")
    recipes = commands.add_parser("recipes")
    recipes.add_argument("--frozen", type=Path, required=True)
    recipes.add_argument("--split", type=Path, required=True)
    recipes.add_argument("--output", type=Path, required=True)
    recipes.add_argument("--student", choices=STUDENTS, required=True)
    recipes.add_argument("--artifact-root", required=True)
    recipes.add_argument("--validation-uri", required=True)
    recipes.add_argument("--baseline-cache", required=True)
    profile = commands.add_parser("profile-recipe")
    profile.add_argument("--split", type=Path, required=True)
    profile.add_argument("--output", type=Path, required=True)
    profile.add_argument("--student", choices=STUDENTS, required=True)
    profile.add_argument("--missing-uri", required=True)
    profile.add_argument("--validation-uri", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.prepared, args.output)
    elif args.command == "freeze":
        exclusions = json.loads(args.exclusions.read_text()) if args.exclusions else None
        result = freeze(args.split, args.rescored, args.output, args.student, exclusions)
    elif args.command == "recipes":
        paths = write_pilot_recipes(
            args.frozen,
            args.split,
            args.output,
            args.student,
            args.artifact_root,
            args.validation_uri,
            args.baseline_cache,
        )
        result = {"recipes": [str(path) for path in paths]}
    else:
        result = {
            "recipe": str(
                write_profile_recipe(args.split, args.output, args.student, args.missing_uri, args.validation_uri)
            )
        }
    print(json.dumps({key: value for key, value in result.items() if key != "parent"}, indent=2))


if __name__ == "__main__":
    main()
