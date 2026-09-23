"""Bounded sampling of the Nemotron Ultra MOPD blend with hardcoded Snowball expert routes."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from infra.rl_data.nemotron_ultra_sample import (
    HUGGING_FACE_RESOLVE_URL,
    RANGE_BYTES,
    RANGES_PER_BATCH,
    range_rows,
)
from infra.rl_data.sources import (
    NEMOTRON_ULTRA_REVISION,
    NEMOTRON_ULTRA_RL_DATASET,
    PreparedRow,
    nemotron_ultra_mopd_source,
)

MOPD_FILENAME = "mopd.jsonl"
MOPD_SIZE = 5_496_714_762
MAX_RANGES = 128
MAX_REQUEST_CHARACTERS = 16_000
ROUTE_COLUMN = "teacher_route"
_PLACEHOLDER_KEY = "_hf_question_placeholder"

# Hardcoded generator -> Snowball expert routes for the first multi-teacher smoke. Only
# generators that occur in the blend and whose verifier runs in-process are listed. The
# judge-backed generators (math_with_judge, abstention, multichallenge, jailbreak, genrm),
# the sandbox-backed ones (ns_tools, lean) and the Harbor-backed SWE pivot rows need services
# this smoke does not provision, so they stay out until the launch host carries a judge key
# and task archives.
TEACHER_ROUTES: dict[str, str] = {
    "reasoning_gym_simple_agent": "math",
    "nvarc_inductive_simple_agent": "math",
    "nvarc_transductive_simple_agent": "math",
    "code_gen_simple_agent": "swe",
    "toolcall_schema_single_step_tool_use_with_argument_comparison_agent": "swe",
    "instruction_following_simple_agent": "terminal",
    "structured_outputs_simple_agent": "terminal",
    "structured_outputs_v3_simple_agent": "terminal",
    "mcqa_simple_agent": "terminal",
    "calendar_simple_agent": "terminal",
    "citation_format_simple_agent": "terminal",
    "freeform_formatting_simple_agent": "terminal",
    "rdkit_chemistry_agent": "terminal",
}


def routed_agent(row: dict[str, Any]) -> str | None:
    """Return the routed generator of a raw blend row, or None when the row is excluded."""
    agent_ref = row.get("agent_ref")
    agent = agent_ref.get("name") if isinstance(agent_ref, dict) else None
    if agent not in TEACHER_ROUTES or row.get(_PLACEHOLDER_KEY) is not None:
        return None
    request = row.get("responses_create_params")
    if not isinstance(request, dict) or len(json.dumps(request, ensure_ascii=False)) > MAX_REQUEST_CHARACTERS:
        return None
    return agent


def sample_raw_subset_rows(
    *,
    seed: int,
    rows_per_agent: int,
    revision: str = NEMOTRON_ULTRA_REVISION,
    max_ranges: int = MAX_RANGES,
) -> dict[str, list[dict[str, Any]]]:
    """Collect up to ``rows_per_agent`` raw rows per routed generator with bounded transfer."""
    if rows_per_agent <= 0 or max_ranges <= 0:
        raise ValueError("rows_per_agent and max_ranges must be positive")
    rng = random.Random(f"{seed}:mopd:{revision}")
    offsets = [rng.randrange(0, MOPD_SIZE - RANGE_BYTES) for _ in range(max_ranges)]
    url = HUGGING_FACE_RESOLVE_URL.format(revision=revision, filename=MOPD_FILENAME)
    candidates: dict[str, list[dict[str, Any]]] = {agent: [] for agent in TEACHER_ROUTES}
    for batch_start in range(0, len(offsets), RANGES_PER_BATCH):
        batch = offsets[batch_start : batch_start + RANGES_PER_BATCH]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as executor:
            for rows in executor.map(lambda offset: range_rows(url, offset), batch):
                for row in rows:
                    agent = routed_agent(row)
                    if agent is not None and len(candidates[agent]) < rows_per_agent:
                        candidates[agent].append(row)
        if all(len(rows) >= rows_per_agent for rows in candidates.values()):
            break
    return candidates


def prepare_subset_rows(candidates: dict[str, list[dict[str, Any]]]) -> list[PreparedRow]:
    """Turn raw rows into SkyRL rows carrying the hardcoded ``teacher_route`` column."""
    from skyrl_gym import get_data_contract

    source = nemotron_ultra_mopd_source()
    contract = get_data_contract(source.env_id)
    prepared: list[PreparedRow] = []
    for agent in sorted(candidates):
        for row in candidates[agent]:
            item = source.prepare_row(row, len(prepared), contract)
            item[ROUTE_COLUMN] = TEACHER_ROUTES[agent]
            prepared.append(item)
    return prepared


def write_mopd_subset(
    output_path: Path,
    *,
    seed: int,
    rows_per_agent: int,
    revision: str = NEMOTRON_ULTRA_REVISION,
    max_ranges: int = MAX_RANGES,
) -> dict[str, Any]:
    """Write a routed parquet subset of the MOPD blend and return its manifest."""
    import datasets

    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite MOPD subset: {output_path}")
    candidates = sample_raw_subset_rows(
        seed=seed, rows_per_agent=rows_per_agent, revision=revision, max_ranges=max_ranges
    )
    found_routes = {TEACHER_ROUTES[agent] for agent, rows in candidates.items() if rows}
    missing = sorted(set(TEACHER_ROUTES.values()) - found_routes)
    if missing:
        raise RuntimeError(f"No rows found for routes {missing} after {max_ranges} MiB of bounded reads")
    prepared = prepare_subset_rows(candidates)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    datasets.Dataset.from_list(prepared).to_parquet(str(output_path))
    routes = Counter(row[ROUTE_COLUMN] for row in prepared)
    return {
        "dataset": NEMOTRON_ULTRA_RL_DATASET,
        "revision": revision,
        "blend": "mopd",
        "seed": seed,
        "rows": len(prepared),
        "rows_per_agent": {agent: len(rows) for agent, rows in sorted(candidates.items())},
        "rows_per_route": dict(sorted(routes.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New parquet path for the routed subset.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rows-per-agent", type=int, required=True)
    parser.add_argument("--max-ranges", type=int, default=MAX_RANGES, help="Bounded 1 MiB range reads.")
    args = parser.parse_args()
    manifest = write_mopd_subset(
        args.output, seed=args.seed, rows_per_agent=args.rows_per_agent, max_ranges=args.max_ranges
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
