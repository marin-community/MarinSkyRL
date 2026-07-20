"""Build the SkyRL-train GRPO parquet for the Delphi math-RLVR pilot (issue #6279 D1).

Normalizes ``allenai/RLVR-MATH`` to the SkyRL ``main_base``/``aime`` contract — each row
is ``{prompt: [chat messages], env_class: "aime", reward_model: {ground_truth: <answer>}}``
with the shared boxed instruction (``Answer: \\boxed{ANSWER}``), byte-identical to the
held-out MATH-500 grader. The train pool is seeded-subsampled to a fixed N (default 500,
seed 42) so step count is controlled; validation is always MATH-500 test.

Ported from the Delphi run-of-record ``main_rl_evals/rl_dataset_prep.py`` (RLVR-MATH arm),
using MarinSkyRL's own ``skyrl_gym.envs.aime`` answer-normalizer instead of a file-path
import. This is a ONE-TIME host/preflight step; write the parquet to a shared URI and pass
it via ``--train_data`` (the config's ``data.kind: parquet`` routes it past the
terminal_bench task extractor).

Usage::

    python -m cloud.iris.delphi_math_dataset \
        --output_dir s3://marin-us-east-02a/iris/rl-data/delphi-rlvr-math \
        --subsample_n 500 --seed 42
"""

from __future__ import annotations

import argparse
import os

import datasets
from skyrl_gym.envs.aime.utils import normalize_final_answer

# The shared boxed contract — IDENTICAL to the Delphi run-of-record and the held-out eval.
INSTRUCTION = (
    " Please reason step by step. At the very end, output your final answer on its "
    "own line in the exact format: 'Answer: \\boxed{ANSWER}'."
)

TRAIN_DATASET = "allenai/RLVR-MATH"
VAL_DATASET = "HuggingFaceH4/MATH-500"  # held-out math eval, shared by the math cells


def _user_content(messages: list[dict]) -> str:
    """Last user-turn content from a chat-messages list."""
    for m in reversed(messages):
        if m.get("role") == "user":
            return m["content"]
    return messages[-1]["content"]


def _strip_rlvr_math_fewshot(text: str) -> str:
    """RLVR-MATH wraps every problem in a fixed 4-shot preamble (four worked
    'Question:...\\nAnswer:...' exemplars, then the real problem as a final 'Question:'
    block with no trailing 'Answer:'). The shared preamble is identical across all 7,500
    rows and pushes every prompt past 512 tokens, so SkyRL's max_prompt_length filter
    would drop every row. Keep only the last 'Question:' block (the real problem)."""
    marker = "Question:"
    idx = text.rfind(marker)
    body = text[idx + len(marker) :] if idx != -1 else text
    return body.strip()


def _math_row(problem: str, ground_truth: str | None, source: str, idx: int) -> dict:
    return {
        "data_source": source,
        "prompt": [{"role": "user", "content": problem + INSTRUCTION}],
        "env_class": "aime",
        "reward_model": {"ground_truth": ground_truth if ground_truth is not None else ""},
        "extra_info": {"split": "train", "index": idx},
    }


def build_rlvr_math() -> datasets.Dataset:
    ds = datasets.load_dataset(TRAIN_DATASET, split="train")

    def _map(ex, i):
        return _math_row(
            _strip_rlvr_math_fewshot(_user_content(ex["messages"])),
            normalize_final_answer(str(ex["ground_truth"])),
            TRAIN_DATASET,
            i,
        )

    return ds.map(_map, with_indices=True, remove_columns=ds.column_names)


def build_math500(split: str = "test") -> datasets.Dataset:
    ds = datasets.load_dataset(VAL_DATASET, split=split)

    def _map(ex, i):
        return _math_row(ex["problem"], normalize_final_answer(ex["answer"]), VAL_DATASET, i)

    return ds.map(_map, with_indices=True, remove_columns=ds.column_names)


def seeded_subsample(ds: datasets.Dataset, n: int | None, seed: int) -> datasets.Dataset:
    """Deterministic random subsample to n rows (all rows if the pool is <= n)."""
    if n is None or n >= len(ds):
        return ds
    return ds.shuffle(seed=seed).select(range(n))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output_dir", required=True, help="Dir/URI for train.parquet + validation.parquet.")
    ap.add_argument("--subsample_n", type=int, default=500, help="Common prompt count (default 500 = MATH-500 floor).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train = seeded_subsample(build_rlvr_math(), args.subsample_n, args.seed)
    val = build_math500("test")

    os.makedirs(args.output_dir, exist_ok=True) if "://" not in args.output_dir else None
    train_path = os.path.join(args.output_dir, "train.parquet")
    val_path = os.path.join(args.output_dir, "validation.parquet")
    train.to_parquet(train_path)
    val.to_parquet(val_path)

    print(f"[rlvr_math] train rows (subsampled to <= {args.subsample_n}, seed {args.seed}): {len(train)} -> {train_path}")
    print(f"[rlvr_math] val rows (MATH-500 test): {len(val)} -> {val_path}")
    print("sample train prompt:", train[0]["prompt"][0]["content"][:200])
    print("sample train gt:", repr(train[0]["reward_model"]["ground_truth"]))


if __name__ == "__main__":
    main()
