"""Convert UltraFeedback binarized preferences to the MarinSkyRL preference-pair parquet.

Rows carry the chat ``prompt`` plus ``chosen`` and ``rejected`` completion text, ready for
``environment.env_class=preference_pair`` with ``+algorithm_recipe=dpo``.

Usage::

    uv run examples/dpo/ultrafeedback_dataset.py --output_dir "$HOME/data/ultrafeedback"
"""

import argparse
from pathlib import Path

import datasets

SOURCE = "trl-lib/ultrafeedback_binarized"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--source", default=SOURCE)
    parser.add_argument("--max_prompt_chars", type=int, default=4096)
    args = parser.parse_args()

    loaded = datasets.load_dataset(args.source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train_prefs", "test_prefs"):
        if split not in loaded:
            continue
        rows = {"prompt": [], "chosen": [], "rejected": []}
        for row in loaded[split]:
            prompt = row["prompt"]
            if not isinstance(prompt, str) or len(prompt) > args.max_prompt_chars:
                continue
            chosen, rejected = row["chosen"], row["rejected"]
            # Message lists include the prompt turn; the completion is the assistant reply.
            chosen_text = chosen[-1]["content"] if isinstance(chosen, list) else chosen
            rejected_text = rejected[-1]["content"] if isinstance(rejected, list) else rejected
            if not chosen_text or not rejected_text:
                continue
            rows["prompt"].append([{"role": "user", "content": prompt}])
            rows["chosen"].append(chosen_text)
            rows["rejected"].append(rejected_text)
        out = datasets.Dataset.from_dict(rows)
        out.to_parquet(str(args.output_dir / f"{split}.parquet"))
        print(f"wrote {len(out)} {split} rows to {args.output_dir / f'{split}.parquet'}")


if __name__ == "__main__":
    main()
