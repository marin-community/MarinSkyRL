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
    written = 0
    for split, aliases in (("train_prefs", ("train_prefs", "train")), ("test_prefs", ("test_prefs", "test"))):
        available = next((name for name in aliases if name in loaded), None)
        if available is None:
            continue
        rows = {"prompt": [], "chosen": [], "rejected": []}
        for row in loaded[available]:
            chosen, rejected = row["chosen"], row["rejected"]
            if "prompt" in row:
                prompt = row["prompt"]
                if not isinstance(prompt, str) or len(prompt) > args.max_prompt_chars:
                    continue
                prompt_messages = [{"role": "user", "content": prompt}]
                # Message lists include the prompt turn; the completion is the assistant reply.
                chosen_text = chosen[-1]["content"] if isinstance(chosen, list) else chosen
                rejected_text = rejected[-1]["content"] if isinstance(rejected, list) else rejected
            else:
                # Chat-format rows carry the full conversation in chosen/rejected.
                if (
                    not isinstance(chosen, list)
                    or not isinstance(rejected, list)
                    or len(chosen) < 2
                    or len(rejected) < 2
                ):
                    continue
                if [m.get("content") for m in chosen[:-1]] != [m.get("content") for m in rejected[:-1]]:
                    continue
                if len(chosen[-2]["content"]) > args.max_prompt_chars:
                    continue
                prompt_messages = chosen[:-1]
                chosen_text = chosen[-1]["content"]
                rejected_text = rejected[-1]["content"]
            if not chosen_text or not rejected_text:
                continue
            rows["prompt"].append(prompt_messages)
            rows["chosen"].append(chosen_text)
            rows["rejected"].append(rejected_text)
        out = datasets.Dataset.from_dict(rows)
        out.to_parquet(str(args.output_dir / f"{split}.parquet"))
        print(f"wrote {len(out)} {split} rows to {args.output_dir / f'{split}.parquet'}")
        written += 1
    if written == 0:
        raise SystemExit(f"no usable splits found in {args.source} (available: {sorted(loaded)})")


if __name__ == "__main__":
    main()
