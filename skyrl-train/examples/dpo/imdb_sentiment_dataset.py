"""Build the DPO paper's IMDb sentiment preference pairs (arXiv:2305.18290, Appendix C.1).

Samples K completions per IMDb prefix from an SFT'd GPT-2, scores them with the paper's
sentiment classifier, and writes best/worst pairs per prefix: prompt (messages), chosen,
rejected. Pairs are separable preference data for ``environment.env_class=preference_pair``.

Usage (GPU recommended; CPU works but samples slowly)::

    uv run examples/dpo/imdb_sentiment_dataset.py --output_dir "$HOME/data/imdb-dpo" \
        --num_prompts 25000 --samples_per_prompt 4
"""

import argparse
from pathlib import Path

import datasets
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

SFT_MODEL = "lvwerra/gpt2-imdb"
CLASSIFIER = "siebert/sentiment-roberta-large-english"
PROMPT_TOKENS = (2, 8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sft_model", default=SFT_MODEL)
    parser.add_argument("--classifier", default=CLASSIFIER)
    parser.add_argument("--num_prompts", type=int, default=25000)
    parser.add_argument("--samples_per_prompt", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_tokenizer = AutoTokenizer.from_pretrained(args.sft_model)
    policy_tokenizer.pad_token = policy_tokenizer.eos_token
    policy = AutoModelForCausalLM.from_pretrained(args.sft_model).to(device).eval()
    label_tokenizer = AutoTokenizer.from_pretrained(args.classifier)
    classifier = AutoModelForSequenceClassification.from_pretrained(args.classifier).to(device).eval()

    imdb = datasets.load_dataset("stanfordnlp/imdb")["train"].shuffle(seed=args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    prefixes = []
    for review in imdb:
        tokens = policy_tokenizer(review["text"])["input_ids"]
        if len(tokens) <= PROMPT_TOKENS[1]:
            continue
        # The paper conditions on a review prefix truncated to 2-8 tokens, not on
        # reviews that are naturally that short (effectively none are).
        length = int(torch.randint(PROMPT_TOKENS[0], PROMPT_TOKENS[1] + 1, (1,), generator=generator))
        prefixes.append(policy_tokenizer.decode(tokens[:length]))
        if len(prefixes) >= args.num_prompts:
            break

    pairs = []
    for start in range(0, len(prefixes), args.batch_size):
        batch = prefixes[start : start + args.batch_size]
        encoded = policy_tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=PROMPT_TOKENS[1]
        ).to(device)
        with torch.no_grad():
            sampled = policy.generate(
                **encoded,
                do_sample=True,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=args.samples_per_prompt,
                pad_token_id=policy_tokenizer.pad_token_id,
            )
        completions = policy_tokenizer.batch_decode(
            sampled[:, encoded["input_ids"].shape[1] :], skip_special_tokens=True
        )
        for index in range(len(batch)):
            group = completions[index * args.samples_per_prompt : (index + 1) * args.samples_per_prompt]
            encoded_labels = label_tokenizer(group, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                scores = torch.softmax(classifier(**encoded_labels).logits, dim=-1)[:, 1]
            best, worst = int(scores.argmax()), int(scores.argmin())
            if best == worst:
                continue
            pairs.append((batch[index], group[best], group[worst]))
        if (start // args.batch_size) % 10 == 0:
            print(f"{start + len(batch)}/{len(prefixes)} prefixes sampled")

    # IMDb continuations are raw text, not chat turns: the prompt message carries the prefix
    # verbatim so the chat template renders it exactly.
    rows = {
        "prompt": [[{"role": "user", "content": prefix}] for prefix, _, _ in pairs],
        "chosen": [chosen for _, chosen, _ in pairs],
        "rejected": [rejected for _, _, rejected in pairs],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = datasets.Dataset.from_dict(rows)
    out.to_parquet(str(args.output_dir / "train.parquet"))
    print(f"wrote {len(out)} preference pairs to {args.output_dir / 'train.parquet'}")


if __name__ == "__main__":
    main()
