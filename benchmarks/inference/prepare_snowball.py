"""Freeze in-region Snowball model and outcome-independent curriculum inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import fsspec
import skyrl_gym
from omegaconf import DictConfig
from skyrl_train.dataset.dataset import PromptDataset
from skyrl_train.inference_engines.utils import hash_with_sha256
from transformers import AutoTokenizer

from cloud.iris.task_runtime import materialize_model_export

MODEL_URI = "s3://marin-us-east-02a/marin/exports/grug/june-67b-a2b-sft-s2-thinking/step-630/hf-bf16-vllm/"
POOL_URI = "s3://marin-us-east-02a/marin/documents/curriculum-rl-pool/2026.09.02.1/train.parquet"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/tmp/snowball-model"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--waves", type=int, default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    materialize_model_export(MODEL_URI, str(args.model), "june-67b-a2b-sft-s2-thinking@step-630")
    source = args.output / "train.parquet"
    with fsspec.open(POOL_URI, "rb") as remote:
        source.write_bytes(remote.read())
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dataset = PromptDataset(str(source), tokenizer, 1024)
    random_source = random.Random(17)
    accepted = []
    candidate_count = 0
    while len(accepted) < args.waves:
        candidate_count += 1
        indices = random_source.sample(range(len(dataset)), 64)
        # Select on the hash allocation only, before generation or outcome access.
        # This gives exactly 128 requests on node zero while retaining production
        # 64-prompt x 8-repetition grouping and hash routing across 32 DP engines.
        count = sum(hash_with_sha256(f"{uid}_{rep}") % 32 < 8 for uid in indices for rep in range(8))
        if count != 128:
            continue
        prompts = []
        for index in indices:
            prompt, env_class, extras, uid = dataset[index]
            env = skyrl_gym.make(env_class, env_config=DictConfig({}), extras=extras)
            history, _ = env.init(prompt)
            tokens = tokenizer.apply_chat_template(
                history, add_generation_prompt=True, tokenize=True, return_dict=False
            )
            if len(tokens) > 1024:
                raise ValueError(f"Templated row {uid} exceeds the production prompt budget")
            prompts.append(
                {
                    "group_id": uid,
                    "prompt_token_ids": tokens,
                    "env_class": env_class,
                    "extra_info": extras.get("extra_info", {}),
                }
            )
        accepted.append(prompts)
    corpus = {
        "model_uri": MODEL_URI,
        "pool_uri": POOL_URI,
        "pool_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "filtered_pool_rows": len(dataset),
        "selection_seed": 17,
        "candidate_batches": candidate_count,
        "selection": "uniform 64 rows; node-zero hash count equals 128",
        "samples_per_prompt": 8,
        "production_dp_engines": 32,
        "waves": accepted,
    }
    payload = json.dumps(corpus, sort_keys=True, separators=(",", ":"))
    (args.output / "corpus.json").write_text(payload + "\n")
    print(
        json.dumps(
            {
                "corpus_sha256": hashlib.sha256((payload + "\n").encode()).hexdigest(),
                "waves": len(accepted),
                "candidate_batches": candidate_count,
                "pool_rows": len(dataset),
                "env_counts": Counter(row["env_class"] for wave in accepted for row in wave),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
