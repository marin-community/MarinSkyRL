"""Train and score a small CatCountCanary model with the shared reward and GRPO loss."""

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding

from skyrl_gym.envs.cat_count.reward import CatCountScore, cat_count_score
from skyrl_train.utils.advantage_estimators import GRPO_FLAT_REWARD_STD_TOLERANCE, compute_grpo_outcome_advantage
from skyrl_train.utils.policy_losses import ppo_policy_loss

PROMPT = "Reply with the word cat exactly {N} times, separated by single spaces. Nothing else."
TEMPLATE = PROMPT.replace("word cat", "word {W}")
OTHER_WORDS = ["dog", "bird", "fox", "cow", "pig", "owl", "bee", "ant"]
SPECIAL = ["<pad>", "<eos>", "<unk>", "<|user|>", "<|assistant|>"]
PROMPT_WORDS = sorted(set(PROMPT.replace("{N}", "").replace(",", "").replace(".", "").split()))
VOCAB = SPECIAL + sorted(set(PROMPT_WORDS) | set(OTHER_WORDS)) + [",", "."] + [str(d) for d in range(10)]
CHAT_TEMPLATE = (
    "{% for m in messages %}<|user|> {{ m['content'] }} {% endfor %}"
    "{% if add_generation_prompt %}<|assistant|> {% endif %}"
)
HELD_OUT_N = [3, 5, 13, 16]
TRAIN_N = [n for n in range(1, 21) if n not in HELD_OUT_N]
GROUP_SIZE = 8
MAX_NEW_TOKENS = 64
DEFAULT_CKPT = Path.home() / ".cache/oa/cat-count-canary/base-530k"


class RolloutMode(StrEnum):
    GREEDY = "greedy"
    SAMPLED = "sampled"


@dataclass(frozen=True)
class Rollout:
    encoding: BatchEncoding
    prompt_length: int
    sequences: torch.Tensor
    response_mask: torch.Tensor
    scores: list[CatCountScore]
    items: list[tuple[str, int]]


def build_tokenizer() -> PreTrainedTokenizerFast:
    tk = Tokenizer(models.WordLevel({w: i for i, w in enumerate(VOCAB)}, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation(), pre_tokenizers.Digits(individual_digits=True)]
    )
    tk.decoder = decoders.WordPiece(prefix="##")
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
        additional_special_tokens=["<|user|>", "<|assistant|>"],
    )
    tok.chat_template = CHAT_TEMPLATE
    tok.padding_side = "left"
    return tok


def prompt_text(tok, n: int, word: str = "cat") -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": TEMPLATE.format(W=word, N=n)}], tokenize=False, add_generation_prompt=True
    )


def pretrain(args) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    tok = build_tokenizer()
    config = LlamaConfig(
        vocab_size=len(VOCAB),
        hidden_size=args.width,
        intermediate_size=4 * args.width,
        num_hidden_layers=args.layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        bos_token_id=None,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(config)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    start = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        batch = []
        for _ in range(32):
            if random.random() < 1 / 9:
                word, n = "cat", 1
            else:
                word, n = random.choice(OTHER_WORDS), random.randint(1, 30)
            p = tok(prompt_text(tok, n, word), add_special_tokens=False).input_ids
            c = tok(" ".join([word] * n), add_special_tokens=False).input_ids + [tok.eos_token_id]
            batch.append((p, c))
        length = max(len(p) + len(c) for p, c in batch)
        ids = torch.zeros(len(batch), length, dtype=torch.long)
        labels = torch.full((len(batch), length), -100)
        attn = torch.zeros(len(batch), length, dtype=torch.long)
        for i, (p, c) in enumerate(batch):
            seq = p + c
            ids[i, : len(seq)] = torch.tensor(seq)
            attn[i, : len(seq)] = 1
            labels[i, len(p) : len(seq)] = torch.tensor(c)
        loss = model(input_ids=ids, attention_mask=attn, labels=labels).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(json.dumps({"pretrain_step": step, "loss": round(loss.item(), 4)}), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    params = sum(p.numel() for p in model.parameters())
    print(
        json.dumps(
            {
                "saved": str(out),
                "params": params,
                "vocab": len(VOCAB),
                "final_loss": round(loss.item(), 4),
                "seconds": round(time.time() - start, 1),
            }
        ),
        flush=True,
    )


def rollout(model, tok, ns, mode: RolloutMode) -> Rollout:
    items = [(prompt_text(tok, n), n) for n in ns for _ in range(1 if mode is RolloutMode.GREEDY else GROUP_SIZE)]
    enc = tok([s for s, _ in items], return_tensors="pt", padding=True, add_special_tokens=False)
    prompt_len = enc.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            **enc,
            do_sample=mode is RolloutMode.SAMPLED,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    resp = out[:, prompt_len:]
    is_eos = resp == tok.eos_token_id
    mask = (~((is_eos.cumsum(1) - is_eos.long()) > 0)).float()  # keep tokens up to and including eos
    texts = tok.batch_decode(resp, skip_special_tokens=True)
    scores = [
        cat_count_score(t, n, stop_reason="stop" if is_eos[i].any() else "length")
        for i, (t, (_, n)) in enumerate(zip(texts, items))
    ]
    return Rollout(enc, prompt_len, out, mask, scores, items)


def evaluate(model, tok, step: int) -> dict:
    model.eval()
    result = {"step": step}
    for name, ns in (("train", TRAIN_N), ("heldout", HELD_OUT_N)):
        sample = rollout(model, tok, ns, RolloutMode.GREEDY)
        result[f"{name}_reward"] = round(sum(s.reward for s in sample.scores) / len(sample.scores), 3)
        result[f"{name}_exact"] = round(sum(s.exact for s in sample.scores) / len(sample.scores), 3)
        if name == "heldout":
            result["heldout_counts"] = {n: s.cat_unigram_count for s, (_, n) in zip(sample.scores, sample.items)}
    return result


def rl(args) -> int:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    tok = PreTrainedTokenizerFast.from_pretrained(args.ckpt)
    tok.padding_side = "left"
    model = LlamaForCausalLM.from_pretrained(args.ckpt)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    loss_cfg = OmegaConf.create(
        dict(
            policy_loss_type="regular",
            loss_reduction="token_mean",
            eps_clip_low=0.2,
            eps_clip_high=0.2,
            clip_ratio_c=3.0,
            use_tis=False,
            tis_imp_ratio_cap=2.0,
            max_seq_len=128,
        )
    )
    group_index = np.array([i // GROUP_SIZE for i in range(len(TRAIN_N) * GROUP_SIZE)])
    first = evaluate(model, tok, 0)
    print(json.dumps({"eval": first}), flush=True)
    start = time.time()
    for step in range(1, args.max_steps + 1):
        model.eval()
        sample = rollout(model, tok, TRAIN_N, RolloutMode.SAMPLED)
        rewards = torch.tensor([s.reward for s in sample.scores])
        token_rewards = torch.zeros_like(sample.response_mask)
        token_rewards[torch.arange(len(sample.items)), (sample.response_mask.sum(1).long() - 1).clamp(min=0)] = rewards
        advantages, _ = compute_grpo_outcome_advantage(token_rewards, sample.response_mask, group_index)
        if args.flip_advantage:
            advantages = -advantages
        model.train()
        attn = torch.cat([sample.encoding.attention_mask, sample.response_mask.long()], 1)
        logits = model(input_ids=sample.sequences, attention_mask=attn).logits[:, sample.prompt_length - 1 : -1].float()
        logprobs = (
            torch.log_softmax(logits, -1)
            .gather(-1, sample.sequences[:, sample.prompt_length :].unsqueeze(-1))
            .squeeze(-1)
        )
        loss, _ = ppo_policy_loss(logprobs, logprobs.detach(), advantages, loss_cfg, loss_mask=sample.response_mask)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.eval_every == 0:
            ev = evaluate(model, tok, step)
            ev["train_rollout_reward"] = round(rewards.mean().item(), 3)
            ev["zero_variance_groups"] = int(
                (rewards.view(len(TRAIN_N), GROUP_SIZE).std(1) <= GRPO_FLAT_REWARD_STD_TOLERANCE).sum()
            )
            print(json.dumps({"eval": ev}), flush=True)
            if ev["train_reward"] >= args.bar and ev["heldout_reward"] >= args.bar:
                verdict = {
                    "verdict": "PASS",
                    "step": step,
                    "rl_seconds": round(time.time() - start, 2),
                    "train_reward": [first["train_reward"], ev["train_reward"]],
                    "heldout_reward": [first["heldout_reward"], ev["heldout_reward"]],
                }
                print(json.dumps(verdict), flush=True)
                return 0
    print(
        json.dumps(
            {
                "verdict": "FAIL",
                "reason": f"eval rewards below {args.bar} after {args.max_steps} steps",
                "rl_seconds": round(time.time() - start, 2),
            }
        ),
        flush=True,
    )
    return 1


def main() -> int:
    torch.set_num_threads(1)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pretrain")
    p.add_argument("--out", default=str(DEFAULT_CKPT))
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("rl")
    r.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    r.add_argument("--lr", type=float, default=1e-4)
    r.add_argument("--max_steps", type=int, default=100)
    r.add_argument("--eval_every", type=int, default=5)
    r.add_argument("--bar", type=float, default=0.9)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--flip-advantage", action="store_true", help="negative control: reverse the GRPO learning signal")
    args = ap.parse_args()
    if args.cmd == "pretrain":
        pretrain(args)
        return 0
    return rl(args)


if __name__ == "__main__":
    sys.exit(main())
