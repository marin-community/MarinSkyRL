"""Compare the production PPO/TIS closure with the released JAX objective."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from skyrl_train.objective.losses import ppo_policy_loss
from skyrl_train.objective.objective import build_objective_micro_batch, compute_policy_objective
from skyrl_train.objective.reduction import step_counts
from skyrl_train.objective.score_centering import ppo_tis_score_centering_correction
from skyrl_train.utils.advantage_estimators import compute_grpo_outcome_advantage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, reference = np.load(args.input), np.load(args.reference)
    masks = torch.from_numpy(np.roll(source["mask"], -1, axis=1)).float()
    masks[:, -1] = 0
    rewards = torch.zeros_like(masks)
    rewards[:, 0] = 2 * torch.from_numpy(source["rewards"]) - 1
    advantages, _ = compute_grpo_outcome_advantage(
        rewards, masks, np.repeat(np.arange(2), 8), grpo_norm_by_std=False
    )
    np.testing.assert_array_equal(advantages.numpy(), reference["advantages"][:, None] * masks.numpy())
    config = OmegaConf.create({
        "eps_clip_low": 0.2, "eps_clip_high": 0.2, "loss_reduction": "token_mean",
        "use_kl_loss": False, "use_entropy_loss": False, "kl_loss_coef": 0,
    })
    labels = torch.from_numpy(np.roll(source["tokens"], -1, axis=1)).long()
    results = []
    for mismatch in (0.0, 2.5):
        q = torch.from_numpy(source["logits"] + mismatch * source["noise"]).log_softmax(-1)
        chosen_q = q.gather(-1, labels[..., None]).squeeze(-1)
        for width in (1, 3, 9):
            head_q, ids = q.topk(width, -1)
            for enabled in (False, True):
                name = f"m{mismatch}-k{width}-sc{int(enabled)}"
                logits = torch.from_numpy(source["logits"]).clone().requires_grad_()
                logp = logits.log_softmax(-1)
                chosen = logp.gather(-1, labels[..., None]).squeeze(-1)
                old = chosen.detach().clone()
                head = logp.gather(-1, ids)
                tis = (old - chosen_q).exp().clamp(max=2)
                correction = ppo_tis_score_centering_correction(
                    head, head.detach(), head_q, advantages, masks,
                    tis_cap=2, eps_clip_low=0.2, eps_clip_high=0.2,
                ) * enabled
                batch = build_objective_micro_batch(
                    action_log_probs=chosen, old_action_log_probs=old, base_action_log_probs=None,
                    advantages=advantages, loss_mask=masks, rollout_logprobs=chosen_q,
                    response_span_tags=None, token_entropy=torch.zeros_like(masks), think_token_weight=1,
                    teacher=None, correction_weights=tis, score_centering=correction,
                )
                counts = step_counts([masks], [masks], [torch.zeros_like(masks)], [advantages], 4, lambda x: x)
                objective = compute_policy_objective(
                    batch, loss=ppo_policy_loss, counts=counts, config=config, loss_scale=1, report_scale=1,
                )
                objective.optimization_loss.backward()
                expected_gradient = torch.from_numpy(reference[f"{name}-gradient"])
                delta = (logits.grad - expected_gradient).abs()
                torch.testing.assert_close(logits.grad, expected_gradient, atol=1e-7, rtol=1e-5)
                assert objective.metrics["ppo_clip_ratio"] == 0
                # REINFORCE logp and PPO ratio have different scalar origins at p=o.
                # The released modeled-tail scalar also retains detached E_p[log p].
                q_tail = (1 - head_q.exp().sum(-1)).clamp_min(1e-6)
                p_tail = (1 - head.exp().sum(-1)).clamp_min(1e-6)
                alpha = (2 * q_tail / p_tail).clamp(max=1)
                plogp = (logp.exp() * logp).sum(-1)
                scalar_offset = advantages * (tis * (1 - chosen) + enabled * alpha * plogp)
                aligned = objective.optimization_loss.detach() + (scalar_offset * masks).sum() / masks.sum()
                reference_value = float(reference[f"{name}-loss"])
                np.testing.assert_allclose(float(aligned), reference_value, atol=1e-7, rtol=1e-5, err_msg=name)
                results.append({
                    "case": name, "raw_skyrl_loss": float(objective.optimization_loss.detach()),
                    "reference_loss": reference_value, "skyrl_loss_after_scalar_offset": float(aligned),
                    "max_gradient_error": float(delta.max()), "mean_gradient_error": float(delta.mean()),
                    "ppo_clip_ratio": objective.metrics["ppo_clip_ratio"], "masked_tokens": int(masks.sum()),
                    "tis_capped_fraction": float((((old - chosen_q).exp() > 2) * masks).sum() / masks.sum()),
                })
    args.output.write_text(json.dumps({"scope": "FP32 p=o, actual +1/-1 group centering; masks; TIS2; natural head and modeled tail", "cases": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
