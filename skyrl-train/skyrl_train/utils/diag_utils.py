"""Opt-in per-group GRPO rollout diagnostics and decoded train-rollout dumps.

Pure-Python helpers (no torch) wired into ``RayPPOTrainer.postprocess_trajectory_batch``
behind ``trainer.diag_group_metrics`` / ``trainer.dump_train_rollouts`` (both default
``false``). The trainer wraps the call site in a non-fatal guard: diagnostics must never
be able to crash or stall training.

The per-group metrics answer the question the batch-level ``reward/*`` and ``generate/*``
metrics cannot: how many prompt groups actually carry a GRPO learning signal. A group whose
trajectories all share the same outcome contributes zero advantage, so a batch can have a
healthy mean reward while most groups are all-wrong or all-correct and teach nothing.
"""

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Union

DIAG_METRIC_PREFIX = "diag"

TRUNCATED_STOP_REASON = "length"
DECLARED_DONE_STOP_REASON = "task_complete"  # harbor Terminus-2: the agent declared the task complete


def scalar_reward(reward: Union[float, List[float]]) -> float:
    """Per-trajectory optimization reward, mirroring ``get_rollout_metrics``.

    ``trajectory_batch["rewards"]`` holds either response-level scalars or token-level
    lists; token-level rewards are summed over the response.
    """
    if isinstance(reward, list):
        return float(sum(reward))
    return float(reward)


def _percentile(sorted_vals: List[float], q: float) -> float:
    """Nearest-rank percentile over an ascending-sorted list; ``0.0`` on empty input."""
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return float(sorted_vals[idx])


def compute_group_diagnostics(
    rewards: Sequence[Union[float, List[float]]],
    successes: Sequence[bool],
    uids: Sequence[Any],
    response_lengths: Sequence[int],
    stop_reasons: Optional[Sequence[Optional[str]]],
    n_samples_per_prompt: int,
    loss_masks: Optional[Sequence[Sequence[int]]] = None,
) -> Dict[str, float]:
    """Compute per-group GRPO learnability diagnostics plus a few batch tail statistics.

    Args:
        rewards: optimization rewards, one per trajectory (scalar or token-level).
        successes: task-success predicate per trajectory. Callers should derive it the same
            way ``pass@n`` does (unshaped outcome reward ``> 0``) so reward shaping cannot
            move the learnability metrics.
        uids: prompt id per trajectory; trajectories sharing a uid form one group.
        response_lengths: response token count per trajectory.
        stop_reasons: generator stop reason per trajectory (``None`` if unavailable).
        n_samples_per_prompt: configured group size ``G``.

    Returns ``{}`` on empty input. Emitted keys (all under ``diag/``):

    * ``num_groups``; ``frac_groups_all_wrong`` / ``frac_groups_mixed`` /
      ``frac_groups_all_correct`` over all groups, by ``successes``;
    * ``frac_groups_zero_reward_std``: groups whose optimization rewards are all equal,
      i.e. groups that yield zero GRPO advantage regardless of the success predicate;
    * ``group_reward_std_mean``: mean within-group population std of the optimization reward;
    * ``phat_frac_{k}_of_{G}`` for ``k = 0..G``: empirical pass-rate histogram, counting
      only full groups of exactly ``G`` trajectories so partial groups cannot skew it;
    * ``out_tok_p50`` / ``out_tok_p99``: response-length percentiles (``generate/*``
      already reports min/max/mean/std);
    * ``truncated_fraction``: trajectories with ``stop_reason == "length"``.
    """
    n = len(rewards)
    if n == 0 or len(uids) != n or len(successes) != n or len(response_lengths) != n:
        return {}

    groups: Dict[Any, List[int]] = {}
    for i, uid in enumerate(uids):
        groups.setdefault(uid, []).append(i)

    num_groups = len(groups)
    all_wrong = mixed = all_correct = zero_std = 0
    std_sum = 0.0
    full_group_correct_counts: List[int] = []
    g = int(n_samples_per_prompt)
    for members in groups.values():
        num_correct = sum(1 for i in members if successes[i])
        if num_correct == 0:
            all_wrong += 1
        elif num_correct == len(members):
            all_correct += 1
        else:
            mixed += 1

        group_rewards = [scalar_reward(rewards[i]) for i in members]
        mean = sum(group_rewards) / len(group_rewards)
        std = (sum((r - mean) ** 2 for r in group_rewards) / len(group_rewards)) ** 0.5
        std_sum += std
        if std == 0.0:
            zero_std += 1

        if len(members) == g:
            full_group_correct_counts.append(num_correct)

    p = DIAG_METRIC_PREFIX
    metrics: Dict[str, float] = {
        f"{p}/num_groups": float(num_groups),
        f"{p}/frac_groups_all_wrong": all_wrong / num_groups,
        f"{p}/frac_groups_mixed": mixed / num_groups,
        f"{p}/frac_groups_all_correct": all_correct / num_groups,
        f"{p}/frac_groups_zero_reward_std": zero_std / num_groups,
        f"{p}/group_reward_std_mean": std_sum / num_groups,
    }

    if g >= 1:
        num_full = len(full_group_correct_counts)
        for k in range(g + 1):
            metrics[f"{p}/phat_frac_{k}_of_{g}"] = full_group_correct_counts.count(k) / num_full if num_full else 0.0

    sorted_lens = sorted(float(length) for length in response_lengths)
    metrics[f"{p}/out_tok_p50"] = _percentile(sorted_lens, 0.50)
    metrics[f"{p}/out_tok_p99"] = _percentile(sorted_lens, 0.99)

    if stop_reasons:
        metrics[f"{p}/truncated_fraction"] = sum(1 for s in stop_reasons if s == TRUNCATED_STOP_REASON) / len(
            stop_reasons
        )
        # Reward decomposition (2026-09-05): mean success = f * q with f the share of
        # trajectories that declared task_complete and q the success rate among them. On the
        # Snowball band arms f rose 3-5x while q fell by a third; mean reward hid it.
        declared = [i for i, s in enumerate(stop_reasons) if s == DECLARED_DONE_STOP_REASON]
        metrics[f"{p}/declared_done_fraction"] = len(declared) / len(stop_reasons)
        metrics[f"{p}/reward_given_done"] = (
            sum(1 for i in declared if successes[i]) / len(declared) if declared else 0.0
        )
        if loss_masks is not None:
            # generator.mask_length_stops: length-stopped samples whose loss mask was zeroed.
            masked = sum(
                1
                for s, m in zip(stop_reasons, loss_masks)
                if s == TRUNCATED_STOP_REASON and not any(m)
            )
            metrics[f"{p}/length_stop_masked_fraction"] = masked / len(stop_reasons)
    else:
        metrics[f"{p}/truncated_fraction"] = 0.0

    return metrics


def dump_train_rollouts(
    trajectory_batch: Dict[str, Any],
    uids: Sequence[Any],
    tokenizer,
    export_path: str,
    global_step: int,
) -> str:
    """Write decoded train rollouts to ``<export_path>/dumped_data/global_step_{N}_train_rollouts.jsonl``.

    One JSON line per trajectory: decoded prompt and response text, the scalar optimization
    reward (see ``scalar_reward``), the unshaped outcome reward when the batch carries one,
    stop reason, response token count, and ``is_last_step`` for step-wise training. Written
    through a temp file + ``os.replace`` so a partially written dump is never observed.

    Must run BEFORE the trainer converts ``trajectory_batch["rewards"]`` to per-token form if
    the response-level values are to be preserved (the extraction handles both shapes).
    Returns the final path. Lives next to ``RayPPOTrainer.dump_data``'s pickles.
    """
    out_dir = os.path.join(export_path, "dumped_data")
    os.makedirs(out_dir, exist_ok=True)
    final_path = os.path.join(out_dir, f"global_step_{global_step}_train_rollouts.jsonl")
    tmp_path = final_path + ".tmp"

    prompt_token_ids = trajectory_batch["prompt_token_ids"]
    response_ids = trajectory_batch["response_ids"]
    rewards = trajectory_batch["rewards"]
    n = len(response_ids)

    def _column(key: str) -> List[Any]:
        values = trajectory_batch.get(key)
        return list(values) if values is not None and len(values) == n else [None] * n

    unshaped_rewards = _column("unshaped_rewards")
    stop_reasons = _column("stop_reasons")
    is_last_step = _column("is_last_step")
    trajectory_ids = _column("trajectory_ids")

    with open(tmp_path, "w", encoding="utf-8") as f:
        for i in range(n):
            trajectory_id = trajectory_ids[i]
            record = {
                "step": global_step,
                "uid": uids[i] if i < len(uids) else None,
                "trajectory_id": (
                    trajectory_id.to_string()
                    if hasattr(trajectory_id, "to_string")
                    else (str(trajectory_id) if trajectory_id is not None else None)
                ),
                "prompt": tokenizer.decode(prompt_token_ids[i], skip_special_tokens=True),
                "response": tokenizer.decode(response_ids[i], skip_special_tokens=True),
                "reward": scalar_reward(rewards[i]),
                "unshaped_reward": (float(unshaped_rewards[i]) if unshaped_rewards[i] is not None else None),
                "stop_reason": stop_reasons[i],
                "response_len": len(response_ids[i]),
                "is_last_step": is_last_step[i],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp_path, final_path)
    return final_path
