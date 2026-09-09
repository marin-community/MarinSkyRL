"""Optional final GRPO truncation treatment with per-response audit values."""

import math

import torch


def cap_truncated_advantages(data, cap: float):
    if not math.isfinite(cap) or cap >= 0:
        raise ValueError("Truncated advantage cap must be finite and negative")
    advantages = data["advantages"]
    selected = data["loss_mask"].bool()
    truncated = data["non_agentic_truncated"].bool()
    if truncated.shape != (advantages.shape[0],) or selected.shape != advantages.shape:
        raise ValueError("Truncation flags and policy actions must align")
    if not torch.isfinite(advantages[selected]).all():
        raise ValueError("Non-finite GRPO advantage before truncation treatment")
    result = advantages.clone()
    override = truncated[:, None] & selected
    result[override] = torch.clamp(advantages[override], max=cap)
    records = []
    for index in range(len(advantages)):
        before = advantages[index][selected[index]]
        after = result[index][selected[index]]
        # This protocol is explicitly sequence-level GRPO, not token-level GAE
        # or an implicit composition with loop credit.
        if before.numel() and not torch.equal(before, before[0].expand_as(before)):
            raise ValueError("Truncation protocol requires constant per-response GRPO advantages")
        records.append(
            {
                "protocol": "negative-truncated-grpo-v1",
                "truncated": bool(truncated[index]),
                "cap": cap,
                "policy_action_count": before.numel(),
                "pre_override": before[0].item() if before.numel() else None,
                "post_override": after[0].item() if after.numel() else None,
            }
        )
    data["advantages"] = result
    data.metadata["non_agentic_advantage_override"] = records
    return data
