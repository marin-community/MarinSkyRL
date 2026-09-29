"""Count training exposure and enforce an exact loss-token budget."""

from dataclasses import asdict, dataclass

import torch


@dataclass
class TrainingTokens:
    loss: int = 0
    response: int = 0
    input: int = 0
    sequences: int = 0

    def limit_loss(self, batch, budget: int | None) -> None:
        """Mask only surplus loss positions in the final batch; keep complete rollouts."""
        if budget is None:
            return
        remaining = budget - self.loss
        if remaining <= 0:
            raise ValueError("Training loss-token budget is already exhausted")
        mask = batch["loss_mask"]
        real_rows = batch.batch_size - batch.metadata.get("pad_size", 0)
        active = mask[:real_rows].reshape(-1)
        positions = torch.cumsum(active.to(torch.int64), dim=0)
        kept = active * (positions <= remaining).to(active.dtype)
        mask[:real_rows] = kept.reshape(mask[:real_rows].shape)

    def consume(self, batch) -> dict[str, int]:
        """Record one successful optimizer update, excluding data-parallel padding."""
        real_rows = batch.batch_size - batch.metadata.get("pad_size", 0)
        step_loss = int(batch["loss_mask"][:real_rows].sum().item())
        self.loss += step_loss
        self.response += int(batch["response_mask"][:real_rows].sum().item())
        step_input = int(batch["attention_mask"][:real_rows].sum().item())
        self.input += step_input
        self.sequences += real_rows
        return {f"consumed/{key}_total": value for key, value in asdict(self).items()} | {
            "consumed/prompt_total": self.input - self.response,
            "consumed/loss_step": step_loss,
            "consumed/input_step": step_input,
        }


def loss_token_budget(config) -> int | None:
    budget = config.get("loss_token_budget")
    if budget is not None:
        if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
            raise ValueError("trainer.loss_token_budget must be a positive integer")
        if config.update_epochs_per_batch != 1:
            raise ValueError("A loss-token budget requires exactly one update epoch per batch")
    return budget
