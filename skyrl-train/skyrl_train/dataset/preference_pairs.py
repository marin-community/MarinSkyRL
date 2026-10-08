"""Preference-pair training data: prompt plus chosen and rejected completions."""

from typing import Any

from loguru import logger
from transformers import PreTrainedTokenizerBase

from skyrl_train.dataset.dataset import PromptDataset


def completion_text(value: Any, field: str) -> str:
    """Normalize a completion column to text (the assistant turn's content for message lists)."""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        last = value[-1]
        if isinstance(last, dict) and isinstance(last.get("content"), str):
            return last["content"]
    raise ValueError(f"preference-pair {field} completions must be text or message lists with assistant content")


class PreferencePairDataset(PromptDataset):
    """A prompt dataset that requires ``chosen``/``rejected`` columns and length-filters both.

    Both completions of a pair must fit the rollout budget; dropping an overlong pair at
    load time is the only legal drop, because a training group must return exactly two
    rows. Message-list completions normalize to their assistant text, and per-row
    ``env_class`` overrides are dropped so the configured ``preference_pair`` runner
    handles every row.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *, max_completion_length: int, **kwargs):
        self.max_completion_length = max_completion_length
        super().__init__(tokenizer=tokenizer, **kwargs)

    def _read_files_and_tokenize(self):
        super()._read_files_and_tokenize()
        for column in ("chosen", "rejected"):
            if column not in self.dataframe.column_names:
                raise ValueError(f"preference-pair training data requires a {column!r} column")
        if self.max_completion_length <= 0:
            raise ValueError("max_completion_length must be positive")
        if "env_class" in self.dataframe.column_names:
            self.dataframe = self.dataframe.remove_columns("env_class")

        tokenizer = self.tokenizer
        budget = self.max_completion_length

        def normalize(row):
            return {
                "chosen": completion_text(row["chosen"], "chosen"),
                "rejected": completion_text(row["rejected"], "rejected"),
            }

        def fits(row) -> bool:
            for column in ("chosen", "rejected"):
                if len(tokenizer(row[column], add_special_tokens=False)["input_ids"]) > budget:
                    return False
            return True

        self.dataframe = self.dataframe.map(normalize, num_proc=self.num_workers, desc="Normalizing completions")
        before = len(self.dataframe)
        self.dataframe = self.dataframe.filter(fits, num_proc=self.num_workers, desc="Filtering preference pairs")
        logger.info(
            "Preference-pair dataset kept {} of {} rows within the {}-token completion budget",
            len(self.dataframe),
            before,
            budget,
        )
