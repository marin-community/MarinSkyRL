import datasets
from loguru import logger
import os
from collections.abc import Mapping
from typing import List
from transformers import PreTrainedTokenizerBase


def read_prompt_files(sources: list[str]) -> datasets.Dataset:
    """Load local parquet/JSON files or explicitly selected Hugging Face splits."""
    loaded = []
    for source in sources:
        ext = os.path.splitext(source)[-1].lower()
        if ext == ".parquet":
            data = datasets.load_dataset("parquet", data_files=source, keep_in_memory=False)["train"]
        elif ext in [".json", ".jsonl"]:
            data = datasets.load_dataset("json", data_files=source, keep_in_memory=False)["train"]
        else:
            name, separator, split = source.partition(":")
            split = split if separator else "train"
            available = datasets.load_dataset(path=name, keep_in_memory=False)
            if split not in available:
                raise ValueError(
                    f"Split `{split}` not found in dataset `{name}`. Configured split was `{split}` and default is `train`"
                )
            data = available[split]
        loaded.append(data)
    return datasets.concatenate_datasets(loaded)


class PromptDataset:
    def __init__(
        self,
        datasets: str | List[str],
        tokenizer: PreTrainedTokenizerBase,
        max_prompt_length: int,
        num_workers: int = 8,
        prompt_key: str = "prompt",
        env_class_key: str = "env_class",
    ):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.prompt_key = prompt_key
        self.env_class_key = env_class_key
        self.num_workers = num_workers

        self.datasets = datasets
        if isinstance(self.datasets, str):
            self.datasets = [self.datasets]

        self._read_files_and_tokenize()

    def _read_files_and_tokenize(self):
        self.dataframe = read_prompt_files(self.datasets)

        logger.info(f"Total dataset size: {len(self.dataframe)}")

        # filter out too long prompts
        tokenizer = self.tokenizer
        prompt_key = self.prompt_key
        self.dataframe = self.dataframe.filter(
            lambda doc: _prompt_token_count(tokenizer, doc[prompt_key]) <= self.max_prompt_length,
            num_proc=self.num_workers,
            desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
        )

        logger.info(f"Filtered dataset size: {len(self.dataframe)}")

    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]
        messages = row_dict.pop(self.prompt_key)
        env_class = row_dict.pop(self.env_class_key, None)

        extra = {key: value for key, value in row_dict.items() if key != self.prompt_key and key != self.env_class_key}

        return messages, env_class, extra, self.uid(item)

    def uid(self, index: int) -> str:
        return str(index)

    def collate_fn(self, item_list):
        all_inputs = []
        for prompt, env_class, env_extras, item_uids in item_list:
            all_inputs.append({"prompt": prompt, "env_class": env_class, "env_extras": env_extras, "uid": item_uids})
        return all_inputs

    def __len__(self):
        return len(self.dataframe)


def _prompt_token_count(tokenizer: PreTrainedTokenizerBase, prompt) -> int:
    """Count token IDs, including with Transformers 5.x ``BatchEncoding`` output."""
    encoded = tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids")
    if encoded is None:
        raise TypeError("tokenizer chat template returned no input_ids")
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, list) and len(encoded) == 1 and isinstance(encoded[0], list):
        encoded = encoded[0]
    if isinstance(encoded, str):
        return len(encoded)
    if not isinstance(encoded, list):
        raise TypeError(f"tokenizer chat template returned unsupported token IDs: {type(encoded).__name__}")
    return len(encoded)
