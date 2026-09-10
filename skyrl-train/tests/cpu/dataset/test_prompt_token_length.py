from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from skyrl_train.dataset import PromptDataset


def test_actual_chat_token_lengths_filter_parquet(tmp_path):
    backend = Tokenizer(WordLevel({"[UNK]": 0, "word": 1, "answer": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.chat_template = (
        "{% for m in messages %}{{m['content']}} {% endfor %}"
        "{% if add_generation_prompt %}answer{% endif %}"
    )
    prompts = [[{"role": "user", "content": " ".join(["word"] * n)}] for n in (2, 3, 4, 100)]
    path = tmp_path / "prompts.parquet"
    Dataset.from_dict({"prompt": prompts, "source_id": ["short", "boundary", "long", "very_long"]}).to_parquet(
        str(path)
    )

    actual = PromptDataset(str(path), tokenizer, max_prompt_length=4, num_workers=0)

    assert [actual[i][2]["source_id"] for i in range(len(actual))] == ["short", "boundary"]
    assert [actual[i][3] for i in range(len(actual))] == ["0", "1"]
    assert [tokenizer.apply_chat_template(actual[i][0], add_generation_prompt=True, return_dict=False)
            for i in range(len(actual))] == [[1, 1, 2], [1, 1, 1, 2]]
