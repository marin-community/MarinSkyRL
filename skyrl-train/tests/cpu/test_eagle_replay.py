"""Behavior tests for cached-conversation EAGLE replay."""

from skyrl_train.eagle_replay import replay_sequence_from_row, select_replay_sequences


class _Tokenizer:
    @staticmethod
    def encode(text: str, *, add_special_tokens: bool) -> list[int]:
        prefix = [1] if add_special_tokens else []
        return [*prefix, *(ord(character) for character in text)]

    @staticmethod
    def apply_chat_template(messages, *, tokenize: bool, add_generation_prompt: bool):
        assert tokenize
        tokens = [1]
        for message in messages:
            tokens.extend(ord(character) for character in f"<{message['role']}>{message['content']}")
        if add_generation_prompt:
            tokens.extend(ord(character) for character in "<assistant>")
        return tokens


def test_replay_sequence_uses_exact_tokens_without_retokenizing() -> None:
    sequence = replay_sequence_from_row(
        {
            "group_id": "rollout-1",
            "prompt_token_ids": [10, 11],
            "response_token_ids": [20, 21, 22],
        },
        _Tokenizer(),
    )

    assert sequence.token_ids == [10, 11, 20, 21, 22]
    assert sequence.loss_start == 2
    assert sequence.group_id == "rollout-1"


def test_replay_selection_enforces_global_tokens_and_group_limit() -> None:
    rows = iter(
        [
            {"group_id": "a", "prompt_token_ids": [1, 2], "response_token_ids": [3, 4]},
            {"group_id": "a", "prompt_token_ids": [1, 2], "response_token_ids": [5, 6]},
            {"group_id": "b", "prompt_token_ids": [7, 8], "response_token_ids": [9, 10]},
        ]
    )

    selection = select_replay_sequences(
        rows,
        _Tokenizer(),
        max_tokens=8,
        max_window_tokens=8,
        max_sequences_per_group=1,
    )

    assert [sequence.group_id for sequence in selection.sequences] == ["a", "b"]
    assert selection.charged_tokens == 8
    assert selection.skipped_rows == 1
