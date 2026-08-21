"""
uv run --group dev --extra cpu --isolated pytest tests/cpu/trajectory_runners/test_trajectory_processing.py
"""

import pytest
from types import SimpleNamespace

from skyrl_train.trajectory_runners.trajectory_processing import (
    apply_overlong_filtering,
    compute_turn_token_counts,
    concatenate_trajectory_batches,
    minimum_captured_global_step,
    encode_messages_subset,
    get_batch_failure_metrics,
    get_response_ids_and_loss_mask_from_messages,
    get_generation_prompt_ids,
)
from transformers import AutoTokenizer


def test_minimum_captured_global_step_uses_oldest_sample_in_group():
    outputs = [
        SimpleNamespace(captured_global_step=9),
        SimpleNamespace(captured_global_step=None),
        SimpleNamespace(captured_global_step=6),
    ]

    assert minimum_captured_global_step(outputs) == 6


@pytest.mark.parametrize(
    "loss_masks,response_ids,eos_token_id,expected_masks",
    [
        # Test case 1: All responses end with eos token - masks should remain unchanged
        (
            [[1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1]],
            [[1, 2, 3, 4], [5, 6, 7, 4], [8, 9, 4]],  # All end with eos_token_id=4
            4,
            [[1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1]],
        ),
        # Test case 2: No responses end with eos token - all masks should be zeroed
        (
            [[1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1]],
            [[1, 2, 3, 5], [5, 6, 7, 8], [8, 9, 10]],  # None end with eos_token_id=4
            4,
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0]],
        ),
        # Test case 3: Mixed responses - only non-eos ending masks should be zeroed
        (
            [[1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1, 0, 1]],
            [[1, 2, 3, 4], [5, 6, 7, 8], [8, 9, 10, 11, 4]],  # First and third end with eos_token_id=4
            4,
            [[1, 1, 0, 1], [0, 0, 0, 0], [1, 0, 1, 0, 1]],
        ),
        # Test case 4: Empty responses should be zeroed
        (
            [[1, 1], [1, 0, 1], [0, 1, 1, 1]],
            [[], [1, 2, 3], [4, 5, 6, 7]],  # Empty, no eos, no eos (eos_token_id=4)
            4,
            [[0, 0], [0, 0, 0], [0, 0, 0, 0]],
        ),
        # Test case 5: Empty lists
        ([], [], 4, []),
        # Test case 6: Different eos token id
        (
            [[1, 1], [1, 0, 1], [0, 1, 1, 1]],
            [[1, 2], [3, 4, 99], [5, 6, 7, 99]],  # Second and third end with eos_token_id=99
            99,
            [[0, 0], [1, 0, 1], [0, 1, 1, 1]],
        ),
    ],
)
def test_apply_overlong_filtering(loss_masks, response_ids, eos_token_id, expected_masks):
    """
    Test the apply_overlong_filtering function which implements DAPO Overlong Filtering.

    This function should zero-out every token's mask whenever the response does not end
    with the eos token id (i.e. truncated), while leaving other masks unchanged.
    """
    result = apply_overlong_filtering(loss_masks, response_ids, eos_token_id)

    assert result == expected_masks, f"Expected {expected_masks}, but got {result}"

    # Verify that the original inputs are not modified (immutability check)
    assert len(result) == len(loss_masks), "Result should have same length as input"

    # Check that each individual mask is processed correctly
    for i, (original_mask, response, expected_mask) in enumerate(zip(loss_masks, response_ids, expected_masks)):
        if len(response) == 0 or response[-1] != eos_token_id:
            # Should be all zeros with same length as original
            assert result[i] == [0] * len(original_mask), f"Mask {i} should be all zeros for truncated response"
        else:
            # Should be unchanged
            assert result[i] == original_mask, f"Mask {i} should be unchanged for response ending with eos token"


def test_apply_overlong_filtering_immutability():
    """
    Test that apply_overlong_filtering doesn't modify the original input lists.
    """
    original_loss_masks = [[1, 1, 0, 1], [0, 1, 1]]
    original_response_ids = [[1, 2, 3, 4], [5, 6, 7]]  # First ends with eos=4, second doesn't
    eos_token_id = 4

    # Create copies to compare against later
    loss_masks_copy = [mask[:] for mask in original_loss_masks]  # Deep copy of lists
    response_ids_copy = [response[:] for response in original_response_ids]  # Deep copy of lists

    result = apply_overlong_filtering(original_loss_masks, original_response_ids, eos_token_id)

    # Verify original inputs are unchanged
    assert original_loss_masks == loss_masks_copy, "Original loss_masks should not be modified"
    assert original_response_ids == response_ids_copy, "Original response_ids should not be modified"

    # Verify result is correct
    expected = [[1, 1, 0, 1], [0, 0, 0]]  # Second mask zeroed due to not ending with eos
    assert result == expected, f"Expected {expected}, got {result}"


@pytest.mark.parametrize(
    "loss_masks,response_ids",
    [
        # Test case 1: More loss_masks than response_ids
        ([[1, 1], [0, 1]], [[1, 2]]),
        # Test case 2: More response_ids than loss_masks
        ([[1, 1]], [[1, 2], [3, 4]]),
        # Test case 3: Empty loss_masks but non-empty response_ids
        ([], [[1, 2]]),
        # Test case 4: Non-empty loss_masks but empty response_ids
        ([[1, 0]], []),
    ],
)
def test_apply_overlong_filtering_length_mismatch_assertion(loss_masks, response_ids):
    """
    Test that apply_overlong_filtering raises AssertionError when loss_masks and response_ids
    have different lengths.
    """
    eos_token_id = 4
    with pytest.raises(AssertionError, match="loss_masks and response_ids must have the same length"):
        apply_overlong_filtering(loss_masks, response_ids, eos_token_id)


def test_compute_turn_token_counts():
    """`compute_turn_token_counts` returns one count per turn (maximal run of non-zero loss-mask
    entries), flattened across trajectories. Observation/non-assistant tokens (0) split turns;
    empty or all-zero masks contribute nothing."""
    # Single turn: all assistant tokens.
    assert compute_turn_token_counts([[1, 1, 1]]) == [3]

    # Observation tokens (0) split assistant runs into separate turns.
    assert compute_turn_token_counts([[1, 1, 0, 0, 1, 1, 1]]) == [2, 3]

    # Leading/trailing zeros don't create or extend turns.
    assert compute_turn_token_counts([[0, 0, 1, 1, 0]]) == [2]

    # Isolated single non-zero tokens are length-1 turns.
    assert compute_turn_token_counts([[0, 1, 0, 1, 0]]) == [1, 1]

    # Multiple trajectories: flat list across all, preserving order.
    assert compute_turn_token_counts([[1, 1], [1, 0, 1, 1]]) == [2, 1, 2]

    # All-zero mask (e.g. dropped by overlong filtering) contributes no turns.
    assert compute_turn_token_counts([[0, 0, 0]]) == []

    # Empty mask is skipped.
    assert compute_turn_token_counts([[]]) == []

    # Empty batch.
    assert compute_turn_token_counts([]) == []

    # Mixed batch: only the trajectory with assistant tokens contributes.
    assert compute_turn_token_counts([[], [0, 0], [1, 1, 1]]) == [3]


dummy_chat_template = (
    "{%- for message in messages %}"
    "{%- if message['role'] == 'user' %}"
    "<USER>{{ message['content'] }}</s>\n"
    "{%- elif message['role'] == 'assistant' %}"
    "<ASSISTANT>{{ message['content'] }}</s>\n"
    "{%- elif message['role'] == 'system' %}"
    "<SYSTEM>{{ message['content'] }}</s>\n"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "<ASSISTANT>"
    "{%- endif %}"
)


@pytest.fixture
def tokenizer_w_dummy_template():
    tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-2-7b")
    tokenizer.chat_template = dummy_chat_template
    return tokenizer


@pytest.mark.parametrize(
    "messages",
    [
        # Test case 1: Single assistant message
        [{"role": "assistant", "content": "Hello, I can help you."}],
        # Test case 2: Single user message
        [{"role": "user", "content": "What is the weather today?"}],
        # Test case 3: Multiple messages (user-assistant exchange)
        [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "The answer is 4."}],
        # Test case 4: Multiple messages starting with assistant
        [
            {"role": "assistant", "content": "I'm here to help."},
            {"role": "user", "content": "Can you explain Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
        ],
    ],
)
def test_encode_messages(messages, tokenizer_w_dummy_template):
    # The fixed-base approach must be SELF-CONSISTENT: encoding the messages jointly must
    # equal concatenating the per-message encodings (each produced as a fixed-base
    # continuation). This is the gen==train invariant production relies on.
    #
    # We deliberately do NOT compare against `apply_chat_template(messages)` directly: on
    # transformers 5.x, standalone-tokenizing the messages diverges from the multi-turn
    # continuation at the SentencePiece boundary token (e.g. a leading `<` tokenizes as
    # 29966 as a continuation but 529 standalone). That is a boundary shift, not a
    # gen/train divergence — the fixed-base path is what serve and re-tokenize both use.
    joint_token_ids = encode_messages_subset(messages, tokenizer_w_dummy_template)
    incremental_token_ids = []
    for message in messages:
        incremental_token_ids += encode_messages_subset([message], tokenizer_w_dummy_template)
    assert joint_token_ids == incremental_token_ids


@pytest.fixture
def qwen_tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")


@pytest.mark.parametrize(
    "messages, expected_str",
    [
        # Test case 1: Single assistant message
        (
            [{"role": "assistant", "content": "Hello, I can help you."}],
            "<|im_start|>assistant\nHello, I can help you.<|im_end|>\n",
        ),
        # Test case 2: Single user message - additional \n because the expectation is that there is a previous assistant turn
        (
            [{"role": "user", "content": "What is the weather today?"}],
            "<|im_start|>user\nWhat is the weather today?<|im_end|>\n",
        ),
        # Test case 3: Multiple messages (user-assistant exchange)
        (
            [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "The answer is 4."}],
            # NOTE: Additional \n because the expectation is that there is a previous assistant turn.
            # All tokens after EOS in the previous turn get pushed into the next user/tool message.
            "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\nThe answer is 4.<|im_end|>\n",
        ),
        # Test case 4: Multiple messages starting with assistant
        (
            [
                {"role": "assistant", "content": "I'm here to help."},
                {"role": "user", "content": "Can you explain Python?"},
                {"role": "assistant", "content": "Python is a programming language."},
            ],
            "<|im_start|>assistant\nI'm here to help.<|im_end|>\n<|im_start|>user\nCan you explain Python?<|im_end|>\n<|im_start|>assistant\nPython is a programming language.<|im_end|>\n",
        ),
    ],
)
def test_encode_messages_qwen(messages, expected_str, qwen_tokenizer):
    expected_token_ids = qwen_tokenizer.encode(expected_str, add_special_tokens=False)
    actual_token_ids = encode_messages_subset(messages, qwen_tokenizer)
    assert expected_token_ids == actual_token_ids, f"Got actual tokens: {qwen_tokenizer.decode(actual_token_ids)}"


@pytest.fixture
def qwen3_tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


THINKING_CONTENT = "<think>\nmock thinking\n</think>\n\n"


@pytest.mark.parametrize(
    "messages, expected_str",
    [
        # Test case 1: Single assistant message
        (
            [{"role": "assistant", "content": THINKING_CONTENT + "Hello, I can help you."}],
            "<|im_start|>assistant\n" + THINKING_CONTENT + "Hello, I can help you.<|im_end|>\n",
        ),
        # Test case 2: Single user message - additional \n because the expectation is that there is a previous assistant turn
        (
            [{"role": "user", "content": "What is the weather today?"}],
            "<|im_start|>user\nWhat is the weather today?<|im_end|>\n",
        ),
        # Test case 3: Multiple messages (user-assistant exchange)
        (
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": THINKING_CONTENT + "The answer is 4."},
            ],
            # NOTE: Additional \n because the expectation is that there is a previous assistant turn.
            # All tokens after EOS in the previous turn get pushed into the next user/tool message.
            "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
            + THINKING_CONTENT
            + "The answer is 4.<|im_end|>\n",
        ),
        # Test case 4: Multiple messages starting with assistant
        (
            [
                {"role": "assistant", "content": THINKING_CONTENT + "I'm here to help."},
                {"role": "user", "content": "Can you explain Python?"},
                {"role": "assistant", "content": THINKING_CONTENT + "Python is a programming language."},
            ],
            "<|im_start|>assistant\nI'm here to help.<|im_end|>\n<|im_start|>user\nCan you explain Python?<|im_end|>\n<|im_start|>assistant\n"
            + THINKING_CONTENT
            + "Python is a programming language.<|im_end|>\n",
        ),
    ],
)
def test_encode_messages_qwen3(messages, expected_str, qwen3_tokenizer):
    expected_token_ids = qwen3_tokenizer.encode(expected_str, add_special_tokens=False)
    actual_token_ids = encode_messages_subset(messages, qwen3_tokenizer)
    assert expected_token_ids == actual_token_ids, f"Got actual tokens: {qwen3_tokenizer.decode(actual_token_ids)}"


# ============================================================================
# Tests for get_response_ids_and_loss_mask_from_messages
# ============================================================================


@pytest.fixture
def llama_tokenizer():
    return AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")


class TestGetResponseIdsAndLossMaskFromMessages:
    """
    Tests for `get_response_ids_and_loss_mask_from_messages`.

    Key things to verify:
    1. Generation prompt tokens should have loss mask 0
    2. Assistant-generated tokens (including EOS) should have loss mask 1
    3. Tokens after EOS (like `\\n` in Qwen models) should have loss mask 0
    4. User message tokens should all have loss mask 0
    5. Total length of response_ids and loss_mask should match
    """

    # ------------------------------------------------------------------
    # Test single assistant message
    # ------------------------------------------------------------------
    @pytest.mark.parametrize(
        "model_name,content",
        [
            ("Qwen/Qwen2.5-0.5B-Instruct", "Hello, I can help you."),
            ("unsloth/Llama-3.2-1B-Instruct", "Hello, I can help you."),
            ("Qwen/Qwen3-0.6B", "Hello, I can help you."),
            ("Qwen/Qwen3-0.6B", THINKING_CONTENT + "Hello, I can help you."),
        ],
        ids=[
            "qwen2_5-simple",
            "llama3_2-simple",
            "qwen3-simple",
            "qwen3-with-thinking",
        ],
    )
    def test_single_assistant_message(self, model_name, content):
        """Test that a single assistant message has correct loss mask."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        messages = [{"role": "assistant", "content": content}]

        response_ids, loss_mask, rollout_logprobs = get_response_ids_and_loss_mask_from_messages(messages, tokenizer)

        # Verify lengths match
        assert len(response_ids) == len(loss_mask)
        assert rollout_logprobs is None

        # Verify the response_ids decode to expected string
        decoded = tokenizer.decode(response_ids)
        assert content in decoded

        # Verify generation prompt tokens have mask 0
        generation_prompt_ids = get_generation_prompt_ids(tokenizer)
        assert loss_mask[: len(generation_prompt_ids)] == [0] * len(generation_prompt_ids)

        # Verify EOS token is present and has mask 1
        assert tokenizer.eos_token_id in response_ids
        last_eos_idx = len(response_ids) - 1 - response_ids[::-1].index(tokenizer.eos_token_id)
        assert loss_mask[last_eos_idx] == 1

        # Verify tokens after EOS have mask 0 (like \n in Qwen)
        if last_eos_idx < len(response_ids) - 1:
            assert all(m == 0 for m in loss_mask[last_eos_idx + 1 :])

        # Verify tokens between generation prompt and EOS have mask 1
        assert all(m == 1 for m in loss_mask[len(generation_prompt_ids) : last_eos_idx + 1])

    # ------------------------------------------------------------------
    # Test single user message
    # ------------------------------------------------------------------
    @pytest.mark.parametrize(
        "model_name",
        [
            "Qwen/Qwen2.5-0.5B-Instruct",
            "unsloth/Llama-3.2-1B-Instruct",
            "Qwen/Qwen3-0.6B",
        ],
        ids=["qwen2_5", "llama3_2", "qwen3"],
    )
    def test_single_user_message(self, model_name):
        """Test that a single user message has all zeros in loss mask."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        messages = [{"role": "user", "content": "What is the weather today?"}]

        response_ids, loss_mask, rollout_logprobs = get_response_ids_and_loss_mask_from_messages(messages, tokenizer)

        # Verify lengths match
        assert len(response_ids) == len(loss_mask)
        assert rollout_logprobs is None

        # All user message tokens should have mask 0
        assert all(m == 0 for m in loss_mask)

        # Verify the content is in the decoded response
        decoded = tokenizer.decode(response_ids)
        assert "What is the weather today?" in decoded

    # ------------------------------------------------------------------
    # Test multi-turn conversation (user-assistant-user-assistant)
    # ------------------------------------------------------------------
    @pytest.mark.parametrize(
        "model_name,assistant_content",
        [
            ("Qwen/Qwen2.5-0.5B-Instruct", "The answer is 4."),
            ("unsloth/Llama-3.2-1B-Instruct", "The answer is 4."),
            ("Qwen/Qwen3-0.6B", "The answer is 4."),
            ("Qwen/Qwen3-0.6B", THINKING_CONTENT + "The answer is 4."),
        ],
        ids=["qwen2_5", "llama3_2", "qwen3-simple", "qwen3-with-thinking"],
    )
    def test_multi_turn_user_assistant(self, model_name, assistant_content):
        """Test multi-turn conversation with user and assistant messages."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": assistant_content},
            {"role": "user", "content": "And what is 3+3?"},
            {"role": "assistant", "content": assistant_content},
        ]

        response_ids, loss_mask, rollout_logprobs = get_response_ids_and_loss_mask_from_messages(messages, tokenizer)

        # Verify lengths match
        assert len(response_ids) == len(loss_mask)
        assert rollout_logprobs is None

        # Count assistant messages and verify we have the right number of 1s in the mask
        generation_prompt_ids = get_generation_prompt_ids(tokenizer)

        # Verify each message's loss mask is correctly assigned
        current_pos = 0
        for msg in messages:
            msg_token_ids = encode_messages_subset([msg], tokenizer)
            msg_loss_mask = loss_mask[current_pos : current_pos + len(msg_token_ids)]

            if msg["role"] == "user":
                # User messages should be all zeros
                assert all(m == 0 for m in msg_loss_mask), "User message should have all 0s in loss mask"
            else:
                # Assistant messages:
                # - Generation prompt: 0
                # - Generated tokens (including EOS): 1
                # - Tokens after EOS: 0
                assert msg_loss_mask[: len(generation_prompt_ids)] == [0] * len(generation_prompt_ids)

                assert tokenizer.eos_token_id in msg_token_ids, "Assistant message should contain EOS token"
                last_eos_idx = len(msg_token_ids) - 1 - msg_token_ids[::-1].index(tokenizer.eos_token_id)
                # Tokens from generation prompt end to EOS (inclusive) should be 1
                assert all(m == 1 for m in msg_loss_mask[len(generation_prompt_ids) : last_eos_idx + 1])
                # Tokens after EOS should be 0
                if last_eos_idx < len(msg_token_ids) - 1:
                    assert all(m == 0 for m in msg_loss_mask[last_eos_idx + 1 :])

            current_pos += len(msg_token_ids)

    # ------------------------------------------------------------------
    # Test with assistant_logprobs
    # ------------------------------------------------------------------
    @pytest.mark.parametrize(
        "model_name",
        [
            "Qwen/Qwen2.5-0.5B-Instruct",
            "unsloth/Llama-3.2-1B-Instruct",
            "Qwen/Qwen3-0.6B",
        ],
        ids=["qwen2_5", "llama3_2", "qwen3"],
    )
    def test_with_assistant_logprobs(self, model_name):
        """Test that assistant_logprobs are correctly handled."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        generation_prompt_ids = get_generation_prompt_ids(tokenizer)

        content = "Hello"
        messages = [{"role": "assistant", "content": content}]

        # First, get the message encoding to determine the correct logprobs length
        msg_token_ids = encode_messages_subset(messages, tokenizer)

        # Calculate the number of generated tokens (excluding generation prompt and tokens after EOS)
        assert tokenizer.eos_token_id in msg_token_ids, "Assistant message should contain EOS token"
        last_eos_idx = len(msg_token_ids) - 1 - msg_token_ids[::-1].index(tokenizer.eos_token_id)
        num_generated_tokens = last_eos_idx + 1 - len(generation_prompt_ids)

        # Create logprobs matching the generated tokens count
        mock_logprobs = [-0.5] * num_generated_tokens
        assistant_logprobs = [mock_logprobs]

        response_ids, loss_mask, rollout_logprobs = get_response_ids_and_loss_mask_from_messages(
            messages, tokenizer, assistant_logprobs=assistant_logprobs
        )

        # Verify lengths match
        assert len(response_ids) == len(loss_mask)
        assert len(rollout_logprobs) == len(response_ids)

        # Verify logprobs are 0.0 for generation prompt
        assert all(lp == 0.0 for lp in rollout_logprobs[: len(generation_prompt_ids)])

        # Verify logprobs are -0.5 for generated tokens
        # We already asserted EOS exists above, reuse last_eos_idx
        assert all(lp == -0.5 for lp in rollout_logprobs[len(generation_prompt_ids) : last_eos_idx + 1])
        # Verify logprobs are 0.0 for tokens after EOS
        if last_eos_idx < len(msg_token_ids) - 1:
            assert all(lp == 0.0 for lp in rollout_logprobs[last_eos_idx + 1 :])

    # ------------------------------------------------------------------
    # Test with multiple assistant messages and logprobs
    # ------------------------------------------------------------------
    @pytest.mark.parametrize(
        "model_name",
        [
            "Qwen/Qwen2.5-0.5B-Instruct",
            "unsloth/Llama-3.2-1B-Instruct",
            "Qwen/Qwen3-0.6B",
        ],
        ids=["qwen2_5", "llama3_2", "qwen3"],
    )
    def test_multi_assistant_with_logprobs(self, model_name):
        """Test multiple assistant messages with logprobs."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        generation_prompt_ids = get_generation_prompt_ids(tokenizer)

        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "Good"},
        ]

        # Calculate the number of generated tokens for each assistant message
        def get_num_generated_tokens(content):
            msg = [{"role": "assistant", "content": content}]
            msg_token_ids = encode_messages_subset(msg, tokenizer)
            assert tokenizer.eos_token_id in msg_token_ids, "Assistant message should contain EOS token"
            last_eos_idx = len(msg_token_ids) - 1 - msg_token_ids[::-1].index(tokenizer.eos_token_id)
            return last_eos_idx + 1 - len(generation_prompt_ids)

        num_tokens_1 = get_num_generated_tokens("Hello")
        num_tokens_2 = get_num_generated_tokens("Good")

        assistant_logprobs = [
            [-0.1] * num_tokens_1,  # logprobs for first assistant message
            [-0.2] * num_tokens_2,  # logprobs for second assistant message
        ]

        response_ids, loss_mask, rollout_logprobs = get_response_ids_and_loss_mask_from_messages(
            messages, tokenizer, assistant_logprobs=assistant_logprobs
        )

        # Verify lengths match
        assert len(response_ids) == len(loss_mask)
        assert len(rollout_logprobs) == len(response_ids)

        # Verify user messages have 0.0 logprobs
        current_pos = 0
        assistant_idx = 0
        for msg in messages:
            msg_token_ids = encode_messages_subset([msg], tokenizer)
            msg_logprobs = rollout_logprobs[current_pos : current_pos + len(msg_token_ids)]

            if msg["role"] == "user":
                assert all(lp == 0.0 for lp in msg_logprobs)
            else:
                # Assistant message
                expected_lp = -0.1 if assistant_idx == 0 else -0.2

                # Generation prompt should be 0.0
                assert all(lp == 0.0 for lp in msg_logprobs[: len(generation_prompt_ids)])

                # Generated tokens should have the expected logprob
                assert tokenizer.eos_token_id in msg_token_ids, "Assistant message should contain EOS token"
                last_eos_idx = len(msg_token_ids) - 1 - msg_token_ids[::-1].index(tokenizer.eos_token_id)
                assert all(lp == expected_lp for lp in msg_logprobs[len(generation_prompt_ids) : last_eos_idx + 1])
                # Tokens after EOS should be 0.0
                if last_eos_idx < len(msg_token_ids) - 1:
                    assert all(lp == 0.0 for lp in msg_logprobs[last_eos_idx + 1 :])

                assistant_idx += 1

            current_pos += len(msg_token_ids)

    # ------------------------------------------------------------------
    # Test error cases
    # ------------------------------------------------------------------
    def test_empty_messages_raises(self, qwen_tokenizer):
        """Test that empty messages list raises AssertionError."""
        with pytest.raises(AssertionError, match="messages list cannot be empty"):
            get_response_ids_and_loss_mask_from_messages([], qwen_tokenizer)

    def test_invalid_role_raises(self, qwen_tokenizer):
        """Test that a genuinely-unknown message role raises ValueError.

        ('system' and 'tool' are now VALID masked-observation roles — see
        test_tool_and_system_roles_masked — so the raise only fires for roles
        outside {user, assistant, tool, system}.)"""
        messages = [{"role": "function_call", "content": "not a real role"}]

        with pytest.raises(ValueError, match="Expected message role to be"):
            get_response_ids_and_loss_mask_from_messages(messages, qwen_tokenizer)

    def test_tool_and_system_roles_masked(self, qwen_tokenizer):
        """tool-result and system messages are OBSERVATIONS: rendered into the
        sequence (non-empty token ids) but fully masked (loss_mask all zeros),
        exactly like user turns. Regression guard for the keep-1 agentic starvation
        bug where role='tool' messages raised → every tool-using trajectory was
        excluded from the batch (v26 gs1 avg_num_tokens=3.34, reward=0)."""
        for role in ("tool", "system", "user"):
            messages = [{"role": role, "content": "observation content"}]
            response_ids, loss_mask, _ = get_response_ids_and_loss_mask_from_messages(messages, qwen_tokenizer)
            assert len(response_ids) > 0, f"role={role} should render to non-empty token ids"
            assert len(loss_mask) == len(response_ids)
            assert all(m == 0 for m in loss_mask), f"role={role} must be fully masked (observation, not trained on)"

    def test_missing_logprobs_degrades_and_reports(self, qwen_tokenizer):
        """Missing logprobs for an assistant message must NOT crash the RL job.

        Hardened contract: degrade gracefully (zero that message's logprobs) and
        RECORD the failure in AlignmentStats so it surfaces as
        tis/alignment_fail_count instead of silently degrading or hard-crashing.
        """
        from skyrl_train.trajectory_runners.trajectory_processing import AlignmentStats

        messages = [
            {"role": "assistant", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        generation_prompt_ids = get_generation_prompt_ids(qwen_tokenizer)
        msg_token_ids = encode_messages_subset([messages[0]], qwen_tokenizer)
        assert qwen_tokenizer.eos_token_id in msg_token_ids, "Assistant message should contain EOS token"
        last_eos_idx = len(msg_token_ids) - 1 - msg_token_ids[::-1].index(qwen_tokenizer.eos_token_id)
        num_tokens = last_eos_idx + 1 - len(generation_prompt_ids)

        assistant_logprobs = [[-0.5] * num_tokens]  # Only one logprobs list for two assistant messages
        stats = AlignmentStats()
        response_ids, loss_mask, rollout_logprobs = get_response_ids_and_loss_mask_from_messages(
            messages, qwen_tokenizer, assistant_logprobs, alignment_stats=stats
        )
        assert len(rollout_logprobs) == len(response_ids) == len(loss_mask)
        # The second assistant message had no logprobs -> recorded as a failure.
        assert stats.n_failed_messages == 1

    def test_logprobs_count_mismatch_degrades_and_reports(self, qwen_tokenizer):
        """Count mismatch with no token ids/strings: degrade + record, do not crash."""
        from skyrl_train.trajectory_runners.trajectory_processing import AlignmentStats

        messages = [{"role": "assistant", "content": "Hello"}]
        # Provide wrong number of logprobs, no token ids/strings for fallback.
        assistant_logprobs = [[-0.5] * 10]
        stats = AlignmentStats()
        response_ids, loss_mask, rollout_logprobs = get_response_ids_and_loss_mask_from_messages(
            messages, qwen_tokenizer, assistant_logprobs, alignment_stats=stats
        )
        assert len(rollout_logprobs) == len(response_ids) == len(loss_mask)
        assert stats.n_failed_messages == 1
        assert stats.n_exact == 0

    # ------------------------------------------------------------------
    # Test exact loss mask values for specific models
    # ------------------------------------------------------------------
    def test_qwen2_5_exact_loss_mask(self, qwen_tokenizer):
        """Test exact loss mask values for Qwen2.5 model."""
        messages = [
            {"role": "assistant", "content": "b"},
        ]

        response_ids, loss_mask, _ = get_response_ids_and_loss_mask_from_messages(messages, qwen_tokenizer)

        # For Qwen2.5: `<|im_start|>assistant\nb<|im_end|>\n`
        # - `<|im_start|>assistant\n` is the generation prompt (mask 0)
        # - `b<|im_end|>` is the assistant generated content (mask 1)
        # - `\n` is after EOS (mask 0)
        generation_prompt_ids = get_generation_prompt_ids(qwen_tokenizer)
        gen_prompt_len = len(generation_prompt_ids)

        expected_loss_mask = [0] * gen_prompt_len + [1, 1] + [0]  # 1 for 'b', 1 for eos, 0 for \n
        assert loss_mask == expected_loss_mask, f"Expected {expected_loss_mask}, got {loss_mask}"

    def test_llama_exact_loss_mask(self, llama_tokenizer):
        """Test exact loss mask values for Llama model."""
        messages = [
            {"role": "assistant", "content": "b"},
        ]

        response_ids, loss_mask, _ = get_response_ids_and_loss_mask_from_messages(messages, llama_tokenizer)

        # For Llama: `<|start_header_id|>assistant<|end_header_id|>\n\nb<|eot_id|>`
        # - `<|start_header_id|>assistant<|end_header_id|>\n\n` is the generation prompt (mask 0)
        # - `b<|eot_id|>` is the assistant generated content (mask 1)
        # - No tokens after EOS for Llama
        generation_prompt_ids = get_generation_prompt_ids(llama_tokenizer)
        gen_prompt_len = len(generation_prompt_ids)

        expected_loss_mask = [0] * gen_prompt_len + [1, 1]  # 1 for 'b', 1 for eos
        assert loss_mask == expected_loss_mask, f"Expected {expected_loss_mask}, got {loss_mask}"

    def test_qwen3_exact_loss_mask_with_thinking(self, qwen3_tokenizer):
        """Test exact loss mask values for Qwen3 model with thinking tokens."""
        thinking_content = "<think>\nmock thinking\n</think>\n\nb"
        messages = [
            {"role": "assistant", "content": thinking_content},
        ]

        response_ids, loss_mask, _ = get_response_ids_and_loss_mask_from_messages(messages, qwen3_tokenizer)

        generation_prompt_ids = get_generation_prompt_ids(qwen3_tokenizer)
        gen_prompt_len = len(generation_prompt_ids)

        # Get the number of tokens in the thinking content (excluding generation prompt and \n after eos)
        content_tokens = qwen3_tokenizer.encode(thinking_content, add_special_tokens=False)
        num_content_tokens = len(content_tokens)

        # Expected: [0]*gen_prompt_len + [1]*num_content_tokens + [1 for eos] + [0 for \n]
        expected_loss_mask = [0] * gen_prompt_len + [1] * num_content_tokens + [1] + [0]
        assert loss_mask == expected_loss_mask, f"Expected {expected_loss_mask}, got {loss_mask}"

    # ------------------------------------------------------------------
    # Test multi-turn exact loss masks
    # ------------------------------------------------------------------
    def test_qwen2_5_multi_turn_exact_loss_mask(self, qwen_tokenizer):
        """Test exact loss mask for multi-turn conversation with Qwen2.5."""
        messages = [
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "b"},
        ]

        response_ids, loss_mask, _ = get_response_ids_and_loss_mask_from_messages(messages, qwen_tokenizer)

        generation_prompt_ids = get_generation_prompt_ids(qwen_tokenizer)
        gen_prompt_len = len(generation_prompt_ids)

        # First assistant message: [0]*gen_prompt + [1, 1] for 'b' and eos + [0] for \n
        # User message: all zeros
        # Second assistant message: [0]*gen_prompt + [1, 1] for 'b' and eos + [0] for \n

        user_msg_tokens = encode_messages_subset([{"role": "user", "content": "1"}], qwen_tokenizer)

        expected_loss_mask = (
            [0] * gen_prompt_len
            + [1, 1, 0]  # first assistant
            + [0] * len(user_msg_tokens)  # user
            + [0] * gen_prompt_len
            + [1, 1, 0]  # second assistant
        )

        assert loss_mask == expected_loss_mask, f"Expected {expected_loss_mask}, got {loss_mask}"

    def test_llama_multi_turn_exact_loss_mask(self, llama_tokenizer):
        """Test exact loss mask for multi-turn conversation with Llama."""
        messages = [
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "b"},
        ]

        response_ids, loss_mask, _ = get_response_ids_and_loss_mask_from_messages(messages, llama_tokenizer)

        generation_prompt_ids = get_generation_prompt_ids(llama_tokenizer)
        gen_prompt_len = len(generation_prompt_ids)

        user_msg_tokens = encode_messages_subset([{"role": "user", "content": "1"}], llama_tokenizer)

        expected_loss_mask = (
            [0] * gen_prompt_len
            + [1, 1]  # first assistant (no \n after eos for Llama)
            + [0] * len(user_msg_tokens)  # user
            + [0] * gen_prompt_len
            + [1, 1]  # second assistant
        )

        assert loss_mask == expected_loss_mask, f"Expected {expected_loss_mask}, got {loss_mask}"

    def test_qwen3_multi_turn_exact_loss_mask(self, qwen3_tokenizer):
        """Test exact loss mask for multi-turn conversation with Qwen3."""
        messages = [
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "b"},
        ]

        response_ids, loss_mask, _ = get_response_ids_and_loss_mask_from_messages(messages, qwen3_tokenizer)

        # Verify our assumptions about token structure
        generation_prompt_ids = get_generation_prompt_ids(qwen3_tokenizer)
        assert len(generation_prompt_ids) == 3, "Qwen3 generation prompt should be 3 tokens: <|im_start|>assistant\\n"

        user_msg_tokens = encode_messages_subset([{"role": "user", "content": "1"}], qwen3_tokenizer)
        assert len(user_msg_tokens) == 6, "User message '1' should be 6 tokens: <|im_start|>user\\n1<|im_end|>\\n"

        assistant_msg_tokens = encode_messages_subset([{"role": "assistant", "content": "b"}], qwen3_tokenizer)
        # For Qwen3 with content "b": <|im_start|>assistant\n<think>\n\n</think>\n\nb<|im_end|>\n
        # = 3 (gen prompt) + 6 (content with thinking + eos) + 1 (\n after eos) = 10 tokens
        assert len(assistant_msg_tokens) == 10, (
            f"Assistant message 'b' should be 10 tokens, got {len(assistant_msg_tokens)}"
        )
        assert assistant_msg_tokens[-2] == qwen3_tokenizer.eos_token_id, "Second to last token should be EOS"

        # Expected loss mask:
        # First assistant: [0,0,0] gen_prompt + [1,1,1,1,1,1] content+eos + [0] \n = 10 tokens
        # User: [0,0,0,0,0,0] = 6 tokens
        # Second assistant: [0,0,0] gen_prompt + [1,1,1,1,1,1] content+eos + [0] \n = 10 tokens
        expected_loss_mask = (
            [0, 0, 0]
            + [1, 1, 1, 1, 1, 1]
            + [0]  # first assistant (10 tokens)
            + [0, 0, 0, 0, 0, 0]  # user (6 tokens)
            + [0, 0, 0]
            + [1, 1, 1, 1, 1, 1]
            + [0]  # second assistant (10 tokens)
        )

        assert len(expected_loss_mask) == 26, "Total should be 26 tokens"
        assert loss_mask == expected_loss_mask, f"Expected {expected_loss_mask}, got {loss_mask}"

    def test_qwen3_multi_turn_exact_loss_mask_with_thinking(self, qwen3_tokenizer):
        """Test exact loss mask for multi-turn conversation with Qwen3 including thinking content."""
        thinking_content = THINKING_CONTENT + "b"  # <think>\nmock thinking\n</think>\n\nb
        messages = [
            {"role": "assistant", "content": thinking_content},
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": thinking_content},
        ]

        response_ids, loss_mask, _ = get_response_ids_and_loss_mask_from_messages(messages, qwen3_tokenizer)

        # Verify our assumptions about token structure
        generation_prompt_ids = get_generation_prompt_ids(qwen3_tokenizer)
        assert len(generation_prompt_ids) == 3, "Qwen3 generation prompt should be 3 tokens"

        user_msg_tokens = encode_messages_subset([{"role": "user", "content": "1"}], qwen3_tokenizer)
        assert len(user_msg_tokens) == 6, "User message '1' should be 6 tokens"

        assistant_msg_tokens = encode_messages_subset(
            [{"role": "assistant", "content": thinking_content}], qwen3_tokenizer
        )
        # For Qwen3 with thinking_content "<think>\nmock thinking\n</think>\n\nb":
        # <|im_start|>assistant\n<think>\nmock thinking\n</think>\n\nb<|im_end|>\n
        # = 3 (gen prompt) + 9 (content with thinking + eos) + 1 (\n after eos) = 13 tokens
        assert len(assistant_msg_tokens) == 13, (
            f"Assistant message with thinking should be 13 tokens, got {len(assistant_msg_tokens)}"
        )
        assert assistant_msg_tokens[-2] == qwen3_tokenizer.eos_token_id, "Second to last token should be EOS"

        # Expected loss mask:
        # First assistant: [0,0,0] gen_prompt + [1]*9 content+eos + [0] \n = 13 tokens
        # User: [0]*6 = 6 tokens
        # Second assistant: [0,0,0] gen_prompt + [1]*9 content+eos + [0] \n = 13 tokens
        expected_loss_mask = (
            [0, 0, 0]
            + [1] * 9
            + [0]  # first assistant (13 tokens)
            + [0] * 6  # user (6 tokens)
            + [0, 0, 0]
            + [1] * 9
            + [0]  # second assistant (13 tokens)
        )

        assert len(expected_loss_mask) == 32, "Total should be 32 tokens"
        assert loss_mask == expected_loss_mask, f"Expected {expected_loss_mask}, got {loss_mask}"


@pytest.mark.parametrize(
    "num_trials,num_failed,expected_fraction",
    [
        (10, 4, 0.4),
        (64, 0, 0.0),  # a healthy batch still reports the series rather than omitting it
        (64, 64, 1.0),  # the orchestrator-failure path, where every trial is reported failed
        (0, 0, 0.0),  # an empty batch has no denominator to divide by
    ],
)
def test_failed_fraction_is_measured_against_the_whole_batch(num_trials, num_failed, expected_fraction):
    metrics = get_batch_failure_metrics(
        num_trials,
        num_failed_trajectories=num_failed,
        num_failed_instances=num_failed,
        num_masked_trajectories=num_failed,
    )

    assert metrics["generate/failed_trajectory_fraction"] == pytest.approx(expected_fraction)


def test_failure_counts_stay_separate_from_the_fraction():
    # Trajectories fail individually, instances group n_samples_per_prompt of
    # them, and only the infrastructure failures are masked out of the RLOO
    # baseline, so the three counts move independently and none substitutes
    # for another.
    metrics = get_batch_failure_metrics(
        64,
        num_failed_trajectories=40,
        num_failed_instances=8,
        num_masked_trajectories=24,
    )

    assert metrics == {
        "generate/num_trials": 64,
        "generate/num_failed_instances": 8,
        "generate/num_failed_trajectories": 40,
        "generate/num_masked_trajectories": 24,
        "generate/failed_trajectory_fraction": pytest.approx(0.625),
    }


def _generated_group(num_trials: int, num_failed: int, error_type: str = "SandboxError") -> dict:
    failure_metrics = get_batch_failure_metrics(
        num_trials,
        num_failed_trajectories=num_failed,
        num_failed_instances=num_failed,
        num_masked_trajectories=num_failed,
    )
    if num_failed:
        failure_metrics[f"generate/errors/{error_type}"] = num_failed
    return {
        "prompt_token_ids": [[1] for _ in range(num_trials)],
        "response_ids": [[2, 3] for _ in range(num_trials)],
        "rewards": [0.0 for _ in range(num_trials)],
        "loss_masks": [[1, 1] for _ in range(num_trials)],
        "stop_reasons": ["error" if i < num_failed else "stop" for i in range(num_trials)],
        "rollout_logprobs": None,
        "rollout_metrics": failure_metrics,
    }


def test_failure_metrics_survive_concatenation():
    # The fully asynchronous trainer generates one rollout group at a time and reads
    # rollout metrics off the concatenated result only, so a count that does not
    # survive this merge never reaches the tracker. The groups are deliberately
    # unequal: the fraction has to come from the summed totals, not from averaging
    # 0.75 and 0.0 to 0.375.
    merged = concatenate_trajectory_batches(
        [_generated_group(4, 3), _generated_group(2, 1, error_type="ContextLengthExceededError")],
        tis_lcs_alert_threshold=0.005,
    )

    assert merged["rollout_metrics"]["generate/num_trials"] == 6
    assert merged["rollout_metrics"]["generate/num_failed_trajectories"] == 4
    assert merged["rollout_metrics"]["generate/failed_trajectory_fraction"] == pytest.approx(4 / 6)
    assert merged["rollout_metrics"]["generate/errors/SandboxError"] == 3
    assert merged["rollout_metrics"]["generate/errors/ContextLengthExceededError"] == 1


def test_generators_that_report_no_trials_get_no_failure_series():
    # Only the agentic generator counts trials, and a fabricated zero series for
    # every other generator would read as a measured "nothing failed".
    merged = concatenate_trajectory_batches(
        [{**_generated_group(2, 0), "rollout_metrics": {}}], tis_lcs_alert_threshold=0.005
    )

    assert "generate/num_trials" not in merged["rollout_metrics"]
    assert "generate/failed_trajectory_fraction" not in merged["rollout_metrics"]


@pytest.mark.parametrize("timeout_first", [False, True])
def test_concatenation_preserves_unshaped_rewards_with_incomplete_timeout_output(timeout_first):
    normal = {**_generated_group(1, 0), "rewards": [1.0], "unshaped_rewards": [1.0]}
    timeout = _generated_group(1, 1, error_type="AgentTimeoutError")
    groups = [timeout, normal] if timeout_first else [normal, timeout]

    merged = concatenate_trajectory_batches(groups, tis_lcs_alert_threshold=0.005)

    assert merged["unshaped_rewards"] == ([0.0, 1.0] if timeout_first else [1.0, 0.0])


def test_partial_failure_metrics_are_rejected_during_concatenation():
    group = _generated_group(2, 1)
    del group["rollout_metrics"]["generate/num_masked_trajectories"]

    with pytest.raises(ValueError, match="generate/num_masked_trajectories"):
        concatenate_trajectory_batches([group], tis_lcs_alert_threshold=0.005)


def test_required_rollout_logprobs_reject_partial_generation_batch():
    with_logprobs = {**_generated_group(2, 0), "rollout_logprobs": [[-0.1, -0.2], [-0.3, -0.4]]}

    with pytest.raises(ValueError, match="rollout_logprobs are required"):
        concatenate_trajectory_batches(
            [with_logprobs, _generated_group(2, 0)],
            require_rollout_logprobs=True,
            tis_lcs_alert_threshold=0.005,
        )
