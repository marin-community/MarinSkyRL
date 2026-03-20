import torch
from difflib import SequenceMatcher
from typing import List, Tuple, Union, Optional, Dict, Any
from collections import defaultdict
import numpy as np
from skyrl_train.generators.base import GeneratorOutput, GeneratorInput, TrajectoryID, BatchMetadata, TrainingPhase
from skyrl_train.inference_engines.base import ConversationType
from omegaconf import DictConfig
from loguru import logger
from skyrl_gym.metrics import aggregate_for_environment


def align_logprobs_with_lcs(
    retokenized_ids: List[int],
    vllm_token_logprobs: List[Dict[str, Any]],
    tokenizer,
) -> List[float]:
    """Align vLLM logprobs to re-tokenized IDs using LCS on token strings.

    When re-tokenizing vLLM output with a different tokenizer (e.g., for TIS training),
    the token counts may differ slightly (off-by-one or more). This function uses
    Longest Common Subsequence (LCS) matching on token strings to align the logprobs
    from vLLM to the re-tokenized sequence.

    Args:
        retokenized_ids: Token IDs from re-tokenizing the response text
        vllm_token_logprobs: List of dicts with "token" (str) and "logprob" (float)
            from Harbor's rollout_details
        tokenizer: HuggingFace tokenizer used for re-tokenization

    Returns:
        List of aligned logprobs, one per retokenized_id. Unmatched tokens get 0.0.

    Example:
        >>> # vLLM tokenized as ["Hello", " world", "!"] with logprobs [-0.1, -0.2, -0.3]
        >>> # Re-tokenizer splits as ["Hello", " ", "world", "!"]
        >>> vllm_logprobs = [
        ...     {"token": "Hello", "logprob": -0.1},
        ...     {"token": " world", "logprob": -0.2},
        ...     {"token": "!", "logprob": -0.3}
        ... ]
        >>> retok_ids = tokenizer.encode("Hello world!")  # [1, 2, 3, 4]
        >>> aligned = align_logprobs_with_lcs(retok_ids, vllm_logprobs, tokenizer)
        >>> # Returns logprobs aligned to retokenized sequence via LCS matching
    """
    if not vllm_token_logprobs:
        return [0.0] * len(retokenized_ids)

    if not retokenized_ids:
        return []

    # Convert re-tokenized IDs to token strings for matching
    retok_strings = tokenizer.convert_ids_to_tokens(retokenized_ids)

    # Extract token strings and logprobs from vLLM output
    vllm_strings = [tl["token"] for tl in vllm_token_logprobs]
    vllm_logprobs = [tl["logprob"] for tl in vllm_token_logprobs]

    # Use SequenceMatcher to find LCS alignment
    matcher = SequenceMatcher(None, retok_strings, vllm_strings)
    aligned = [0.0] * len(retokenized_ids)

    # Get all matching blocks and assign logprobs
    for a_start, b_start, size in matcher.get_matching_blocks():
        for i in range(size):
            aligned[a_start + i] = vllm_logprobs[b_start + i]

    # Log alignment statistics for debugging
    matched_count = sum(1 for lp in aligned if lp != 0.0)
    if matched_count < len(retokenized_ids) * 0.9:  # Less than 90% matched
        logger.debug(
            f"LCS alignment: matched {matched_count}/{len(retokenized_ids)} tokens "
            f"(vLLM had {len(vllm_token_logprobs)} tokens). "
            f"First few retok: {retok_strings[:5]}, vLLM: {vllm_strings[:5]}"
        )

    return aligned


CUSTOM_CHAT_TEMPLATES = {
    # chat template for qwen3 that preserves thinking tokens
    "qwen3_with_thinking": (
        "{% for message in messages %}"
        "{% if (message['role'] != 'assistant') %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% elif (message['role'] == 'assistant')%}"
        "{{'<|im_start|>' + message['role'] + '\n'}}"
        "{% generation %}"
        "{{message['content'] + '<|im_end|>'}}"
        "{% endgeneration %}"
        "{{'\n'}}"
        "{% endif %}"
        "{% endfor %}"
    ),
    # chat template for qwen3 that strips non-last-turn thinking tokens (same as the official Qwen3 chat
    # template but we add `generation` and `endgeneration` tags)
    "qwen3_without_thinking": (
        "{% for message in messages %}"
        "{% if (message['role'] != 'assistant') %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% elif (message['role'] == 'assistant')%}"
        "{{'<|im_start|>' + message['role'] + '\n'}}"
        "{% generation %}"
        "{% set full_content = message['content'] %}"
        "{% set mycontent = message['content'] %}"
        "{% set is_last_message = loop.last and messages[-1]['role'] == 'assistant' %}"
        "{% if '</think>' in full_content and not is_last_message %}"
        "{% set mycontent = full_content.split('</think>')[-1].lstrip('\n') %}"
        "{% endif %}"
        "{{mycontent + '<|im_end|>'}}"
        "{% endgeneration %}"
        "{{'\n'}}"
        "{% endif %}"
        "{% endfor %}"
    ),
    # Qwen2.5 chat template but with `generation` and `endgeneration` tags, and simplified
    "qwen2_5_with_generation_tag_simplified": (
        "{% for message in messages %}"
        "{% if (message.role == 'user') or (message.role == 'system' and not loop.first) %}"
        "{{ '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}"
        "{% elif message.role == 'assistant' %}"
        "{{ '<|im_start|>' + message.role + '\n'}}"
        "{% generation %}"
        "{{ message.content + '<|im_end|>'}}"
        "{% endgeneration %}"
        "{{ '\n' }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
    ),
}


def get_custom_chat_template(chat_template_config: Optional[Union[dict, DictConfig]] = None) -> Optional[str]:
    """
    Get custom chat template based on the new config structure.

    Args:
        chat_template_config: Config dict with 'source' and 'name_or_path' fields.

    Returns:
        Chat template string or None
    """
    if chat_template_config is None:
        return None

    source = chat_template_config.get("source")
    if not source:
        raise ValueError("'source' is required in chat_template_config")

    name_or_path = chat_template_config.get("name_or_path")
    if not name_or_path:
        return None  # if name_or_path is not provided, use the default chat template from the tokenizer

    if source == "name":
        if name_or_path in CUSTOM_CHAT_TEMPLATES:
            return CUSTOM_CHAT_TEMPLATES[name_or_path]
        else:
            raise ValueError(
                f"Template name '{name_or_path}' not found. Available templates: {list(CUSTOM_CHAT_TEMPLATES.keys())}"
            )
    elif source == "file":
        try:
            with open(name_or_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError as e:
            raise ValueError(f"Template file '{name_or_path}' not found") from e
        except OSError as e:
            raise ValueError(f"Error reading template file '{name_or_path}': {e}") from e
    else:
        raise ValueError(f"Invalid source '{source}'. Must be 'name' or 'file'")


def get_generation_prompt_ids(tokenizer, custom_chat_template=None) -> List[int]:
    """
    Helper function to get the generation prompt ids for a given tokenizer.
    """
    empty_user = tokenizer.apply_chat_template([{"role": "user", "content": ""}], tokenize=True, chat_template=custom_chat_template)
    empty_user_with_generation_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=True, chat_template=custom_chat_template
    )

    generation_prompt_ids = empty_user_with_generation_prompt[len(empty_user) :]
    return generation_prompt_ids


@torch.no_grad()
def get_metrics_from_generator_output(generator_output: GeneratorOutput, uids: List[str]) -> Tuple[float, float]:
    """
    Get `mean_raw_reward` (or avg_score), `pass_at_n` from generator output.

    The `n` in `pass_at_n` is the number of trajectories we generate for each example. It is
    calculated as `len(generator_output["rewards"]) / len(uids)`, where `len(uids)` is the number of
    unique examples.

    Rewards can be either per-trajectory or per-token, and metrics are computed correspondingly.
    """
    rewards: Union[List[float], List[List[float]]] = generator_output["rewards"]
    if not len(rewards):
        raise ValueError(f"`rewards` must be a non-empty list, got {rewards}")

    # TODO: We should make metrics customizable by the environment.
    # Map from the example's uid to each trajectory's reward on that same example
    uid_to_trajectory_rewards = defaultdict(list)
    if isinstance(rewards[0], list):
        # Token-level rewards: rewards is List[List[float]]
        # For each trajectory, we sum over the token rewards for `mean_raw_reward` computation
        mean_raw_reward = float(np.mean([sum(trajectory_rewards) for trajectory_rewards in rewards]))
        # Assume the last token's reward signifies the trajectory's reward for `pass_at_n` computation
        for i, cur_trajectory_rewards in enumerate(rewards):
            if len(cur_trajectory_rewards) == 0:
                raise ValueError("Token-level rewards must be a non-empty list.")
            uid_to_trajectory_rewards[uids[i]].append(cur_trajectory_rewards[-1])
    else:
        mean_raw_reward = float(np.mean(rewards))
        for i, reward in enumerate(rewards):
            uid_to_trajectory_rewards[uids[i]].append(reward)

    # For each example, pass@n = 1 if any trajectory achieves full success (reward >= 1.0).
    # This threshold handles shaped rewards correctly: partial credit (e.g. 3/10 tests = 0.3)
    # no longer inflates pass@n, only fully solved tasks count.
    pass_at_n = sum(1 for v in uid_to_trajectory_rewards.values() if any(r >= 1.0 for r in v)) / len(
        uid_to_trajectory_rewards
    )

    return mean_raw_reward, pass_at_n


def concatenate_generator_outputs(generator_outputs: List[GeneratorOutput]) -> GeneratorOutput:
    """
    Concatenate the generator outputs of multiple batches.

    We only aggregate rollout metrics the can deduced by responses and rewards, but not
    those that use `env_metrics` or `env_classes`.
    """
    assert len(generator_outputs) > 0
    has_rollout_logprobs = [output.get("rollout_logprobs") is not None for output in generator_outputs]
    any_has_logprobs = any(has_rollout_logprobs)

    # Handle mixed rollout_logprobs: if some batches have logprobs and others don't,
    # fill in placeholder [0.0] values for the batches that don't have them.
    # This can happen when all trials in a batch fail (returns None) while other batches succeed.
    rollout_logprobs_concat = None
    if any_has_logprobs:
        rollout_logprobs_concat = []
        for output in generator_outputs:
            if output.get("rollout_logprobs") is not None:
                rollout_logprobs_concat.extend(output["rollout_logprobs"])
            else:
                # Fill in placeholder logprobs for batches that don't have them
                # Each trajectory needs logprobs matching its response_ids length
                for response_ids in output["response_ids"]:
                    rollout_logprobs_concat.append([0.0] * len(response_ids))

    result: GeneratorOutput = {
        "prompt_token_ids": sum([output["prompt_token_ids"] for output in generator_outputs], []),
        "response_ids": sum([output["response_ids"] for output in generator_outputs], []),
        "rewards": sum([output["rewards"] for output in generator_outputs], []),
        "loss_masks": sum([output["loss_masks"] for output in generator_outputs], []),
        "stop_reasons": (
            sum([output["stop_reasons"] for output in generator_outputs], [])
            if "stop_reasons" in generator_outputs[0] and generator_outputs[0]["stop_reasons"] is not None
            else None
        ),
        "rollout_logprobs": rollout_logprobs_concat,
    }

    # propagate additional keys with list values as-is
    additional_keys = [
        key for key in generator_outputs[0] if key not in result and isinstance(generator_outputs[0][key], list)
    ]
    if len(additional_keys):
        logger.info(f"Attempting to concatenate values for additional keys {additional_keys}")
    for key in additional_keys:
        result[key] = sum([generator_output[key] for generator_output in generator_outputs], [])

    # Re-aggregate rollout metrics
    rollout_metrics = get_rollout_metrics(result["response_ids"], result["rewards"])
    result["rollout_metrics"] = rollout_metrics

    # Validate the generator output using the number of prompts
    # Import here to avoid circular dependency.
    from skyrl_train.utils.trainer_utils import validate_generator_output

    num_prompts = len(result["prompt_token_ids"])
    validate_generator_output(num_prompts, result)

    return result


def apply_overlong_filtering(
    loss_masks: List[List[int]],
    response_ids: List[List[int]],
    eos_token_id: int,
) -> List[List[int]]:
    """
    Implements DAPO Overlong Filtering: zero-out every token's mask whenever
    the response does not end with the eos token id (i.e. truncated).

    Returns:
        - The loss masks with tokens zeroed out for truncated responses
    """
    assert len(loss_masks) == len(response_ids), "loss_masks and response_ids must have the same length"
    return [
        [0] * len(mask) if not response or response[-1] != eos_token_id else mask
        for mask, response in zip(loss_masks, response_ids)
    ]


def get_rollout_metrics(
    responses: List[List[int]],
    rewards: Union[List[float], List[List[float]]],
    env_metrics: Optional[List[Dict[str, Any]]] = None,
    env_classes: Optional[List[str]] = None,
):
    """
    Computes rollout metrics including token statistics and optional environment-specific metrics.

    Args:
        responses: List of token ID sequences for each response
        rewards: List of rewards (either per-trajectory or per-token)
        env_metrics: Optional list of environment-specific metrics for each trajectory
        env_classes: Optional list of environment class names for each trajectory

    Returns:
        Dictionary of aggregated metrics
    """
    num_tokens_arr = np.array([len(response) for response in responses])
    # Support both response-level and token-level rewards
    flat_rewards = []
    for r in rewards:
        if isinstance(r, list):
            flat_rewards.append(float(np.sum(r)))
        else:
            flat_rewards.append(float(r))
    flat_rewards_arr = np.array(flat_rewards)
    non_zero_rewards_arr = flat_rewards_arr > 0.0
    zero_rewards_arr = flat_rewards_arr == 0.0
    # average tokens for non zero rewards
    avg_tokens_non_zero_rewards = (
        np.mean(num_tokens_arr[non_zero_rewards_arr]) if non_zero_rewards_arr.sum() > 0 else np.zeros(1)
    )
    # average tokens for zero rewards
    avg_tokens_zero_rewards = np.mean(num_tokens_arr[zero_rewards_arr]) if zero_rewards_arr.sum() > 0 else np.zeros(1)

    rollout_metrics = {
        "generate/min_num_tokens": np.min(num_tokens_arr).item(),
        "generate/max_num_tokens": np.max(num_tokens_arr).item(),
        "generate/avg_num_tokens": np.mean(num_tokens_arr).item(),
        "generate/std_num_tokens": np.std(num_tokens_arr).item(),
        "generate/avg_tokens_non_zero_rewards": avg_tokens_non_zero_rewards.item(),
        "generate/avg_tokens_zero_rewards": avg_tokens_zero_rewards.item(),
    }

    if env_metrics is not None and env_classes is not None:
        env_to_metrics = defaultdict(list)
        for i, metrics in enumerate(env_metrics):
            env_to_metrics[env_classes[i]].append(metrics)
        for env_name, metrics in env_to_metrics.items():
            # Aggregate metrics across all trajectories for the same environment
            agg = aggregate_for_environment(env_name, metrics)
            for key, value in agg.items():
                rollout_metrics[f"environment/{key}"] = value

    return rollout_metrics


def prepare_generator_input(
    prompts: List[Any],
    n_samples_per_prompt: int,
    sampling_params: Dict[str, Any],
    default_env_class: str,
    training_phase: TrainingPhase,
    global_step: int,
) -> Tuple[GeneratorInput, List[str]]:
    """Prepares the generator input for training and eval

    Args:
        prompts (List[Any]): list of prompts
        n_samples_per_prompt (int): how many samples to create per prompt
        sampling_params (Dict[str, Any]): sampling parameters
        default_env_class (str): env class to use if env class missing from prompts
        training_phase (TrainingPhase): training or eval
        global_step (int): current global step

    Returns:
        Tuple[GeneratorInput, List[str]]: generator input and list of uuids
    """

    all_prompts = [prompt["prompt"] for prompt in prompts for _ in range(n_samples_per_prompt)]

    all_envs = [
        prompt["env_class"] if prompt["env_class"] is not None else default_env_class
        for prompt in prompts
        for _ in range(n_samples_per_prompt)
    ]

    # all the other columns are env_extras
    env_extras = [prompt["env_extras"] for prompt in prompts for _ in range(n_samples_per_prompt)]

    # Create TrajectoryID objects - one UID per row, repetition_id for multiple samples
    trajectory_ids = []
    uids = []
    for _, prompt in enumerate(prompts):
        uid: str = prompt["uid"]

        # Create TrajectoryID for each repetition
        for repetition_id in range(n_samples_per_prompt):
            trajectory_ids.append(TrajectoryID(instance_id=uid, repetition_id=repetition_id))
            uids.append(uid)

    generator_input: GeneratorInput = {
        "prompts": all_prompts,
        "env_classes": all_envs,
        "env_extras": env_extras,
        "sampling_params": sampling_params,
        "trajectory_ids": trajectory_ids,
        "batch_metadata": BatchMetadata(global_step=global_step, training_phase=training_phase),
    }

    return generator_input, uids


def encode_messages_subset(messages: ConversationType, tokenizer, custom_chat_template=None):
    """Encodes a subset of messages from a multi-turn conversation using the fixed base approach.

    This function tokenizes messages as if they are part of a larger conversation, ensuring
    no additional default system messages are prepended by the tokenizer's chat template

    The "fixed base approach" works by:
    - Creating a dummy base conversation to establish context
    - Appending the target messages to this base
    - Tokenizing the full conversation and extracting only the tokens for the target messages

    For simple chat templates without complex token splitting behavior, this produces the same
    result as directly tokenizing the messages. For templates like Qwen's ChatML format where
    a default system prompt can be appended, this ensures correct tokenization.

    In addition, for Qwen3, this function will keep all the thinking tokens from the messages.

    Reference: https://jybsuper.github.io/posts/multiturn_tokenization/#the-breakthrough-fixed-base-approach

    Args:
        messages: List of message dicts with 'role' and 'content' keys. Must contain at least
                 one message. These are assumed to be a subset from a larger conversation.
        tokenizer: HuggingFace tokenizer with chat_template support and eos_token_id defined.
        custom_chat_template: Optional custom chat template string to use instead of tokenizer's default.

    Returns:
        List[int]: Token IDs for the given messages, with proper multi-turn context handling.
    """
    assert len(messages), "messages list cannot be empty"
    # Follows https://jybsuper.github.io/posts/multiturn_tokenization/#the-breakthrough-fixed-base-approach
    base_conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "I am a user."},
    ]
    base_conversation_token_ids = tokenizer.apply_chat_template(
        base_conversation,
        add_generation_prompt=False,
        tokenize=True,
        chat_template=custom_chat_template,
    )

    full_conversation = base_conversation + messages
    full_conversation_token_ids = tokenizer.apply_chat_template(
        full_conversation,
        add_generation_prompt=False,
        tokenize=True,
        chat_template=custom_chat_template,
    )
    conversation_token_ids = full_conversation_token_ids[len(base_conversation_token_ids) :]
    return conversation_token_ids


def extract_logprobs_from_rollout_details(
    rollout_details: Optional[List[Dict[str, Any]]],
) -> Optional[List[List[Dict[str, Any]]]]:
    """
    Extract per-turn logprobs (with token strings) from Harbor's rollout_details structure.

    Harbor stores rollout details as a list of RolloutDetail dicts. Each RolloutDetail
    contains per-turn data for a conversation trajectory:
        - prompt_token_ids: list[list[int]] - prompt tokens per turn
        - completion_token_ids: list[list[int]] - completion tokens per turn
        - logprobs: list[list[dict]] - logprobs per turn, where each dict has
            {"token": str, "logprob": float} for LCS alignment

    For agents with subagents or summarization, multiple RolloutDetail objects may exist.
    By convention, the first RolloutDetail contains the main agent's conversation.

    Args:
        rollout_details: List of RolloutDetail dicts from Harbor's AgentContext.
            Can be None or empty if rollout details weren't collected.

    Returns:
        Per-turn logprobs in format [[{token, logprob}_turn1], [{token, logprob}_turn2], ...],
        or None if rollout_details is empty/missing or doesn't contain logprobs.
        Each inner dict has "token" (str) and "logprob" (float) keys.

    Example:
        >>> rollout_details = result.agent_result.rollout_details
        >>> assistant_logprobs = extract_logprobs_from_rollout_details(rollout_details)
        >>> if assistant_logprobs:
        ...     response_ids, loss_mask, rollout_logprobs = get_response_ids_and_loss_mask_from_messages(
        ...         messages, tokenizer, assistant_logprobs
        ...     )
    """
    if not rollout_details or len(rollout_details) == 0:
        return None

    # First rollout_detail contains the main agent's conversation
    main_rollout = rollout_details[0]

    # Handle both dict and object-like access patterns
    if isinstance(main_rollout, dict):
        logprobs = main_rollout.get("logprobs")
    else:
        logprobs = getattr(main_rollout, "logprobs", None)

    if not logprobs:
        return None

    # Validate structure: should be list of lists
    if not isinstance(logprobs, list):
        logger.warning(f"Unexpected logprobs type: {type(logprobs)}, expected list")
        return None

    if len(logprobs) > 0 and not isinstance(logprobs[0], list):
        logger.warning(
            f"Unexpected logprobs[0] type: {type(logprobs[0])}, expected list. "
            f"rollout_details may have unexpected structure."
        )
        return None

    # Validate the inner structure contains dicts with token+logprob (new format)
    # or just floats (legacy format - handle gracefully)
    if len(logprobs) > 0 and len(logprobs[0]) > 0:
        first_item = logprobs[0][0]
        if isinstance(first_item, (int, float)):
            # Legacy format: list[list[float]] - convert to new format without token strings
            # This allows backward compatibility but LCS alignment won't work
            logger.warning(
                "Detected legacy logprobs format (list[list[float]]). "
                "Token strings not available, LCS alignment will be disabled. "
                "Update Harbor to get token strings for proper TIS support."
            )
            return None  # Return None to disable logprobs for legacy format

    logger.debug(f"Extracted logprobs from rollout_details: {len(logprobs)} turns")
    return logprobs


def get_response_ids_and_loss_mask_from_messages(messages: ConversationType, tokenizer, assistant_logprobs=None, custom_chat_template=None):
    """
    Get the response ids and loss mask from a list of messages.

    We encode each message one by one, using a fixed base approach, building response token IDs, loss mask,
    and rollout logprobs if provided. For Qwen3, this function will keep all the thinking tokens from the messages.

    When assistant_logprobs contains token strings (new format from Harbor), this function uses LCS
    (Longest Common Subsequence) alignment to handle tokenization mismatches between vLLM and the
    training tokenizer. This solves the off-by-one problem in TIS (Truncated Importance Sampling).

    Args:
        messages: List of message dicts with 'role' and 'content' keys. Must contain at least
                 one message.
        tokenizer: HuggingFace tokenizer with chat_template support and eos_token_id defined.
        assistant_logprobs: Optional list of logprobs for each assistant message. Supports two formats:
            - New format (with token strings): [[{"token": str, "logprob": float}, ...], ...]
            - Legacy format (floats only): [[float, ...], ...] - will trigger a warning
        custom_chat_template: Optional custom chat template string to use instead of tokenizer's default.

    Returns:
        Tuple[List[int], List[int], Optional[List[float]]]: response ids, loss mask, and rollout logprobs
    """
    assert len(messages), "messages list cannot be empty"

    # Needed to correctly mask it zero for assistant messages.
    generation_prompt_ids = get_generation_prompt_ids(tokenizer, custom_chat_template=custom_chat_template)

    # 1. Initalize the things to accumulate
    response_ids = []
    loss_mask = []
    rollout_logprobs = None if assistant_logprobs is None else []
    assistant_msg_idx = 0

    for i in range(len(messages)):
        # 2. Use fixed base approach to encode the message and accumulate
        cur_message = messages[i]
        cur_token_ids = encode_messages_subset([cur_message], tokenizer, custom_chat_template)
        response_ids.extend(cur_token_ids)

        # 3. Set loss mask and rollout logprobs.
        # Regardless of the message role, each message is responsible for adding its own generation
        # prompt, and we apply the correct masking.
        if cur_message["role"] == "user":
            # 3.1. For user messages, it is simply zeros
            loss_mask.extend([0] * len(cur_token_ids))
            if assistant_logprobs:
                rollout_logprobs.extend([0.0] * len(cur_token_ids))
        elif cur_message["role"] == "assistant":
            # 3.2. For assistant messages, we need to separate out:
            # 1) generation prompt IDs -- mask is 0
            # 2) tokens actually generated by the assistant (including the EOS) -- mask is 1
            # 3) tokens after the EOS token (the `\n` in Qwen models) -- mask is 0
            prefix_len = len(generation_prompt_ids)
            prefix_matches = cur_token_ids[:prefix_len] == generation_prompt_ids
            if not prefix_matches:
                actual_prefix = cur_token_ids[:prefix_len]
                logger.warning(
                    "Assistant message prefix mismatch (expected {}, got {}). "
                    "Falling back to treating the entire assistant message as generated tokens.",
                    generation_prompt_ids,
                    actual_prefix,
                )
                prefix_len = 0

            if tokenizer.eos_token_id in cur_token_ids:
                last_eos_token_index = len(cur_token_ids) - 1 - cur_token_ids[::-1].index(tokenizer.eos_token_id)
                generated_token_ids = cur_token_ids[prefix_len : last_eos_token_index + 1]
                tokens_after_eos = cur_token_ids[last_eos_token_index + 1 :]
            else:
                generated_token_ids = cur_token_ids[prefix_len:]
                tokens_after_eos = []
            assert prefix_len + len(generated_token_ids) + len(tokens_after_eos) == len(
                cur_token_ids
            ), "The sum of the lengths of the generation prompt IDs, the generated tokens, and the tokens after the EOS token should equal the length of the current token IDs"

            # 3.2.1. Add the generation prompt IDs.
            loss_mask.extend([0] * prefix_len)
            if assistant_logprobs:
                rollout_logprobs.extend([0.0] * prefix_len)

            # 3.2.2. Add what the assistant actually generated
            loss_mask.extend([1] * len(generated_token_ids))
            if assistant_logprobs:
                msg_logprobs = None
                if assistant_msg_idx >= len(assistant_logprobs):
                    logger.warning(
                        "Missing logprobs for assistant message #{} (provided {} lists). "
                        "Proceeding with zeroed logprobs.",
                        assistant_msg_idx + 1,
                        len(assistant_logprobs),
                    )
                else:
                    candidate_logprobs = assistant_logprobs[assistant_msg_idx]

                    # Check if we have the new format with token strings (for LCS alignment)
                    has_token_strings = (
                        len(candidate_logprobs) > 0
                        and isinstance(candidate_logprobs[0], dict)
                        and "token" in candidate_logprobs[0]
                    )

                    if has_token_strings:
                        # New format: use LCS alignment to handle tokenization mismatches
                        msg_logprobs = align_logprobs_with_lcs(
                            generated_token_ids,
                            candidate_logprobs,
                            tokenizer
                        )
                        logger.debug(
                            f"LCS aligned logprobs for assistant message #{assistant_msg_idx + 1}: "
                            f"vLLM tokens={len(candidate_logprobs)}, retokenized={len(generated_token_ids)}"
                        )
                    else:
                        # Legacy format: simple count-based matching (may fail on off-by-one)
                        if isinstance(candidate_logprobs[0], dict):
                            # Dict format but missing token strings - extract just logprobs
                            candidate_logprobs = [lp.get("logprob", 0.0) for lp in candidate_logprobs]

                        if len(candidate_logprobs) != len(generated_token_ids):
                            logger.warning(
                                "Logprob count ({}) does not match token count ({}) for assistant message #{}. "
                                "Token strings not available for LCS alignment. Proceeding with zeroed logprobs.",
                                len(candidate_logprobs),
                                len(generated_token_ids),
                                assistant_msg_idx + 1,
                            )
                        else:
                            msg_logprobs = candidate_logprobs

                rollout_logprobs.extend(msg_logprobs if msg_logprobs is not None else [0.0] * len(generated_token_ids))

            # 3.2.3. Add the tokens after the EOS token.
            loss_mask.extend([0] * len(tokens_after_eos))
            if assistant_logprobs:
                rollout_logprobs.extend([0.0] * len(tokens_after_eos))

            assistant_msg_idx += 1
        else:
            raise ValueError(f"Expected message role to be 'user' or 'assistant', got {cur_message['role']}")

        assert len(loss_mask) == len(response_ids)
        assert len(rollout_logprobs) == len(response_ids) if rollout_logprobs is not None else True

    return response_ids, loss_mask, rollout_logprobs
