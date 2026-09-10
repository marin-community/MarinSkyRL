"""Version-dispatched endpoint metrics; optimization rewards remain unmodified."""

import math
from collections import defaultdict

from skyrl_gym.envs.thinking_contract import (
    ACCEPTED_STOPS,
    THINKING_CONTRACT_VERSION,
    native_answer_reward,
    post_thinking_segment,
)
from skyrl_train.evaluation_contract import CONTRACTS, evaluation_contract_metrics

from skyrl_train.non_agentic_evaluation import METRIC_VERSION


def endpoint_contract_metrics(envs, extras, batch, tokenizer, parser_protocol):
    count = len(extras)
    stops = batch.get("stop_reasons", [None] * count)
    if parser_protocol is None:
        return evaluation_contract_metrics(
            envs,
            extras,
            [
                tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                for tokens in batch["response_ids"]
            ],
            batch["rewards"],
            stops,
        )
    if parser_protocol != THINKING_CONTRACT_VERSION:
        raise ValueError("Unknown endpoint parser protocol")
    contracts = batch.get("non_agentic_contract")
    if contracts is None or len(contracts) != count:
        raise ValueError("Corrected endpoint metrics require every original contract receipt")
    columns = [envs, batch["response_ids"], batch["prompt_token_ids"], batch["rewards"], stops]
    if any(len(values) != count for values in columns):
        raise ValueError("Endpoint evidence has inconsistent row counts")
    unshaped = batch.get("unshaped_rewards")
    if unshaped is not None and len(unshaped) != count:
        raise ValueError("Endpoint unshaped reward count differs")
    populations = defaultdict(list)
    for index, (env, extra, tokens, prompt, optimization, stop, contract) in enumerate(
        zip(
            envs,
            extras,
            batch["response_ids"],
            batch["prompt_token_ids"],
            batch["rewards"],
            stops,
            contracts,
            strict=True,
        )
    ):
        if env not in CONTRACTS or extra.get("extra_info", {}).get("contract") != CONTRACTS[env]:
            raise ValueError("Corrected endpoint has an unknown source answer contract")
        if contract.get("metric_protocol") != METRIC_VERSION:
            raise ValueError("Unknown corrected endpoint metric protocol")
        if contract["parser_protocol"] != parser_protocol:
            raise ValueError("Mixed endpoint parser versions")
        gold = extra["reward_model"]["ground_truth"]
        if gold != extra["reward_spec"]["ground_truth"]:
            raise ValueError("Endpoint gold channels differ")
        segment, boundary = post_thinking_segment(tokenizer.backend_tokenizer, prompt, tokens)
        verifier = native_answer_reward(env, gold, "" if segment is None else segment)
        correct = int(verifier >= 1) if env == "reasoning_gym" else int(verifier == 1)
        completed = correct * int(stop in ACCEPTED_STOPS)
        if (
            contract["boundary_status"] != boundary
            or contract["verifier_reward"] != verifier
            or contract["contract_correct"] != correct
            or contract["score_contract_completed"] != completed
        ):
            raise ValueError("Endpoint contract differs from the actual token-delimited verifier")
        if unshaped is not None and unshaped[index] != verifier:
            raise ValueError("Endpoint pre-shaping reward differs from the corrected verifier")
        raw = sum(optimization) if isinstance(optimization, list) else optimization
        legacy = contract["legacy_full_text_reward"]
        if not all(math.isfinite(value) for value in (raw, verifier, legacy)):
            raise ValueError("Nonfinite endpoint reward")
        components = batch.get("reward_shaping_components")
        if components is not None and len(components) != count:
            raise ValueError("Shaping component count differs")
        penalty = sum(components[index].values()) if components is not None else 0.0
        # Only this arithmetic reconstruction tolerates floating-point addition; raw is emitted verbatim.
        if not math.isclose(raw, verifier + penalty, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Optimization reward differs from retained shaping components")
        values = (float(correct), float(completed), verifier, raw, legacy)
        source = (extra.get("data_source") or "unknown").replace("/", "_")
        if source == "all":
            raise ValueError("Dataset name collides with aggregate metrics")
        populations["all"].append(values)
        populations[source].append(values)
    names = (
        "contract_correct",
        "contract_completed",
        "corrected_verifier_reward",
        "score_contract",
        "legacy_full_text_reward_diagnostic",
    )
    return {
        f"eval/{source}/{name}": sum(row[index] for row in rows) / len(rows)
        for source, rows in populations.items()
        for index, name in enumerate(names)
    }
