"""Stream frozen-policy profiling through the normal rollout and verifier path."""

from copy import deepcopy
from dataclasses import asdict
import asyncio

from loguru import logger

from marinskyrl.pivot import is_context_exclusion
from marinskyrl.pivot_history import ProfileHistory, load_profile_history, profile_sample_outcome

from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.trajectory_runners.trajectory_processing import (
    prepare_trajectory_request,
    select_request_rows,
    validate_trajectory_batch,
)


async def profile_candidates(dataloader, runner, cfg) -> dict[str, float]:
    """Retain each batch before discarding its token arrays; never update the policy."""
    count, passed, excluded, errors = 0, 0.0, 0, 0
    prompt_tokens, response_tokens, attempts, recovered, unresolved = 0, 0, 0, 0, 0
    max_retries = cfg.generator.pivot_profiling_max_retries
    if max_retries < 0:
        raise ValueError("pivot_profiling_max_retries must be nonnegative")
    history = ProfileHistory()
    if cfg.generator.get("pivot_profiling_resume", False):
        retention = cfg.generator.trajectory_retention
        if not (
            retention.enabled
            and retention.required
            and retention.sample_fraction == 1.0
            and retention.max_bytes_per_run is None
            and retention.max_bytes_per_step is None
            and "eval" in retention.phases
        ):
            raise ValueError("Profiling resume requires complete, unbounded, durable evaluation retention")
        model = cfg.trainer.policy.model
        history = await asyncio.to_thread(load_profile_history, retention.output_path, model.path, model.revision)
    previous = {}
    for (source, repetition, attempt), record in history.samples.items():
        previous.setdefault((source, repetition), {})[attempt] = record
    completed, next_attempts = set(), {}
    for slot, saved in previous.items():
        if sorted(saved) != list(range(len(saved))) or len(saved) > max_retries + 1:
            raise ValueError("Incomplete or incompatible retained profiling retry history")
        if any(profile_sample_outcome(saved[i])[1] != "infrastructure_error" for i in range(len(saved) - 1)):
            raise ValueError("Retained profiling retried a completed sample")
        errors += sum(profile_sample_outcome(record)[1] == "infrastructure_error" for record in saved.values())
        score, exclusion = profile_sample_outcome(saved[len(saved) - 1])
        next_attempts[slot] = len(saved)
        if exclusion != "infrastructure_error" or len(saved) == max_retries + 1:
            completed.add(slot)
            if exclusion == "context_window":
                excluded += 1
            elif exclusion == "infrastructure_error":
                unresolved += 1
            else:
                count += 1
                passed += score
                recovered += int(len(saved) > 1)
    prompt_tokens, response_tokens, attempts = history.prompt_tokens, history.response_tokens, history.attempts
    visited = set()
    await runner.start_eval_session(run_name=cfg.trainer.run_name, eval_step=0)
    try:
        for prompts in dataloader:
            request, _ = prepare_trajectory_request(
                prompts,
                cfg.generator.eval_n_samples_per_prompt,
                get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.eval_sampling_params),
                cfg.environment.env_class,
                "eval",
                0,
            )
            pending = []
            for index, (identity, extras) in enumerate(zip(request["trajectory_ids"], request["env_extras"])):
                slot = (extras.get("extra_info", {}).get("source_id", identity.instance_id), identity.repetition_id)
                if slot in visited:
                    raise ValueError("Profiling input repeats a row/sample identity")
                visited.add(slot)
                if slot not in completed:
                    pending.append(index)
            request = select_request_rows(request, pending)
            while pending:
                # Retention includes the attempt number so failures and their replacements
                # remain auditable even when archives are read out of order.
                request = deepcopy(request)
                slots = []
                for identity, extras in zip(request["trajectory_ids"], request["env_extras"]):
                    extra = extras.setdefault("extra_info", {})
                    slot = (extra.get("source_id", identity.instance_id), identity.repetition_id)
                    extra["profiling_attempt"] = next_attempts.get(slot, 0)
                    slots.append(slot)
                batch = await runner.run(request)
                validate_trajectory_batch(len(request["prompts"]), batch)
                attempts += len(request["prompts"])
                prompt_tokens += sum(map(len, batch["prompt_token_ids"]))
                response_tokens += sum(map(len, batch["response_ids"]))
                failed_indices = []
                for index, verdict in enumerate(batch["verification_results"]):
                    attempt = request["env_extras"][index]["extra_info"]["profiling_attempt"]
                    next_attempts[slots[index]] = attempt + 1
                    if verdict is not None and is_context_exclusion(asdict(verdict)):
                        excluded += 1
                        continue
                    if verdict is None or verdict.status != "verified":
                        errors += 1
                        if attempt < max_retries:
                            failed_indices.append(index)
                        else:
                            unresolved += 1
                        continue
                    if verdict.score not in (0, 1):
                        raise ValueError("Profiling requires a binary verifier verdict for every response")
                    passed += verdict.score
                    count += 1
                    recovered += int(attempt > 0)
                if not failed_indices:
                    break
                logger.warning("Retrying {} failed profiling requests", len(failed_indices))
                request = select_request_rows(request, failed_indices)
                pending = failed_indices
            logger.info(
                "Profiled {} actions; mean success={:.4f}; errors={}; context exclusions={}",
                count,
                passed / max(1, count),
                errors,
                excluded,
            )
    finally:
        await runner.stop_eval_session()
    if set(previous) - visited:
        raise ValueError("Retained profiling contains samples outside this candidate split")
    if not count:
        raise ValueError("Profiling produced no responses")
    return {
        "profile/actions": count,
        "profile/attempts": attempts,
        "profile/recovered_actions": recovered,
        "profile/unresolved_actions": unresolved,
        "profile/errors": errors,
        "profile/replayed_actions_discarded": len(history.discarded_record_ids),
        "profile/mean_success": passed / count,
        "profile/context_exclusions": excluded,
        "profile/prompt_tokens": prompt_tokens,
        "profile/generated_tokens": response_tokens,
    }
