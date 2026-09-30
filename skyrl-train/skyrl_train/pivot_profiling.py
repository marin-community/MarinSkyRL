"""Stream frozen-policy profiling through the normal rollout and verifier path."""

from copy import deepcopy
from dataclasses import asdict

from loguru import logger

from marinskyrl.pivot import is_context_exclusion

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
            for attempt in range(max_retries + 1):
                # Retention includes the attempt number so failures and their replacements
                # remain auditable even when archives are read out of order.
                request = deepcopy(request)
                for extras in request["env_extras"]:
                    extras.setdefault("extra_info", {})["profiling_attempt"] = attempt
                batch = await runner.run(request)
                validate_trajectory_batch(len(request["prompts"]), batch)
                attempts += len(request["prompts"])
                prompt_tokens += sum(map(len, batch["prompt_token_ids"]))
                response_tokens += sum(map(len, batch["response_ids"]))
                failed_indices = []
                for index, verdict in enumerate(batch["verification_results"]):
                    if verdict is not None and is_context_exclusion(asdict(verdict)):
                        excluded += 1
                        continue
                    if verdict is None or verdict.status != "verified":
                        errors += 1
                        failed_indices.append(index)
                        continue
                    if verdict.score not in (0, 1):
                        raise ValueError("Profiling requires a binary verifier verdict for every response")
                    passed += verdict.score
                    count += 1
                    recovered += int(attempt > 0)
                if not failed_indices or attempt == max_retries:
                    unresolved += len(failed_indices)
                    break
                logger.warning("Retrying {} failed profiling requests (attempt {})", len(failed_indices), attempt + 1)
                request = select_request_rows(request, failed_indices)
            logger.info(
                "Profiled {} actions; mean success={:.4f}; errors={}; context exclusions={}",
                count,
                passed / max(1, count),
                errors,
                excluded,
            )
    finally:
        await runner.stop_eval_session()
    if not count:
        raise ValueError("Profiling produced no responses")
    return {
        "profile/actions": count,
        "profile/attempts": attempts,
        "profile/recovered_actions": recovered,
        "profile/unresolved_actions": unresolved,
        "profile/errors": errors,
        "profile/mean_success": passed / count,
        "profile/context_exclusions": excluded,
        "profile/prompt_tokens": prompt_tokens,
        "profile/generated_tokens": response_tokens,
    }
