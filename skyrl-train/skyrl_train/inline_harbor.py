"""Run pinned Harbor validation jobs from the trainer's installed runtime."""

import argparse
import asyncio
import json
import math
import signal
from pathlib import Path

from harbor.job import Job
from harbor.models.job.config import JobConfig
from harbor.models.job.result import JobResult


def validation_metrics(result: JobResult, config: JobConfig, reward_key: str) -> dict[str, float]:
    """Count only completed, verified trials; expose missing scores separately."""
    expected = {str(task.get_task_id()) for task in config.tasks}
    observed = [str(trial.task_id) for trial in result.trial_results]
    if len(expected) != len(config.tasks) or config.n_attempts != 1:
        raise ValueError("Inline validation requires unique tasks and one attempt per task")
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("Harbor validation returned missing, duplicate, or unexpected tasks")
    scores = []
    for trial in result.trial_results:
        if trial.exception_info is not None and trial.exception_info.exception_type != "AgentTimeoutError":
            continue
        verifier = trial.verifier_result
        if verifier is None or verifier.rewards is None or reward_key not in verifier.rewards:
            continue
        score = float(verifier.rewards[reward_key])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid normalized reward for {trial.task_name}: {score}")
        scores.append(score)
    return {
        "num_attempted": float(len(expected)),
        "num_scored": float(len(scores)),
        "reward_sum": math.fsum(scores),
        "avg_verifier_score": math.fsum(scores) / len(scores) if scores else 0.0,
    }


async def run_validation(spec: dict) -> dict[str, dict[str, float]]:
    """Persist full Harbor traces at each job's configured durable jobs_dir."""
    metrics = {}
    for cohort in spec["cohorts"]:
        config = JobConfig.model_validate(cohort["job"])
        if config.datasets or not config.tasks or len(config.agents) != 1:
            raise ValueError("Inline validation requires explicit pinned tasks and exactly one agent")
        if any(not task.git_url or not task.git_commit_id for task in config.tasks):
            raise ValueError("Inline validation task sources must have immutable Git revisions")
        name = cohort["name"]
        if name in metrics:
            raise ValueError(f"Duplicate validation cohort: {name}")
        job = await Job.create(config)
        result = await job.run()
        metrics[name] = validation_metrics(result, config, cohort["reward_key"])
    return metrics


async def run_cancellable_validation(spec: dict) -> dict[str, dict[str, float]]:
    loop = asyncio.get_running_loop()
    task = asyncio.create_task(run_validation(spec))
    loop.add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        return await task
    finally:
        loop.remove_signal_handler(signal.SIGTERM)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = asyncio.run(run_cancellable_validation(json.loads(args.spec.read_text())))
    args.output.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
