"""Submit and score the selected-runtime CatCount GPU nightly."""

import argparse
import json
import re
import time
from dataclasses import replace
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

from iris.cli.connect import connect_controller, open_iris_client, rpc_client
from iris.client.workload import JobStatus
from iris.client.client import IrisClient
from iris.cluster.types import JobName
from iris.resources.state import JobState, TaskState, TERMINAL_JOB_STATES
from iris.rpc import controller_pb2, job_pb2
from rigging.timing import Duration

from ci.marin_nightly.gate import MetricKind, check_log_patterns, check_run, load_spec, parse_metrics

DEADLINE_SECONDS = 1200
POLL_INTERVAL = 10
TERMINAL_CONFIRMATION_INTERVAL = 60
HUB_CONFIG = Path("lib/iris/config/marin.yaml")
INFRASTRUCTURE_REASONS = frozenset({"PodDeleted", "Evicted", "Preempted", "WorkloadEvictedDueToPreempted"})


class Conclusion(StrEnum):
    PASS = "PASS"
    GATE_FAILURE = "GATE_FAILURE"
    INFRASTRUCTURE_FAILURE = "INFRASTRUCTURE_FAILURE"


def infrastructure_reason(statuses: list[JobStatus], log_text: str, deadline_expired: bool) -> str | None:
    if statuses[0].preemption_count > 2:
        return "more than two coordinator preemptions"
    for status in statuses:
        if status.state == JobState.SUCCEEDED:
            continue
        for task in status.tasks:
            if task.state == TaskState.SUCCEEDED:
                continue
            for attempt in task.attempts:
                if attempt.attempt_number != task.current_attempt_number:
                    continue
                if attempt.is_worker_failure or attempt.terminal_reason in INFRASTRUCTURE_REASONS:
                    return f"{task.task_id.to_wire()}: {attempt.terminal_reason or 'worker lost'}"
    if re.search(r"Session name .*does not match persisted value", log_text):
        return "Ray host session collision"
    trained = any(row.kind == MetricKind.TRAIN for row in parse_metrics(log_text))
    if deadline_expired and not trained:
        return "deadline elapsed before training"
    return None


def preflight(marin_root: Path, cluster: str) -> None:
    with connect_controller(config_file=marin_root / HUB_CONFIG) as endpoint:
        with rpc_client(endpoint.url, credentials=endpoint.credentials) as client:
            peers = client.list_peers(controller_pb2.Controller.ListPeersRequest())
            available = {
                peer.peer_id: [dict(backend.availability.amounts) for backend in peer.backends]
                for peer in peers.peers
                if peer.reachable
            }
            print("CAT_COUNT_CAPACITY " + json.dumps(available), flush=True)
            if cluster == "cw-us-east-02a" and sum(row.get("h100", 0) for row in available.get(cluster, [])) < 16:
                raise RuntimeError("East reports fewer than 16 available H100s")
            state = client.get_scheduler_state(controller_pb2.Controller.GetSchedulerStateRequest())
            print(
                "CAT_COUNT_HUB_BUDGET "
                + json.dumps(
                    [
                        {
                            "user": row.user_id,
                            "limit": row.budget_limit,
                            "spent": row.budget_spent,
                            "effective_band": job_pb2.PriorityBand.Name(row.effective_band),
                        }
                        for row in state.user_budgets
                    ]
                ),
                flush=True,
            )


def workload_statuses(client: IrisClient, job_name: JobName) -> list[JobStatus]:
    summaries = [client.job_status(job_name), *client.list_jobs(prefix=job_name.to_wire() + "/")]
    return [replace(status, tasks=tuple(client.list_tasks(status.job_id))) for status in summaries]


def cancel_owned_job(client: IrisClient, job_name: JobName, name: str) -> None:
    status = client.job_status(job_name)
    if status.job_id != job_name or status.name != name:
        raise RuntimeError("recorded CatCount job identity differs from Iris")
    if status.state not in TERMINAL_JOB_STATES:
        client.cancel_job(job_name)
        print(f"CAT_COUNT_CANCEL id={job_name.to_wire()} preemptions={status.preemption_count}", flush=True)


def cancel_recorded_jobs(marin_root: Path, receipt_path: Path) -> None:
    with open_iris_client(config_file=marin_root / HUB_CONFIG, workspace=marin_root) as client:
        for receipt in json.loads(receipt_path.read_text()):
            job_name = JobName.from_wire(receipt["id"])
            cancel_owned_job(client, job_name, receipt["name"])


def wait_for_job(client: IrisClient, job_name: JobName, name: str, started: float) -> tuple[float, bool]:
    terminal_observed = None
    deadline_expired = False
    cancelled = False
    elapsed = 0.0
    while True:
        status = client.job_status(job_name)
        deadline_expired = deadline_expired or time.monotonic() - started >= DEADLINE_SECONDS
        if status.state in TERMINAL_JOB_STATES:
            if terminal_observed is not None and terminal_observed[0] == status.state:
                if time.monotonic() - terminal_observed[1] >= TERMINAL_CONFIRMATION_INTERVAL:
                    break
            else:
                terminal_observed = (status.state, time.monotonic())
                elapsed = time.monotonic() - started
            time.sleep(POLL_INTERVAL)
            continue
        if not cancelled and (deadline_expired or status.preemption_count > 2):
            cancel_owned_job(client, job_name, name)
            cancelled = True
        time.sleep(POLL_INTERVAL)
    return elapsed, deadline_expired


def run_attempt(args: argparse.Namespace, attempt: int) -> tuple[Conclusion, float]:
    from fray.iris_backend import FrayIrisClient
    from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment

    name = f"{args.job_name}-a{attempt}"
    version = datetime.now(timezone.utc).strftime("%Y.%m.%d") + f".{time.time_ns()}"
    request = JobRequest(
        name=name,
        entrypoint=Entrypoint.from_binary(
            "python",
            [
                "-m",
                "experiments.post_training.cat_count_canary.nightly",
                "--preset",
                "gate",
                "--lane",
                "async",
                "--cluster",
                args.cluster,
                "--seed",
                "17",
                "--runtime-commit",
                args.runtime_commit,
                "--version",
                version,
                "--job-timeout-seconds",
                str(DEADLINE_SECONDS),
                "--set",
                "trainer.logger=console",
                "--run",
            ],
        ),
        resources=ResourceConfig.with_cpu(cpu=4, ram="16GB", disk="8GB", target_cluster=args.cluster),
        environment=create_environment(workspace=str(args.marin_root), extras=["cpu"]),
        priority=job_pb2.PRIORITY_BAND_INTERACTIVE,
        max_retries_preemption=2,
        timeout=Duration.from_seconds(DEADLINE_SECONDS),
    )
    preflight(args.marin_root, args.cluster)
    started = time.monotonic()
    with open_iris_client(config_file=args.marin_root / HUB_CONFIG, workspace=args.marin_root) as client:
        handle = FrayIrisClient.from_iris_client(client).submit(request, adopt_existing=False)
        job_name = JobName.from_wire(handle.job_id)
        receipt_path = args.log.with_suffix(".jobs.json")
        receipts = json.loads(receipt_path.read_text()) if receipt_path.exists() else []
        receipts.append({"id": handle.job_id, "name": name})
        receipt_path.write_text(json.dumps(receipts))
        print(f"CAT_COUNT_JOB attempt={attempt} id={handle.job_id}", flush=True)
        elapsed, deadline_expired = wait_for_job(client, job_name, name, started)
        statuses = workload_statuses(client, job_name)
        entries = client.job(job_name).logs(max_lines=60000, tail=False)
        log_text = "\n".join(entry.data for entry in entries)
        log_path = args.log.with_name(f"{args.log.stem}-a{attempt}{args.log.suffix}")
        log_path.write_text(log_text + "\n")
        print(log_text, flush=True)
    reason = infrastructure_reason(statuses, log_text, deadline_expired)
    if reason is not None:
        print(f"CAT_COUNT_INFRASTRUCTURE_FAILURE attempt={attempt} seconds={elapsed:.2f} reason={reason}", flush=True)
        return Conclusion.INFRASTRUCTURE_FAILURE, elapsed
    spec = load_spec(args.spec)
    failures = check_run(parse_metrics(log_text), spec, elapsed) + check_log_patterns(log_text, spec)
    if failures or any(status.state != JobState.SUCCEEDED for status in statuses):
        for failure in failures:
            print(
                f"CAT_COUNT_GATE_FAILURE {failure.kind} {failure.stream}/{failure.metric}: {failure.message}",
                flush=True,
            )
        if not failures:
            print("CAT_COUNT_GATE_FAILURE launcher did not complete successfully", flush=True)
        return Conclusion.GATE_FAILURE, elapsed
    args.log.write_text(log_text + "\n")
    print(f"OK against {args.spec}", flush=True)
    return Conclusion.PASS, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marin-root", type=Path, required=True)
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--cluster", choices=("cw-rno2a", "cw-us-east-02a"), required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for attempt in (1, 2):
        conclusion, elapsed = run_attempt(args, attempt)
        results.append({"attempt": attempt, "conclusion": conclusion, "seconds": elapsed})
        if conclusion != Conclusion.INFRASTRUCTURE_FAILURE:
            break
    print("CAT_COUNT_NIGHTLY_RESULT " + json.dumps({"attempts": results, "conclusion": conclusion}), flush=True)
    return 0 if conclusion == Conclusion.PASS else 2 if conclusion == Conclusion.INFRASTRUCTURE_FAILURE else 1


if __name__ == "__main__":
    raise SystemExit(main())
