"""Bind Ray rendezvous to actual Iris task-attempt membership."""

import hashlib
import json
import time
from dataclasses import dataclass

from iris.client.client import IrisClient
from iris.client.workload import TaskStatus
from iris.cluster.types import JobName, TaskAttempt
from iris.resources.state import TaskState


class IncompleteMembership(RuntimeError):
    """The controller has not assigned a complete live gang yet."""


@dataclass(frozen=True)
class Member:
    rank: int
    attempt: int
    uid: str


def membership_snapshot(
    rows: list[TaskStatus], job: str, num_tasks: int, own_task: str, own_uid: str
) -> tuple[Member, ...]:
    """Return complete current membership, rejecting an obsolete caller."""
    caller = TaskAttempt.from_wire(own_task)
    if caller.attempt_id is None or not own_uid:
        raise ValueError("Native rendezvous requires an explicit attempt and UID")
    if str(caller.task_id).rsplit("/", 1)[0] != job:
        raise ValueError("Caller does not belong to the requested Iris job")
    members = []
    seen = set()
    uids = set()
    for row in rows:
        task = str(row.task_id)
        parent, rank_text = task.rsplit("/", 1)
        rank = int(rank_text)
        if parent != job or rank in seen or not 0 <= rank < num_tasks:
            raise ValueError("Controller returned unexpected rendezvous membership")
        seen.add(rank)
        attempts = [a for a in row.attempts if a.attempt_number == row.current_attempt_number]
        if len(attempts) != 1 or not attempts[0].attempt_uid:
            raise IncompleteMembership("Current attempt identity is unavailable")
        attempt = attempts[0]
        if attempt.attempt_uid in uids:
            raise ValueError("Controller returned duplicate task-attempt UIDs")
        uids.add(attempt.attempt_uid)
        if task == str(caller.task_id) and (
            attempt.attempt_number != caller.attempt_id or attempt.attempt_uid != own_uid
        ):
            raise RuntimeError("Caller is no longer the current Iris task attempt")
        active = {TaskState.ASSIGNED, TaskState.BUILDING, TaskState.RUNNING}
        if row.state not in active or attempt.state not in active:
            raise IncompleteMembership("A gang member is not active")
        members.append(Member(rank, attempt.attempt_number, attempt.attempt_uid))
    if seen != set(range(num_tasks)):
        raise IncompleteMembership("Not every expected rank has a current attempt")
    own_rank = int(str(caller.task_id).rsplit("/", 1)[1])
    if not any(m.rank == own_rank and m.uid == own_uid for m in members):
        raise RuntimeError("Caller is absent from the current Iris gang")
    return tuple(sorted(members, key=lambda member: member.rank))


@dataclass
class RendezvousGuard:
    client: IrisClient
    job: str
    num_tasks: int
    own_task: str
    own_uid: str
    members: tuple[Member, ...]

    def validate(self) -> None:
        current = membership_snapshot(
            self.client.list_tasks(JobName.from_wire(self.job)),
            self.job,
            self.num_tasks,
            self.own_task,
            self.own_uid,
        )
        if current != self.members:
            raise RuntimeError("Iris gang membership changed during Ray startup")

    def diagnostic(self) -> dict:
        return {
            "job_id": self.job,
            "members": [{"rank": m.rank, "attempt_id": m.attempt, "attempt_uid": m.uid} for m in self.members],
        }

    def namespace(self, root: str) -> str:
        digest = hashlib.sha256(
            json.dumps(self.diagnostic(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return root.rstrip("/") + "/members-" + digest


def resolve_membership(
    controller_address: str, own_task: str, own_uid: str, num_tasks: int, timeout: float
) -> RendezvousGuard:
    """Read complete membership with bounded RPCs before any Ray startup."""
    job = own_task.rsplit("/", 1)[0]
    client = IrisClient.in_cluster(controller_address, timeout_ms=5000)
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            try:
                members = membership_snapshot(
                    client.list_tasks(JobName.from_wire(job)),
                    job,
                    num_tasks,
                    own_task,
                    own_uid,
                )
                return RendezvousGuard(client, job, num_tasks, own_task, own_uid, members)
            except IncompleteMembership:
                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        raise TimeoutError("Complete current Iris gang membership was not available before Ray startup")
    except BaseException:
        client.shutdown()
        raise
