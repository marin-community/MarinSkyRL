"""Exercise stale-attempt isolation through the real object-store helpers."""

from pathlib import Path
import sys

import pytest
from iris.client.workload_codec import task_status_from_proto
from iris.rpc import job_pb2

from cloud.iris import ray_membership, runtime_bundle, task_runtime
from cloud.iris.ray_membership import IncompleteMembership, RendezvousGuard, membership_snapshot


def task(rank, attempt, uid):
    return task_status_from_proto(
        job_pb2.TaskStatus(
            task_id=f"/owner/job/{rank}",
            state=job_pb2.TASK_STATE_RUNNING,
            current_attempt_id=attempt,
            attempts=[
                job_pb2.TaskAttempt(
                    attempt_id=attempt,
                    attempt_uid=uid,
                    state=job_pb2.TASK_STATE_RUNNING,
                )
            ],
        )
    )


class Controller:
    def __init__(self, rows):
        self.rows = rows
        self.closed = False

    def list_tasks(self, job):
        return self.rows

    def shutdown(self):
        self.closed = True


def guard(client, rank, attempt, uid):
    own = f"/owner/job/{rank}:{attempt}"
    members = membership_snapshot(client.rows, "/owner/job", 2, own, uid)
    return RendezvousGuard(client, "/owner/job", 2, own, uid, members)


def test_unequal_counters_share_current_namespace_and_ignore_fresh_old_writer(tmp_path):
    old = Controller([task(0, 0, "head-old"), task(1, 0, "worker-old")])
    # A rank can have an extra failed attempt before its sibling starts.
    current = Controller([task(0, 2, "head-new"), task(1, 1, "worker-new")])
    old_head = guard(old, 0, 0, "head-old")
    head = guard(current, 0, 2, "head-new")
    worker = guard(current, 1, 1, "worker-new")
    root = str(tmp_path)
    Path(head.namespace(root)).mkdir(exist_ok=True)
    Path(old_head.namespace(root)).mkdir(exist_ok=True)
    assert head.namespace(root) == worker.namespace(root)
    task_runtime.write_rendezvous(head.namespace(root), "10.0.0.2", 6379, "new", head)
    # Even a later timestamp cannot move the old writer into current membership.
    task_runtime.write_rendezvous(old_head.namespace(root), "10.0.0.1", 6379, "old", old_head)
    task_runtime.write_rendezvous(root, "10.0.0.9", 6379, "legacy-old")
    payload = task_runtime.poll_rendezvous(worker.namespace(root), 1, membership=worker)
    assert (payload.head_ip, payload.gang_epoch) == ("10.0.0.2", "new")
    task_runtime.write_head_result(old_head.namespace(root), "old")
    assert not task_runtime.head_succeeded(worker.namespace(root), "old")


@pytest.mark.parametrize("rows", [[task(0, 0, "head")], [task(0, 0, "head"), task(1, 0, "")]])
def test_incomplete_membership_never_selects_a_namespace(rows):
    with pytest.raises(IncompleteMembership):
        membership_snapshot(rows, "/owner/job", 2, "/owner/job/0:0", "head")


def test_stale_local_attempt_is_rejected_even_when_its_rank_is_active():
    with pytest.raises(RuntimeError, match="no longer"):
        membership_snapshot([task(0, 2, "new"), task(1, 1, "worker")], "/owner/job", 2, "/owner/job/0:1", "old")


def test_membership_change_blocks_publication_and_worker_poll(tmp_path):
    client = Controller([task(0, 0, "head"), task(1, 0, "worker")])
    head = guard(client, 0, 0, "head")
    namespace = head.namespace(str(tmp_path))
    client.rows = [task(0, 0, "head"), task(1, 1, "worker-new")]
    with pytest.raises(RuntimeError, match="changed"):
        task_runtime.write_rendezvous(namespace, "10.0.0.1", 6379, "old", head)
    with pytest.raises(RuntimeError, match="changed"):
        task_runtime.poll_rendezvous(namespace, 1, membership=head)
    assert not list(tmp_path.rglob(task_runtime.RENDEZVOUS_FILENAME))


def test_incomplete_membership_times_out_and_closes_client(monkeypatch):
    client = Controller([task(0, 0, "head")])
    monkeypatch.setattr(ray_membership.IrisClient, "in_cluster", lambda *a, **kw: client)
    ticks = iter([0, 0, 0, 2])
    monkeypatch.setattr(ray_membership.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(ray_membership.time, "sleep", lambda _: None)
    with pytest.raises(TimeoutError, match="Complete current"):
        ray_membership.resolve_membership("unused", "/owner/job/0:0", "head", 2, 1)
    assert client.closed


def test_bootstrap_binds_current_members_before_staging_and_rejects_later_change(tmp_path, monkeypatch):
    client = Controller([task(0, 2, "head"), task(1, 1, "worker")])
    monkeypatch.setattr(ray_membership.IrisClient, "in_cluster", lambda *a, **kw: client)
    monkeypatch.setenv("IRIS_TASK_ID", "/owner/job/0:2")
    monkeypatch.setenv("IRIS_ATTEMPT_UID", "head")
    monkeypatch.setenv("IRIS_NUM_TASKS", "2")
    monkeypatch.setenv("IRIS_CONTROLLER_ADDRESS", "unused")
    (tmp_path / runtime_bundle.BUNDLE_IDENTITY_FILE).write_text('{"launcher_commit":"test-membership","files":[]}')
    monkeypatch.setattr(runtime_bundle, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["runtime", "--rendezvous-dir", str(tmp_path), "--", "unused-driver"])
    for name in ("_print_env_snapshot", "_pin_boto3_s3_addressing_style", "ensure_fr_dump_dir"):
        monkeypatch.setattr(task_runtime, name, lambda: None)
    monkeypatch.setattr(task_runtime, "pin_socket_ifname", lambda: None)
    monkeypatch.setattr(task_runtime, "export_telemetry_environment", lambda _: None)

    def start_head(args, train_argv, interface, membership):
        Path(args.rendezvous_dir).mkdir()
        client.rows = [task(0, 2, "head"), task(1, 2, "worker-next")]
        task_runtime.write_rendezvous(args.rendezvous_dir, "10.0.0.1", 6379, "old", membership)
        raise AssertionError("Changed membership must fail before starting a Ray head")

    monkeypatch.setattr(task_runtime, "run_head", start_head)
    with pytest.raises(RuntimeError, match="changed"):
        task_runtime.main()
    assert client.closed
    assert not list(tmp_path.rglob(task_runtime.RENDEZVOUS_FILENAME))
