"""The driver plans from what the ranks report and refuses a sync that did not land exactly the plan."""

import asyncio
from dataclasses import asdict
import threading
from types import SimpleNamespace

import pytest

from marinskyrl.inference_placement import InferenceReplicaPlacement, InferenceWorkerPlacement
from skyrl_train.weight_sync.expert_block.driver import ExpertBlockSync, plan_from_inventories
from skyrl_train.weight_sync.expert_block.schedule import receiver_participant, to_wire
from skyrl_train.weight_sync.expert_block.stream import InstallReport
from tests.cpu.weight_sync.expert_block.test_schedule import (
    EP,
    LAYERS_BY_PP,
    NUM_EXPERTS,
    dense_of,
    entries_of,
    receivers,
    trainers,
)

MODEL = {"num_experts": NUM_EXPERTS, "hidden_size": 3, "intermediate_size": 2}


def policy_inventories():
    return [
        {
            "trainer": to_wire(trainer),
            "expert_parallel_size": EP,
            "layers": list(LAYERS_BY_PP[trainer.pp]),
            "model": MODEL,
            "experts": [to_wire(item) for item in entries_of(trainer)],
            "dense": [to_wire(item) for item in dense_of(trainer)],
        }
        for trainer in trainers()
    ]


def installed_dense():
    return {
        "model.norm.weight": [[8], "bfloat16"],
        "model.layers.0.mlp.router.weight": [[4, 3], "float32"],
        "model.layers.2.mlp.router.weight": [[4, 3], "float32"],
    }


def receiver_inventory(row):
    return {
        "gpu_uuid": f"GPU-{row.replica}-{row.ep}",
        "ep_rank": row.ep,
        "expert_parallel_size": EP,
        "pp_rank": 0,
        "pp_size": 1,
        "layers": [0, 1, 2],
        "model": {**MODEL, "num_hidden_layers": 3},
        "dense": installed_dense(),
    }


def placement(row):
    worker = InferenceWorkerPlacement(
        f"host-{row.replica}", f"GPU-{row.replica}-{row.ep}", row.ep, EP, row.ep, EP, row.ep, EP
    )
    return InferenceReplicaPlacement(row.replica, f"node-{row.replica}", row.ep, worker, 1 + row.rank)


def test_plan_pairs_each_receiver_with_its_verified_gpu():
    schedule, participants = plan_from_inventories(
        policy_inventories(), [(receiver_inventory(row), placement(row)) for row in receivers()]
    )
    assert participants == {
        f"GPU-{row.replica}-{row.ep}": receiver_participant(schedule.trainer_count, row) for row in receivers()
    }
    assert schedule.trainer_count == len(trainers())


@pytest.mark.parametrize(
    "change,error",
    [
        (lambda report: report.update(expert_parallel_size=3), "do not split evenly|native ranks"),
        (lambda report: report["dense"].pop("model.norm.weight"), "Dense weights differ"),
        (lambda report: report.update(pp_size=2), "stages are incomplete"),
        (lambda report: report["dense"].update({"model.norm.weight": [[9], "bfloat16"]}), "cover 8 of 9"),
        (
            lambda report: report["dense"].update({"model.norm.weight": [[8], "float32"]}),
            "bfloat16 on the trainer and float32",
        ),
        (lambda report: report["model"].update(hidden_size=5), "differs from trainer model"),
    ],
)
def test_receivers_that_do_not_match_the_trainers_are_refused(change, error):
    rows = [(receiver_inventory(row), placement(row)) for row in receivers()]
    for report, _ in rows:
        change(report)
    with pytest.raises(
        error if isinstance(error, type) else ValueError, match=None if isinstance(error, type) else error
    ):
        plan_from_inventories(policy_inventories(), rows)


def test_a_receiver_whose_report_disagrees_with_its_verified_gpu_is_refused():
    rows = [(receiver_inventory(row), placement(row)) for row in receivers()]
    rows[0][0]["ep_rank"] = 1 - rows[0][0]["ep_rank"]
    with pytest.raises(ValueError, match="does not match its verified placement"):
        plan_from_inventories(policy_inventories(), rows)


def test_receivers_that_disagree_with_each_other_are_refused():
    rows = [(receiver_inventory(row), placement(row)) for row in receivers()]
    rows[1][0]["dense"].pop("model.norm.weight")
    with pytest.raises(ValueError, match="Receivers of stage 0 disagree"):
        plan_from_inventories(policy_inventories(), rows)


class FakeRanks:
    """A policy model and an engine client that answer every RPC from the plan itself."""

    def __init__(self, schedule=None):
        self.receiver_rows = receivers()
        self.engines = [SimpleNamespace(worker_placements=[placement(row)]) for row in self.receiver_rows]
        self.generation_paused_event = threading.Event()
        self.generation_paused_event.set()
        self.schedule = schedule
        self.report_changes = {}
        self.calls = []

    def reply(self, value):
        future = asyncio.get_event_loop().create_future()
        future.set_result(value)
        return future

    def report(self, participant, version):
        expected = dict(self.schedule.receiver_bytes)
        if participant in expected:
            report = InstallReport(
                participant,
                version,
                dict(self.schedule.receiver_experts)[participant],
                expected[participant],
                0.1,
                expert_seconds=0.07,
                dense_seconds=0.03,
            )
        else:
            sent = sum(item.entry.nbytes for item in self.schedule.experts if item.root == participant)
            sent += sum(item.source.nbytes for item in self.schedule.dense if item.root == participant)
            report = InstallReport(participant, version, 0, sent, 0.1, expert_seconds=0.02, dense_seconds=0.08)
        return {**asdict(report), **self.report_changes.get(participant, {})}

    # policy model
    def async_run_ray_method(self, dispatch, method_name, method, *args):
        assert (dispatch, method_name) == ("pass_through", "expert_block_rpc")
        self.calls.append(("policy", method))
        replies = []
        for trainer, inventory in zip(trainers(), policy_inventories(), strict=True):
            if method == "inventory":
                replies.append(inventory)
            elif method == "trainer_init":
                replies.append({"participant": trainer.rank, "warmup_seconds": {}})
            elif method == "send_weights":
                replies.append(self.report(trainer.rank, args[0]["version"]))
            else:
                replies.append(None)
        return [self.reply(value) for value in replies]

    # engine client: one list per engine, one reply per worker (one worker per engine here)
    async def expert_block_rpc(self, method, *args):
        self.calls.append(("engines", method))
        replies = []
        for row in self.receiver_rows:
            if method == "inventory":
                replies.append([receiver_inventory(row)])
            elif method == "init_transfer_engine":
                replies.append(
                    [{"participant": args[0]["participants"][f"GPU-{row.replica}-{row.ep}"], "warmup_seconds": {}}]
                )
            elif method == "receive_weights":
                replies.append([self.report(receiver_participant(len(trainers()), row), args[0]["version"])])
            else:
                replies.append([None])
        return replies


@pytest.fixture
def local_store(monkeypatch):
    monkeypatch.setattr("skyrl_train.weight_sync.expert_block.groups.ray.util.get_node_ip_address", lambda: "127.0.0.1")


def prepared(ranks):
    sync = ExpertBlockSync(policy_model=ranks, inference_engine_client=ranks, timeout_seconds=30)
    asyncio.run(sync.prepare())
    ranks.schedule = sync.schedule
    return sync


def test_prepare_then_sync_accepts_reports_that_match_the_plan(local_store):
    ranks = FakeRanks()
    sync = prepared(ranks)
    timings = asyncio.run(sync.sync(3))
    assert timings.receiver_seconds == 0.1
    # The slowest participant of each phase, whichever side it is on.
    assert (timings.expert_seconds, timings.dense_seconds) == (0.07, 0.08)
    assert set(timings.as_metrics()) == {
        f"expert_block_sync/{name}"
        for name in ("install_seconds", "policy_seconds", "receiver_seconds", "expert_seconds", "dense_seconds")
    }
    assert [call for call in ranks.calls if call[1] != "inventory"] == [
        ("policy", "trainer_init"),
        ("engines", "init_transfer_engine"),
        ("policy", "send_weights"),
        ("engines", "receive_weights"),
    ]
    asyncio.run(sync.close())
    assert ranks.calls[-2:] == [("policy", "shutdown"), ("engines", "shutdown")]


@pytest.mark.parametrize(
    "participant_offset,change,error",
    [
        (0, {"expert_matrices": 5}, "landed 5 expert matrices"),
        (0, {"wire_bytes": 1}, "and 1 bytes"),
        (0, {"version": 2}, "installed version 2, expected 3"),
        (None, {"wire_bytes": 1}, "sent 1 bytes"),
    ],
)
def test_a_report_that_deviates_from_the_plan_fails_the_sync(local_store, participant_offset, change, error):
    ranks = FakeRanks()
    sync = prepared(ranks)
    if participant_offset is None:
        participant = next(item.root for item in sync.schedule.experts)
    else:
        participant = receiver_participant(sync.schedule.trainer_count, receivers()[participant_offset])
    ranks.report_changes[participant] = change
    with pytest.raises(RuntimeError, match=error):
        asyncio.run(sync.sync(3))


def test_a_missing_receiver_fails_the_sync(local_store):
    ranks = FakeRanks()
    sync = prepared(ranks)
    ranks.receiver_rows = ranks.receiver_rows[:-1]
    with pytest.raises(RuntimeError, match="Not every planned receiver reported"):
        asyncio.run(sync.sync(3))


def test_sync_requires_paused_generation(local_store):
    ranks = FakeRanks()
    sync = prepared(ranks)
    ranks.generation_paused_event.clear()
    with pytest.raises(RuntimeError, match="paused"):
        asyncio.run(sync.sync(3))


def test_prepare_requires_node_local_placement(local_store):
    ranks = FakeRanks()
    ranks.engines[1].worker_placements = None
    sync = ExpertBlockSync(policy_model=ranks, inference_engine_client=ranks, timeout_seconds=30)
    with pytest.raises(ValueError, match="inference_engine_node_local"):
        asyncio.run(sync.prepare())
