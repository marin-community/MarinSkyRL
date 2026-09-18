"""The training driver's side of expert-block weight sync.

``prepare`` runs once: it collects what every trainer rank owns and what every
receiver holds, builds the schedule, hosts the rendezvous store and has every
participant create its groups. ``sync`` runs after each update with generation
paused: every participant runs the stream and the driver fails closed if any
participant raised, any planned receiver did not report, or a report names a
different version. Each trainer rank refuses to send unless the update named
is the one it just finished and its parameters are the storage the plan was
built on; each receiver refuses unless its parameters are still the storage
the groups were bound to. Byte-exactness of what landed is the opt-in gate's
job, not this path's.
"""

import asyncio
from collections import Counter
from dataclasses import asdict, dataclass
import time

from skyrl_train.weight_sync.expert_block.groups import RendezvousStore
from skyrl_train.weight_sync.expert_block.schedule import (
    DenseSlice,
    ExpertEntry,
    ReceiverRank,
    Schedule,
    TrainerRank,
    UnequalExpertParallelism,
    build_schedule,
    from_wire,
    receiver_participant,
    to_wire,
)
from skyrl_train.weight_sync.expert_block.source_views import BF16, ROUTER_WEIGHT_SUFFIX
from skyrl_train.weight_sync.expert_block.stream import InstallReport


@dataclass(frozen=True)
class SyncTimings:
    install_seconds: float
    policy_seconds: float
    receiver_seconds: float


def plan_from_inventories(policy: list[dict], receivers: list[tuple[dict, object]]) -> tuple[Schedule, dict[str, int]]:
    """Build the schedule from every trainer's and receiver's report; returns it with the GPU-to-participant map.

    ``receivers`` pairs each engine's report with its verified node-local placement.
    """
    trainers = sorted((from_wire(TrainerRank, row["trainer"]) for row in policy), key=lambda row: row.rank)
    by_rank = {row.rank: row for row in trainers}
    rows = {by_rank[from_wire(TrainerRank, row["trainer"]).rank]: row for row in policy}
    eps = {row["expert_parallel_size"] for row in policy}
    models = {tuple(sorted(row["model"].items())) for row in policy}
    if len(eps) != 1 or len(models) != 1:
        raise ValueError("Trainer ranks disagree on expert-parallel size or model dimensions")
    trainer_ep = eps.pop()
    model = dict(models.pop())
    stages: dict[int, dict] = {}
    for trainer, row in rows.items():
        reference = stages.setdefault(trainer.pp, row)
        if (reference["layers"], reference["dense"]) != (row["layers"], row["dense"]):
            raise ValueError(f"Trainer ranks of stage {trainer.pp} disagree on their layers or dense weights")
    for trainer, row in rows.items():
        peer = next(other for other in rows if (other.pp, other.ep) == (trainer.pp, trainer.ep) and other.dp == 0)
        if rows[peer]["experts"] != row["experts"]:
            raise ValueError(f"Trainer rank {trainer.rank} holds different experts from its data-parallel peer")
    layers_by_pp = tuple(tuple(stages[pp]["layers"]) for pp in sorted(stages))
    entries = [
        from_wire(ExpertEntry, item) for trainer, row in rows.items() if trainer.dp == 0 for item in row["experts"]
    ]
    dense = [from_wire(DenseSlice, item) for pp in sorted(stages) for item in stages[pp]["dense"]]

    receiver_ranks, participants_by_gpu = [], {}
    receiver_models, installed = set(), {}
    for report, placement in receivers:
        worker = placement.worker
        if report["gpu_uuid"] != worker.gpu_uuid or report["ep_rank"] != worker.ep_rank:
            raise ValueError(f"Receiver on {report['gpu_uuid']} does not match its verified placement {placement}")
        if report["expert_parallel_size"] != trainer_ep:
            raise UnequalExpertParallelism(
                f"trainer EP {trainer_ep} differs from receiver EP {report['expert_parallel_size']}"
            )
        receiver = ReceiverRank(placement.replica * trainer_ep + worker.ep_rank, placement.replica, worker.ep_rank)
        receiver_ranks.append(receiver)
        participants_by_gpu[worker.gpu_uuid] = receiver_participant(len(trainers), receiver)
        receiver_models.add(tuple(sorted(report["model"].items())))
        installed[receiver.rank] = report["dense"]
    if len(receiver_models) != 1:
        raise ValueError("Receivers disagree on model dimensions")
    receiver_model = dict(receiver_models.pop())
    if any(receiver_model[key] != model[key] for key in model) or receiver_model["num_hidden_layers"] != sum(
        map(len, layers_by_pp)
    ):
        raise ValueError(f"Receiver model {receiver_model} differs from trainer model {model}")
    reference = next(iter(installed.values()))
    if any(value != reference for value in installed.values()):
        raise ValueError("Receivers disagree on their dense parameters")
    _check_dense_coverage(dense, reference)
    schedule = build_schedule(
        trainers,
        sorted(receiver_ranks, key=lambda row: row.rank),
        entries,
        dense,
        trainer_ep=trainer_ep,
        receiver_ep=trainer_ep,
        layers_by_pp=layers_by_pp,
        num_experts=model["num_experts"],
    )
    return schedule, participants_by_gpu


def _check_dense_coverage(dense: list[DenseSlice], installed: dict[str, list]) -> None:
    """Every dense parameter on the receiver is fully covered by trainer slices of a compatible dtype."""
    numel = Counter()
    dtypes = {}
    for item in dense:
        numel[item.hf_name] += item.numel
        dtypes.setdefault(item.hf_name, item.wire_dtype)
    if set(numel) != set(installed):
        raise ValueError(
            f"Dense weights differ: trainer {sorted(set(numel) - set(installed))}, receiver {sorted(set(installed) - set(numel))}"
        )
    for name, (shape, dtype) in installed.items():
        expected = 1
        for dimension in shape:
            expected *= dimension
        if numel[name] != expected:
            raise ValueError(f"Trainer slices cover {numel[name]} of {expected} elements of {name}")
        widened = name.endswith(ROUTER_WEIGHT_SUFFIX) and dtypes[name] == BF16 and dtype == "float32"
        if dtypes[name] != dtype and not widened:
            raise ValueError(f"{name} is {dtypes[name]} on the trainer and {dtype} on the receiver")


class ExpertBlockSync:
    def __init__(self, *, policy_model, inference_engine_client, timeout_seconds: int):
        self.policy_model = policy_model
        self.client = inference_engine_client
        self.timeout_seconds = timeout_seconds
        self.store: RendezvousStore | None = None
        self.schedule: Schedule | None = None
        self.root_bytes: dict[int, int] = {}

    async def _policy(self, method: str, *args) -> list:
        refs = self.policy_model.async_run_ray_method("pass_through", "expert_block_rpc", method, *args)
        return list(await asyncio.gather(*refs))

    async def prepare(self) -> dict[str, float]:
        """Plan, rendezvous and bind every participant; returns the one-time cost in seconds by phase."""
        if self.schedule is not None:
            raise RuntimeError("Expert-block sync is already prepared")
        engines = self.client.engines
        if any(engine.replica_placement is None for engine in engines):
            raise ValueError("Expert-block sync requires generator.inference_engine_node_local=true")
        started = time.perf_counter()
        policy_rows, receiver_rows = await asyncio.gather(
            self._policy("inventory"), self.client.expert_block_rpc("inventory")
        )
        if len(receiver_rows) != len(engines):
            raise RuntimeError(f"{len(engines) - len(receiver_rows)} inference engines did not report an inventory")
        schedule, participants = plan_from_inventories(
            policy_rows, list(zip(receiver_rows, (engine.replica_placement for engine in engines), strict=True))
        )
        planned = time.perf_counter()
        self.store = RendezvousStore(f"expert-block-{id(self):x}", timeout_seconds=self.timeout_seconds)
        init_info = {"schedule": to_wire(schedule), "rendezvous": asdict(self.store.rendezvous)}
        try:
            bound = await asyncio.gather(
                self._policy("trainer_init", init_info),
                self.client.expert_block_rpc("init_transfer_engine", {**init_info, "participants": participants}),
            )
        except BaseException:
            await self.close()
            raise
        reported = sorted(row["participant"] for rows in bound for row in rows)
        expected = sorted(set(range(schedule.trainer_count)) | set(participants.values()))
        if reported != expected:
            raise RuntimeError(f"Bound participants {reported} differ from the planned {expected}")
        self.schedule = schedule
        self.root_bytes = Counter()
        for item in schedule.experts:
            self.root_bytes[item.root] += item.entry.nbytes
        for item in schedule.dense:
            self.root_bytes[item.root] += item.source.nbytes
        return {"plan": planned - started, "bind": time.perf_counter() - planned}

    async def sync(self, version: int) -> SyncTimings:
        """Run one sync for ``version`` and check every receiver landed exactly its share."""
        if self.schedule is None:
            raise RuntimeError("Expert-block sync is not prepared")
        if not self.client.generation_paused_event.is_set():
            raise RuntimeError("Expert-block sync requires paused generation")
        started = time.perf_counter()
        update_info = {"version": version}
        policy_rows, receiver_rows = await asyncio.gather(
            self._policy("send_weights", update_info), self.client.expert_block_rpc("receive_weights", update_info)
        )
        policy = [InstallReport(**row) for row in policy_rows]
        receivers = [InstallReport(**row) for row in receiver_rows]
        expected_bytes = dict(self.schedule.receiver_bytes)
        if sorted(report.participant for report in receivers) != sorted(expected_bytes):
            raise RuntimeError("Not every planned receiver reported this sync")
        for report in receivers:
            if report.version != version:
                raise RuntimeError(
                    f"Receiver {report.participant} installed version {report.version}, expected {version}"
                )
            if (
                report.expert_matrices != self.schedule.receiver_expert_count
                or report.wire_bytes != expected_bytes[report.participant]
            ):
                raise RuntimeError(
                    f"Receiver {report.participant} landed {report.expert_matrices} expert matrices and {report.wire_bytes} bytes, "
                    f"expected {self.schedule.receiver_expert_count} and {expected_bytes[report.participant]}"
                )
        for report in policy:
            if report.version != version or report.wire_bytes != self.root_bytes.get(report.participant, 0):
                raise RuntimeError(
                    f"Trainer rank {report.participant} sent {report.wire_bytes} bytes for version {report.version}"
                )
        return SyncTimings(
            install_seconds=time.perf_counter() - started,
            policy_seconds=max(report.seconds for report in policy),
            receiver_seconds=max(report.seconds for report in receivers),
        )

    async def close(self) -> None:
        try:
            await asyncio.gather(self._policy("shutdown"), self.client.expert_block_rpc("shutdown"))
        finally:
            if self.store is not None:
                self.store.close()
                self.store = None
            self.schedule = None
