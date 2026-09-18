"""Real gloo broadcasts over the expert-block schedule: bytes land where the schedule says, and nowhere else.

Every process is one participant. Every tensor is filled from its identity, so a
receiver can check what it installed against what the owning trainer must have
sent without any side channel. Three topologies: the qualified shape (TP=1, equal
EP, one receiver stage, two replicas); unequal EP with two receiver stages; and
a tensor-parallel trainer whose dense row shards and expert shards land through
scratch. Each also runs the gate: a replay finds no differing byte, then exactly
the one byte flipped on one receiver and the one flipped on one data-parallel peer.
"""

from dataclasses import dataclass
from datetime import timedelta
from itertools import product
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from skyrl_train.weight_sync.expert_block.gate import compare_replicas, replay
from skyrl_train.weight_sync.expert_block.groups import Rendezvous, create_groups, destroy_groups, warm_groups
from skyrl_train.weight_sync.expert_block.schedule import (
    DenseSlice,
    ExpertEntry,
    Group,
    ReceiverRank,
    TrainerRank,
    build_schedule,
)
from skyrl_train.weight_sync.expert_block.source_views import ExpertSource
from skyrl_train.weight_sync.expert_block.stream import Stream

NUM_EXPERTS = 4
TIMEOUT = 60


@dataclass(frozen=True)
class Topology:
    trainer_layers: tuple[tuple[int, ...], ...]
    trainer_ep: int
    trainer_dp: int
    trainer_tp: int
    replicas: int
    receiver_layers: tuple[tuple[int, ...], ...]
    receiver_ep: int
    hidden: int
    intermediate: int
    # dense HF name -> (trainer stage, shape, installed dtype, receiver stages holding it, "column" | "row")
    dense: dict

    def trainers(self):
        coordinates = product(
            range(self.trainer_dp), range(len(self.trainer_layers)), range(self.trainer_ep), range(self.trainer_tp)
        )
        return tuple(TrainerRank(index, dp, pp, ep, tp) for index, (dp, pp, ep, tp) in enumerate(coordinates))

    def receivers(self):
        coordinates = product(range(self.replicas), range(len(self.receiver_layers)), range(self.receiver_ep))
        return tuple(ReceiverRank(index, replica, ep, stage) for index, (replica, stage, ep) in enumerate(coordinates))


QUALIFIED = Topology(
    trainer_layers=((0,), (1,)),
    trainer_ep=2,
    trainer_dp=2,
    trainer_tp=1,
    replicas=2,
    receiver_layers=((0, 1),),
    receiver_ep=2,
    hidden=3,
    intermediate=2,
    dense={
        "model.layers.0.mlp.router.weight": (
            0,
            (NUM_EXPERTS, 3),
            "float32",
            (0,),
            "column",
        ),  # BF16 wire, FP32 installed
        "model.layers.1.mlp.router.weight": (1, (NUM_EXPERTS, 3), "float32", (0,), "column"),
        "model.norm.weight": (1, (3,), "bfloat16", (0,), "column"),
        "lm_head.weight": (1, (5, 3), "bfloat16", (0,), "column"),
    },
)
# The receiver's copy of a vocabulary tensor has more rows than the HF tensor (vLLM pads the
# vocabulary); the transport sees the leading HF rows and must leave the padded tail alone.
PADDED_ROWS = {"lm_head.weight": 8}
# Trainer EP 2 feeding receivers at EP 1 across two receiver stages; the norm is held by both stages.
UNEQUAL_STAGED = Topology(
    trainer_layers=((0, 1), (2,)),
    trainer_ep=2,
    trainer_dp=1,
    trainer_tp=1,
    replicas=1,
    receiver_layers=((0,), (1, 2)),
    receiver_ep=1,
    hidden=3,
    intermediate=2,
    dense={
        "model.layers.0.mlp.router.weight": (0, (NUM_EXPERTS, 3), "float32", (0,), "column"),
        "model.layers.2.mlp.router.weight": (1, (NUM_EXPERTS, 3), "float32", (1,), "column"),
        "model.norm.weight": (1, (3,), "bfloat16", (0, 1), "column"),
    },
)
# Two tensor-parallel ranks per slot: expert shards, a column-sharded norm and a row-sharded (column block) o_proj.
TENSOR_PARALLEL = Topology(
    trainer_layers=((0,), (1,)),
    trainer_ep=2,
    trainer_dp=1,
    trainer_tp=2,
    replicas=1,
    receiver_layers=((0, 1),),
    receiver_ep=2,
    hidden=4,
    intermediate=2,
    dense={
        "model.layers.0.self_attn.o_proj.weight": (0, (4, 4), "bfloat16", (0,), "row"),
        "model.layers.1.mlp.router.weight": (1, (NUM_EXPERTS, 4), "float32", (0,), "row"),
        "model.norm.weight": (1, (4,), "bfloat16", (0,), "column"),
    },
)


def expert_value(layer, expert, projection):
    return float(layer * 100 + expert * 10 + (1 if projection == "fc1" else 2))


def dense_value(name):
    return float(sum(map(ord, name)) % 97)


def expert_sources_for(topology, trainer):
    """Every expert matrix shard a trainer rank owns, as its own parameter, filled from its identity."""
    per_block = NUM_EXPERTS // topology.trainer_ep
    shards = topology.trainer_tp
    piece = topology.intermediate // shards
    sources, experts = {}, {}
    for layer in topology.trainer_layers[trainer.pp]:
        for expert in range(trainer.ep * per_block, (trainer.ep + 1) * per_block):
            for projection in ("fc1", "fc2"):
                shape = (2 * piece, topology.hidden) if projection == "fc1" else (topology.hidden, piece)
                key = f"decoder.layers.{layer}.mlp.experts.linear_{projection}.weight{expert}"
                sources[key] = torch.full(shape, expert_value(layer, expert, projection), dtype=torch.bfloat16)
                suffix = f".shard{trainer.tp}" if shards > 1 else ""
                name = f"model.layers.{layer}.mlp.experts.{projection}.expert{expert}{suffix}"
                entry = ExpertEntry(
                    name, layer, trainer.pp, expert, projection, sources[key].numel() * 2, trainer.tp, shards
                )
                experts[name] = ExpertSource(entry, key, shape)
    return sources, experts


def dense_slices_for(topology, trainer):
    """The dense regions a trainer rank holds: TP column shards are one run, row shards a column block."""
    slices = []
    size = topology.trainer_tp
    for name, (owner_pp, shape, _, _, layout) in sorted(topology.dense.items()):
        if owner_pp != trainer.pp:
            continue
        numel = int(torch.Size(shape).numel()) // size
        if layout == "column" or size == 1:
            slices.append(DenseSlice(name, trainer.tp * numel, numel, "bfloat16", f"source.{name}", 0, trainer.pp))
        else:
            rows, columns = shape
            slices.append(
                DenseSlice(
                    name,
                    trainer.tp * (columns // size),
                    numel,
                    "bfloat16",
                    f"source.{name}",
                    0,
                    trainer.pp,
                    rows,
                    columns,
                )
            )
    return slices


def schedule(topology):
    experts = {
        row.rank: [source.entry for source in expert_sources_for(topology, row)[1].values()]
        for row in topology.trainers()
    }
    dense = {row.rank: dense_slices_for(topology, row) for row in topology.trainers()}
    return build_schedule(
        topology.trainers(),
        topology.receivers(),
        experts,
        dense,
        receiver_ep=topology.receiver_ep,
        num_experts=NUM_EXPERTS,
        receiver_layers_by_pp=topology.receiver_layers,
        dense_holders={name: holders for name, (_, _, _, holders, _) in topology.dense.items()},
        dense_numel={name: int(torch.Size(shape).numel()) for name, (_, shape, _, _, _) in topology.dense.items()},
    )


def participant_main(rank, topology, port, directory):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/default-{rank}",
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=TIMEOUT),
    )
    plan = schedule(topology)
    trainer_count = plan.trainer_count
    device = torch.device("cpu")
    if rank < trainer_count:
        trainer = topology.trainers()[rank]
        sources, experts = expert_sources_for(topology, trainer)
        for item in dense_slices_for(topology, trainer):
            sources[item.source_key] = torch.full((item.numel,), dense_value(item.hf_name), dtype=torch.bfloat16)
        before = {key: value.clone() for key, value in sources.items()}
        kwargs = dict(sources=sources, expert_sources=experts)
    else:
        receiver = topology.receivers()[rank - trainer_count]
        parameters, maps = {}, {}
        per_block = NUM_EXPERTS // topology.receiver_ep
        held_layers = topology.receiver_layers[receiver.pp]
        for layer in held_layers:
            prefix = f"model.layers.{layer}.mlp.experts.routed_experts"
            parameters[f"{prefix}.w13_weight"] = torch.zeros(
                (per_block, 2 * topology.intermediate, topology.hidden), dtype=torch.bfloat16
            )
            parameters[f"{prefix}.w2_weight"] = torch.zeros(
                (per_block, topology.hidden, topology.intermediate), dtype=torch.bfloat16
            )
            maps[prefix] = tuple(
                expert - receiver.ep * per_block if expert // per_block == receiver.ep else -1
                for expert in range(NUM_EXPERTS)
            )
        held_dense = {name: spec for name, spec in topology.dense.items() if receiver.pp in spec[3]}
        padded = {}
        for name, (_, shape, dtype, _, _) in held_dense.items():
            if name in PADDED_ROWS:
                padded[name] = torch.zeros((PADDED_ROWS[name], *shape[1:]), dtype=getattr(torch, dtype))
                parameters[name] = padded[name].narrow(0, 0, shape[0])
            else:
                parameters[name] = torch.zeros(shape, dtype=getattr(torch, dtype))
        kwargs = dict(parameters=parameters, expert_maps=maps, hidden_size=topology.hidden)
    rendezvous = Rendezvous("127.0.0.1", port, "test", TIMEOUT)
    groups = create_groups(rank, plan.groups, rendezvous, backend="gloo")
    try:
        warm = warm_groups(rank, plan.groups, groups, device)
        assert set(warm) == set(groups)
        stream = Stream(rank, plan, groups, device=device, **kwargs)
        report = stream.run(7)
        assert report.version == 7
        if rank < trainer_count:
            roots = {group.members[0] for group in plan.groups if group.name.startswith("expert-")}
            if rank not in roots:
                assert groups == {} and report.wire_bytes == 0 and report.expert_matrices == 0
            else:
                assert report.expert_matrices == len([item for item in plan.experts if item.root == rank])
            # Sending never writes a source.
            assert all(torch.equal(before[key], sources[key]) for key in sources)
        else:
            assert report.expert_matrices == dict(plan.receiver_experts)[rank]
            assert report.wire_bytes == dict(plan.receiver_bytes)[rank]
            for layer in held_layers:
                prefix = f"model.layers.{layer}.mlp.experts.routed_experts"
                for expert in range(NUM_EXPERTS):
                    local = maps[prefix][expert]
                    if local < 0:
                        continue
                    assert torch.all(parameters[f"{prefix}.w13_weight"][local] == expert_value(layer, expert, "fc1"))
                    assert torch.all(parameters[f"{prefix}.w2_weight"][local] == expert_value(layer, expert, "fc2"))
            for name in held_dense:
                expected = torch.tensor(dense_value(name), dtype=torch.bfloat16).to(parameters[name].dtype)
                assert torch.all(parameters[name] == expected), name
            for name, whole in padded.items():
                assert torch.all(whole[parameters[name].shape[0] :] == 0), f"{name}: the padded tail was written"
        # --- The gate: a replay of the same sync matches every installed byte and covers them all ---
        report = replay(stream, 7)
        if rank >= trainer_count:
            assert report.mismatched_bytes == 0
            assert report.compared_bytes == dict(plan.receiver_bytes)[rank] == report.parameter_bytes
        # One flipped byte on one receiver's installed slot is found by the next replay, and only there.
        if rank == trainer_count:
            slot = parameters[f"model.layers.{held_layers[0]}.mlp.experts.routed_experts.w13_weight"]
            slot.view(-1).view(torch.uint8)[1] ^= 0xFF
        report = replay(stream, 7)
        if rank >= trainer_count:
            assert report.mismatched_bytes == (1 if rank == trainer_count else 0)
        # --- Data-parallel peers hold byte-identical parameters, or the gate says which bytes are not ---
        if rank < trainer_count and topology.trainer_dp == 2:
            trainer = topology.trainers()[rank]
            peers = tuple(
                row.rank
                for row in topology.trainers()
                if (row.pp, row.ep, row.tp) == (trainer.pp, trainer.ep, trainer.tp)
            )
            peer_group = create_groups(
                rank, (Group(f"peers-{trainer.pp}-{trainer.ep}", peers),), rendezvous, backend="gloo"
            )
            replica_groups = dict.fromkeys(sources, next(iter(peer_group.values())))
            try:
                # A chunk far smaller than any parameter, so every tensor is compared in several pieces.
                compare = lambda: compare_replicas(sources, replica_groups, rank, 7, chunk_bytes=8)  # noqa: E731
                assert compare().mismatched_bytes == 0
                if trainer.dp == 1:
                    next(iter(sources.values())).view(-1).view(torch.uint8)[-1] ^= 0xFF
                assert compare().mismatched_bytes == 1
            finally:
                destroy_groups(peer_group)
    finally:
        destroy_groups(groups)
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "topology",
    [QUALIFIED, UNEQUAL_STAGED, TENSOR_PARALLEL],
    ids=["equal-ep-one-stage", "unequal-ep-two-stages", "tensor-parallel-trainer"],
)
def test_every_receiver_installs_exactly_its_experts_and_the_dense_weights_its_stage_holds(topology, tmp_path):
    store = dist.TCPStore(
        "127.0.0.1", 0, world_size=None, is_master=True, timeout=timedelta(seconds=TIMEOUT), wait_for_workers=False
    )
    world = len(topology.trainers()) + len(topology.receivers())
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mp.spawn(participant_main, args=(topology, store.port, str(tmp_path)), nprocs=world, join=True)
