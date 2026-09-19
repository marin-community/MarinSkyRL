"""Run the schedule over real gloo broadcasts and check every element the receivers got.

Each process is one participant. Trainer ranks hold Megatron-layout parameters with a value
unique to each position and describe them with conversion tasks, as in production. Receivers
compare what they received with an independent reference conversion. There are two topologies:
equal EP with one receiver stage and two replicas, and unequal EP with two receiver stages.

Each topology then verifies the sync. A replay finds no differing byte, then exactly the one byte
flipped on one receiver, and the peer comparison finds the one byte flipped on one data-parallel rank.
"""

from dataclasses import dataclass
from datetime import timedelta
from itertools import product
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from skyrl_train.weight_sync.expert_block.groups import Rendezvous, create_groups, destroy_groups, warm_groups
from skyrl_train.weight_sync.expert_block.schedule import Group, ReceiverRank, TrainerRank, build_schedule
from skyrl_train.weight_sync.expert_block.source_views import local_expert_sources, local_source_slices
from skyrl_train.weight_sync.expert_block.stream import Stream
from skyrl_train.weight_sync.expert_block.verify_weights import compare_replicas, replay
from tests.cpu.weight_sync.expert_block.megatron_layout import (
    HIDDEN,
    INTERMEDIATE,
    NUM_EXPERTS,
    PROVIDER,
    conversion_tasks,
    megatron_parameters,
    megatron_shapes,
    reference_hf,
)

TIMEOUT = 60
ROUTED_EXPERTS = "mlp.experts.routed_experts"
# vLLM pads the vocabulary, so the receiver's tensor has more rows than the HF tensor. The
# transport writes the HF rows and must not touch the padding.
PADDED_VOCAB = 8


@dataclass(frozen=True)
class Topology:
    trainer_layers: tuple[tuple[int, ...], ...]
    trainer_ep: int
    trainer_dp: int
    replicas: int
    receiver_layers: tuple[tuple[int, ...], ...]
    receiver_ep: int
    # Receiver stages holding the final norm.
    norm_stages: tuple[int, ...]

    def trainers(self):
        coordinates = product(range(self.trainer_dp), range(len(self.trainer_layers)), range(self.trainer_ep))
        return tuple(TrainerRank(index, dp, pp, ep) for index, (dp, pp, ep) in enumerate(coordinates))

    def receivers(self):
        coordinates = product(range(self.replicas), range(len(self.receiver_layers)), range(self.receiver_ep))
        return tuple(ReceiverRank(index, replica, ep, stage) for index, (replica, stage, ep) in enumerate(coordinates))

    def trainer_block(self, ep: int) -> range:
        per_block = NUM_EXPERTS // self.trainer_ep
        return range(ep * per_block, (ep + 1) * per_block)

    def model_names(self) -> list[str]:
        last = len(self.trainer_layers) - 1
        return sorted(
            {
                name
                for pp, layers in enumerate(self.trainer_layers)
                for name in megatron_shapes(layers, range(NUM_EXPERTS), last_stage=pp == last)
            }
        )

    def parameters_of(self, trainer: TrainerRank) -> dict[str, torch.Tensor]:
        return megatron_parameters(
            self.trainer_layers[trainer.pp],
            self.trainer_block(trainer.ep),
            last_stage=trainer.pp == len(self.trainer_layers) - 1,
            model_names=self.model_names(),
        )

    def dense_stages(self, hf_name: str) -> tuple[int, ...]:
        """The receiver stages that hold a dense HF tensor."""
        if hf_name == "model.norm.weight":
            return self.norm_stages
        if hf_name == "lm_head.weight":
            return (len(self.receiver_layers) - 1,)
        layer = int(hf_name.split(".")[2])
        return tuple(stage for stage, layers in enumerate(self.receiver_layers) if layer in layers)


EQUAL_EP = Topology(
    trainer_layers=((0,), (1,)),
    trainer_ep=2,
    trainer_dp=2,
    replicas=2,
    receiver_layers=((0, 1),),
    receiver_ep=2,
    norm_stages=(0,),
)
# Trainer EP 2 feeding receivers at EP 1 across two receiver stages; both stages hold the norm.
UNEQUAL_STAGED = Topology(
    trainer_layers=((0, 1), (2,)),
    trainer_ep=2,
    trainer_dp=1,
    replicas=1,
    receiver_layers=((0,), (1, 2)),
    receiver_ep=1,
    norm_stages=(0, 1),
)


def trainer_sources(topology, trainer):
    """A trainer rank's parameters and the slices the production code extracts from them."""
    local = local_source_slices(conversion_tasks(topology.parameters_of(trainer)), PROVIDER, pp=trainer.pp)
    experts = local_expert_sources(
        local.experts,
        local.sources,
        trainer,
        num_experts=NUM_EXPERTS,
        expert_parallel_size=topology.trainer_ep,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
    )
    return local, {item.entry.name: item for item in experts}


def reference(topology):
    """Every dense HF tensor and expert matrix of the whole model."""
    dense, experts = {}, {}
    for trainer in topology.trainers():
        rank_dense, rank_experts = reference_hf(topology.parameters_of(trainer))
        dense.update(rank_dense)
        experts.update(rank_experts)
    return dense, experts


def schedule(topology):
    inventories = {row.rank: trainer_sources(topology, row) for row in topology.trainers()}
    dense, _ = reference(topology)
    return build_schedule(
        topology.trainers(),
        topology.receivers(),
        {rank: [source.entry for source in experts.values()] for rank, (_, experts) in inventories.items()},
        {rank: local.dense for rank, (local, _) in inventories.items()},
        receiver_ep=topology.receiver_ep,
        num_experts=NUM_EXPERTS,
        receiver_layers_by_pp=topology.receiver_layers,
        dense_holders={name: topology.dense_stages(name) for name in dense},
        dense_numel={name: value.numel() for name, value in dense.items()},
    )


def receiver_parameters(topology, receiver, dense):
    """Zeroed vLLM-layout parameters for one worker: expert slots, its stage's dense tensors and a padded LM head."""
    per_block = NUM_EXPERTS // topology.receiver_ep
    parameters, maps = {}, {}
    for layer in topology.receiver_layers[receiver.pp]:
        prefix = f"model.layers.{layer}.{ROUTED_EXPERTS}"
        parameters[f"{prefix}.w13_weight"] = torch.zeros((per_block, 2 * INTERMEDIATE, HIDDEN), dtype=torch.bfloat16)
        parameters[f"{prefix}.w2_weight"] = torch.zeros((per_block, HIDDEN, INTERMEDIATE), dtype=torch.bfloat16)
        maps[prefix] = tuple(
            expert - receiver.ep * per_block if expert // per_block == receiver.ep else -1
            for expert in range(NUM_EXPERTS)
        )
    padded_head = None
    for name, value in dense.items():
        if receiver.pp not in topology.dense_stages(name):
            continue
        # vLLM keeps the router in FP32. Its weight is BF16 on the trainer and the wire; its bias is FP32 on both.
        dtype = torch.float32 if ".mlp.router." in name else torch.bfloat16
        if name == "lm_head.weight":
            padded_head = torch.zeros((PADDED_VOCAB, HIDDEN), dtype=dtype)
            parameters[name] = padded_head.narrow(0, 0, value.shape[0])
        else:
            parameters[name] = torch.zeros(value.shape, dtype=dtype)
    return parameters, maps, padded_head


def check_trainer(rank, plan, groups, report, before, sources):
    roots = {group.members[0] for group in plan.groups if group.name.startswith("expert-")}
    if rank not in roots:
        assert groups == {} and report.wire_bytes == 0 and report.expert_matrices == 0
    else:
        assert report.expert_matrices == len([item for item in plan.experts if item.root == rank])
    # Sending never writes a source.
    assert all(torch.equal(before[key], sources[key]) for key in before)


def check_receiver(rank, topology, plan, report, parameters, maps, padded_head, dense, experts):
    assert report.expert_matrices == dict(plan.receiver_experts)[rank]
    assert report.wire_bytes == dict(plan.receiver_bytes)[rank]
    served = 0
    held_bytes = 0
    for prefix, expert_map in maps.items():
        layer = int(prefix.split(".")[2])
        for expert, slot in enumerate(expert_map):
            if slot < 0:
                continue
            served += 1
            for slot_name, projection in (("w13_weight", "fc1"), ("w2_weight", "fc2")):
                matrix = experts[projection, layer, expert]
                assert torch.equal(parameters[f"{prefix}.{slot_name}"][slot], matrix)
                held_bytes += matrix.numel() * matrix.element_size()
    assert served == len(maps) * NUM_EXPERTS // topology.receiver_ep
    assert report.expert_matrices == 2 * served
    held = [name for name in parameters if ROUTED_EXPERTS not in name]
    assert held
    for name in held:
        assert torch.equal(parameters[name], dense[name].to(parameters[name].dtype)), name
        held_bytes += dense[name].numel() * dense[name].element_size()
    # Each tensor this receiver holds is sent once, in the trainer's dtype.
    assert report.wire_bytes == held_bytes
    if padded_head is not None:
        assert torch.all(padded_head[dense["lm_head.weight"].shape[0] :] == 0), "the padded vocabulary tail was written"


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
    dense, experts = reference(topology)
    if rank < trainer_count:
        local, expert_sources = trainer_sources(topology, topology.trainers()[rank])
        before = {key: value.clone() for key, value in local.sources.items()}
        kwargs = dict(sources=local.sources, expert_sources=expert_sources)
    else:
        receiver = topology.receivers()[rank - trainer_count]
        parameters, maps, padded_head = receiver_parameters(topology, receiver, dense)
        kwargs = dict(parameters=parameters, expert_maps=maps)
    rendezvous = Rendezvous("127.0.0.1", port, "test", TIMEOUT)
    groups = create_groups(rank, plan.groups, rendezvous, backend="gloo")
    try:
        warm = warm_groups(rank, plan.groups, groups, device)
        assert set(warm) == set(groups)
        stream = Stream(rank, plan, groups, device=device, **kwargs)
        report = stream.run(7)
        assert report.version == 7
        if rank < trainer_count:
            check_trainer(rank, plan, groups, report, before, local.sources)
        else:
            check_receiver(rank, topology, plan, report, parameters, maps, padded_head, dense, experts)
        # --- A replay of the same sync matches every installed byte and covers them all ---
        replayed = replay(stream, 7)
        if rank >= trainer_count:
            assert replayed.mismatched_bytes == 0
            assert replayed.compared_bytes == dict(plan.receiver_bytes)[rank] == replayed.parameter_bytes
        # One flipped byte on one receiver's installed slot is found by the next replay, and only there.
        if rank == trainer_count:
            next(iter(parameters.values())).view(-1).view(torch.uint8)[1] ^= 0xFF
        replayed = replay(stream, 7)
        if rank >= trainer_count:
            assert replayed.mismatched_bytes == (1 if rank == trainer_count else 0)
        # --- Data-parallel peers hold byte-identical parameters, or the comparison counts the bytes that are not ---
        if rank < trainer_count and topology.trainer_dp == 2:
            trainer = topology.trainers()[rank]
            peers = tuple(row.rank for row in topology.trainers() if (row.pp, row.ep) == (trainer.pp, trainer.ep))
            peer_group = create_groups(
                rank, (Group(f"peers-{trainer.pp}-{trainer.ep}", peers),), rendezvous, backend="gloo"
            )
            replica_groups = dict.fromkeys(local.sources, next(iter(peer_group.values())))
            try:
                # A chunk far smaller than any parameter, so every tensor is compared in several pieces.
                compare = lambda: compare_replicas(local.sources, replica_groups, rank, 7, chunk_bytes=8)  # noqa: E731
                assert compare().mismatched_bytes == 0
                if trainer.dp == 1:
                    next(iter(local.sources.values())).view(-1).view(torch.uint8)[-1] ^= 0xFF
                assert compare().mismatched_bytes == 1
            finally:
                destroy_groups(peer_group)
    finally:
        destroy_groups(groups)
        dist.destroy_process_group()


@pytest.mark.parametrize("topology", [EQUAL_EP, UNEQUAL_STAGED], ids=["equal-ep-one-stage", "unequal-ep-two-stages"])
def test_every_receiver_installs_exactly_its_experts_and_the_dense_weights_its_stage_holds(topology, tmp_path):
    store = dist.TCPStore(
        "127.0.0.1", 0, world_size=None, is_master=True, timeout=timedelta(seconds=TIMEOUT), wait_for_workers=False
    )
    world = len(topology.trainers()) + len(topology.receivers())
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mp.spawn(participant_main, args=(topology, store.port, str(tmp_path)), nprocs=world, join=True)
