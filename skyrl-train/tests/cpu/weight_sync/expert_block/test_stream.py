"""Real gloo broadcasts over the expert-block schedule: bytes land where the schedule says, and nowhere else.

Twelve processes: a PP2 x EP2 x DP2 trainer and two EP2 inference replicas. Every
tensor is filled from its identity, so a receiver can check what it installed
against what the owning trainer must have sent without any side channel.
"""

from datetime import timedelta
from itertools import product
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from skyrl_train.weight_sync.expert_block.groups import Rendezvous, create_groups, destroy_groups, warm_groups
from skyrl_train.weight_sync.expert_block.schedule import (
    DenseSlice,
    ExpertEntry,
    ReceiverRank,
    TrainerRank,
    build_schedule,
)
from skyrl_train.weight_sync.expert_block.source_views import ExpertSource
from skyrl_train.weight_sync.expert_block.stream import Stream

LAYERS_BY_PP = ((0,), (1,))
NUM_EXPERTS = 4
EP = 2
DP = 2
REPLICAS = 2
HIDDEN, INTERMEDIATE = 3, 2
TIMEOUT = 60


def trainers():
    return tuple(
        TrainerRank(index, dp, pp, ep) for index, (dp, pp, ep) in enumerate(product(range(DP), range(2), range(EP)))
    )


def receivers():
    return tuple(ReceiverRank(replica * EP + ep, replica, ep) for replica in range(REPLICAS) for ep in range(EP))


def expert_value(layer, expert, projection):
    return float(layer * 100 + expert * 10 + (1 if projection == "fc1" else 2))


def dense_value(name):
    return float(sum(map(ord, name)) % 97)


DENSE = {
    "model.layers.0.mlp.router.weight": (0, (NUM_EXPERTS, HIDDEN), "float32"),  # BF16 on the wire, FP32 installed
    "model.layers.1.mlp.router.weight": (1, (NUM_EXPERTS, HIDDEN), "float32"),
    "model.norm.weight": (1, (HIDDEN,), "bfloat16"),
}


def expert_sources_for(trainer):
    """Every expert matrix a trainer owns, as its own parameter, filled from its identity."""
    per_block = NUM_EXPERTS // EP
    sources, experts = {}, {}
    for layer in LAYERS_BY_PP[trainer.pp]:
        for expert in range(trainer.ep * per_block, (trainer.ep + 1) * per_block):
            for projection in ("fc1", "fc2"):
                shape = (2 * INTERMEDIATE, HIDDEN) if projection == "fc1" else (HIDDEN, INTERMEDIATE)
                key = f"decoder.layers.{layer}.mlp.experts.linear_{projection}.weight{expert}"
                sources[key] = torch.full(shape, expert_value(layer, expert, projection), dtype=torch.bfloat16)
                entry = ExpertEntry(
                    f"model.layers.{layer}.mlp.experts.{projection}.expert{expert}",
                    layer,
                    trainer.pp,
                    expert,
                    projection,
                    sources[key].numel() * 2,
                )
                experts[entry.name] = ExpertSource(entry, key, shape)
    return sources, experts


def dense_slices_for(pp):
    slices = []
    for name, (owner_pp, shape, _) in sorted(DENSE.items()):
        if owner_pp != pp:
            continue
        numel = int(torch.Size(shape).numel())
        slices.append(DenseSlice(name, 0, numel, "bfloat16", f"source.{name}", 0, pp))
    return slices


def schedule():
    entries = []
    for trainer in trainers():
        if trainer.dp == 0:
            entries.extend(source.entry for source in expert_sources_for(trainer)[1].values())
    dense = dense_slices_for(0) + dense_slices_for(1)
    return build_schedule(
        trainers(),
        receivers(),
        entries,
        dense,
        trainer_ep=EP,
        receiver_ep=EP,
        layers_by_pp=LAYERS_BY_PP,
        num_experts=NUM_EXPERTS,
    )


def participant_main(rank, port, directory):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/default-{rank}",
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=TIMEOUT),
    )
    plan = schedule()
    trainer_count = plan.trainer_count
    device = torch.device("cpu")
    if rank < trainer_count:
        trainer = trainers()[rank]
        sources, experts = expert_sources_for(trainer)
        for item in dense_slices_for(trainer.pp):
            sources[item.source_key] = torch.full((item.numel,), dense_value(item.hf_name), dtype=torch.bfloat16)
        before = {key: value.clone() for key, value in sources.items()}
        kwargs = dict(sources=sources, expert_sources=experts)
    else:
        receiver = receivers()[rank - trainer_count]
        parameters, maps = {}, {}
        per_block = NUM_EXPERTS // EP
        for layer in range(2):
            prefix = f"model.layers.{layer}.mlp.experts.routed_experts"
            parameters[f"{prefix}.w13_weight"] = torch.zeros(
                (per_block, 2 * INTERMEDIATE, HIDDEN), dtype=torch.bfloat16
            )
            parameters[f"{prefix}.w2_weight"] = torch.zeros((per_block, HIDDEN, INTERMEDIATE), dtype=torch.bfloat16)
            maps[prefix] = tuple(
                expert - receiver.ep * per_block if expert // per_block == receiver.ep else -1
                for expert in range(NUM_EXPERTS)
            )
        for name, (_, shape, dtype) in DENSE.items():
            parameters[name] = torch.zeros(shape, dtype=getattr(torch, dtype))
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
            roots = {group.members[0] for group in plan.groups if group.name.startswith("expert-")}
            if rank not in roots:
                assert groups == {} and report.wire_bytes == 0 and report.expert_matrices == 0
            else:
                assert report.expert_matrices == len([item for item in plan.experts if item.root == rank])
            # Sending never writes a source.
            assert all(torch.equal(before[key], sources[key]) for key in sources)
        else:
            assert report.expert_matrices == plan.receiver_expert_count
            assert report.wire_bytes == dict(plan.receiver_bytes)[rank]
            for layer in range(2):
                prefix = f"model.layers.{layer}.mlp.experts.routed_experts"
                for expert in range(NUM_EXPERTS):
                    local = maps[prefix][expert]
                    if local < 0:
                        continue
                    assert torch.all(parameters[f"{prefix}.w13_weight"][local] == expert_value(layer, expert, "fc1"))
                    assert torch.all(parameters[f"{prefix}.w2_weight"][local] == expert_value(layer, expert, "fc2"))
            for name in DENSE:
                expected = torch.tensor(dense_value(name), dtype=torch.bfloat16).to(parameters[name].dtype)
                assert torch.all(parameters[name] == expected), name
            assert parameters["model.layers.0.mlp.router.weight"].dtype == torch.float32
    finally:
        destroy_groups(groups)
        dist.destroy_process_group()


def test_every_receiver_installs_exactly_its_experts_and_all_dense_weights(tmp_path):
    store = dist.TCPStore(
        "127.0.0.1", 0, world_size=None, is_master=True, timeout=timedelta(seconds=TIMEOUT), wait_for_workers=False
    )
    world = len(trainers()) + len(receivers())
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mp.spawn(participant_main, args=(store.port, str(tmp_path)), nprocs=world, join=True)
