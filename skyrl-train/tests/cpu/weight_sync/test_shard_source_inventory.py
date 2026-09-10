from dataclasses import replace
from itertools import product

import pytest
import torch

from skyrl_train.weight_sync.frozen_source_views import FrozenSourceSlice, source_view
from skyrl_train.weight_sync.shard_group_schedule import ReceiverRank, TrainerRank, build_shard_group_schedule
from skyrl_train.weight_sync.shard_source_inventory import (
    DenseSourceRank,
    dense_stream_plan,
    expert_destination_view,
    expert_source_view,
    local_shard_inventory,
)


def source_fixture(trainer):
    sources, slices = {}, []
    for expert in range(trainer.ep * 2, trainer.ep * 2 + 2):
        for projection, shape in [('fc1', (4, 3)), ('fc2', (3, 2))]:
            key = f'decoder.layers.{trainer.pp}.mlp.experts.linear_{projection}.weight{expert}'
            value = torch.arange(shape[0] * shape[1], dtype=torch.bfloat16).reshape(shape)
            value += 32 * trainer.pp + 8 * expert
            sources[key] = value
            parts = [('gate', 0), ('up', 6)] if projection == 'fc1' else [('down', 0)]
            for part, offset in parts:
                slices.append(FrozenSourceSlice(
                    f'model.layers.{trainer.pp}.mlp.experts.{part}_proj.weight',
                    expert * 6, 6, 'bfloat16', key, offset, True,
                ))
    key = f'decoder.layers.{trainer.pp}.mlp.router.expert_bias'
    sources[key] = torch.tensor([-0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
    slices.append(FrozenSourceSlice(f'model.layers.{trainer.pp}.mlp.router.bias', 0, 4, 'float32', key, 0, False))
    return tuple(slices), sources


def inventory(trainer, slices, sources):
    return local_shard_inventory(slices, sources, trainer, layers=(trainer.pp,), num_experts=4,
                                 expert_parallel_size=2, hidden_size=3, intermediate_size=2)


@pytest.mark.parametrize(('dp_count', 'replicas'), [(2, 1), (2, 2), (1, 3)])
def test_complete_schedule_installs_current_expert_views_on_every_replica(dp_count, replicas):
    trainers = tuple(TrainerRank(i, dp, pp, ep) for i, (dp, pp, ep) in enumerate(product(range(dp_count), range(2), range(2))))
    receivers = tuple(ReceiverRank(i, replica, ep) for i, (replica, ep) in enumerate(product(range(replicas), range(2))))
    states = {}
    entries = {}
    for trainer in trainers:
        slices, sources = source_fixture(trainer)
        local = inventory(trainer, slices, sources)
        assert len(local.experts) == 4 and len(local.dense) == 1
        states[trainer.rank] = (local, sources)
        for item in local.experts:
            entries[item.entry.name] = item.entry
            assert expert_source_view(item, sources).data_ptr() == sources[item.source_key].data_ptr()
        # Simulate the next optimizer update after preparing immutable metadata.
        for value in sources.values():
            value.add_(4)
    schedule = build_shard_group_schedule(trainers, receivers, tuple(entries.values()), trainer_ep=2,
                                          receiver_ep=2, layers_by_pp=((0,), (1,)), num_experts=4)
    native_trainers = {global_rank: native for native, global_rank in schedule.trainer_global_ranks}
    native_receivers = {global_rank: receivers[native] for native, global_rank in schedule.receiver_global_ranks}
    destinations = {}
    for receiver in receivers:
        parameters, maps = {}, {}
        for layer in range(2):
            prefix = f'model.layers.{layer}.mlp.experts.routed_experts'
            parameters[prefix + '.w13_weight'] = torch.full((2, 4, 3), -1, dtype=torch.bfloat16)
            parameters[prefix + '.w2_weight'] = torch.full((2, 3, 2), -1, dtype=torch.bfloat16)
            maps[prefix] = [expert % 2 if expert // 2 == receiver.ep else -1 for expert in range(4)]
        destinations[receiver.rank] = parameters, maps
    observed = set()
    for call in schedule.broadcasts:
        local, sources = states[native_trainers[call.root]]
        item = next(value for value in local.experts if value.entry.name == call.name)
        source = expert_source_view(item, sources)
        for rank in call.receiver_destinations:
            receiver = native_receivers[rank]
            parameters, maps = destinations[receiver.rank]
            target = expert_destination_view(item, parameters, maps, backend='TRITON')
            target.copy_(source)
            # Independent baseline is the original per-expert matrix with one update.
            expected = torch.arange(target.numel(), dtype=torch.bfloat16).reshape(target.shape)
            expected += 32 * item.entry.layer + 8 * item.entry.expert + 4
            assert torch.equal(target.view(torch.uint8), expected.view(torch.uint8))
            key = (receiver.rank, item.entry.layer, item.entry.expert, item.entry.projection)
            assert key not in observed
            observed.add(key)
    assert len(observed) == replicas * 2 * 4 * 2
    assert sum(value for _, value in schedule.logical_root_bytes) == 2 * 4 * (12 + 6) * 2
    assert sum(value for _, value in schedule.logical_receiver_bytes) == replicas * 2 * 4 * (12 + 6) * 2


@pytest.mark.parametrize('failure', ['missing', 'wrong_ep', 'wrong_pp', 'swapped_halves', 'alias', 'split_storage'])
def test_invalid_all_expert_inventory_fails_before_install(failure):
    trainer = TrainerRank(0, 0, 0, 0)
    slices, sources = source_fixture(trainer)
    slices = list(slices)
    if failure == 'missing':
        slices.pop(0)
    elif failure == 'wrong_ep':
        trainer = replace(trainer, ep=1)
    elif failure == 'wrong_pp':
        trainer = replace(trainer, pp=1)
    elif failure == 'swapped_halves':
        slices[0] = replace(slices[0], source_offset=6)
        slices[1] = replace(slices[1], source_offset=0)
    elif failure == 'alias':
        sources[slices[3].source_key] = sources[slices[0].source_key]
    else:
        sources['other'] = sources[slices[0].source_key].clone()
        slices[1] = replace(slices[1], source_key='other')
    with pytest.raises(ValueError):
        inventory(trainer, tuple(slices), sources)


def test_receiver_same_byte_count_wrong_shape_and_wrong_expert_map_reject():
    trainer = TrainerRank(0, 0, 0, 0)
    slices, sources = source_fixture(trainer)
    item = inventory(trainer, slices, sources).experts[0]
    prefix = 'model.layers.0.mlp.experts.routed_experts'
    parameters = {prefix + '.w13_weight': torch.empty(2, 3, 4, dtype=torch.bfloat16)}
    maps = {prefix: [0, 1, -1, -1]}
    with pytest.raises(ValueError):
        expert_destination_view(item, parameters, maps, backend='TRITON')
    parameters[prefix + '.w13_weight'] = torch.empty(2, 4, 3, dtype=torch.bfloat16)
    maps[prefix] = [0, 0, -1, -1]
    with pytest.raises(ValueError):
        expert_destination_view(item, parameters, maps, backend='TRITON')


@pytest.mark.parametrize(('dp_count', 'replicas'), [(2, 1), (2, 2), (1, 3)])
def test_dense_stream_covers_qkv_shared_router_bytes_with_single_landing_per_replica(dp_count, replicas):
    rows, sources_by_rank, expected_shapes = [], {}, {}
    for rank, (dp, pp, ep) in enumerate(product(range(dp_count), range(2), range(2))):
        sources, descriptors = {}, []
        # Two nonadjacent Q slices emulate the qualified interleaved QKV views.
        key = f'layer{pp}.qkv'
        sources[key] = torch.arange(12, dtype=torch.bfloat16) + 16 * pp
        name = f'model.layers.{pp}.self_attn.q_proj.weight'
        descriptors.extend(FrozenSourceSlice(name, i * 3, 3, 'bfloat16', key, i * 6, False) for i in range(2))
        expected_shapes[name] = ((2, 3), 'bfloat16')
        key = f'layer{pp}.shared'
        sources[key] = torch.arange(12, dtype=torch.bfloat16) + 32 * pp
        for offset, part in [(0, 'gate'), (6, 'up')]:
            name = f'model.layers.{pp}.shared_expert.{part}_proj.weight'
            descriptors.append(FrozenSourceSlice(name, 0, 6, 'bfloat16', key, offset, False))
            expected_shapes[name] = ((2, 3), 'bfloat16')
        key = f'layer{pp}.bias'
        sources[key] = torch.tensor([-0.0, 1.0], dtype=torch.float32)
        name = f'model.layers.{pp}.mlp.router.bias'
        descriptors.append(FrozenSourceSlice(name, 0, 2, 'float32', key, 0, False))
        expected_shapes[name] = ((2,), 'float32')
        rows.append(DenseSourceRank(TrainerRank(rank, dp, pp, ep), tuple(descriptors)))
        sources_by_rank[rank] = sources
    receivers = tuple(ReceiverRank(i, replica, ep) for i, (replica, ep) in enumerate(product(range(replicas), range(2))))
    plan = dense_stream_plan(rows, receivers, expected_shapes, expert_parallel_size=2)
    installed = {r.rank: {name: torch.zeros(shape, dtype=getattr(torch, dtype)) for name, (shape, dtype) in expected_shapes.items()} for r in receivers}
    coverage = set()
    for transfer in plan:
        assert len(transfer.landing_native_ranks) == replicas
        source = source_view(transfer.source, sources_by_rank[transfer.root_native_rank])
        for landing, local_ranks in zip(transfer.landing_native_ranks, transfer.replica_fanout, strict=True):
            assert landing in local_ranks and len(local_ranks) == 2
            for rank in local_ranks:
                target = installed[rank][transfer.source.hf_name].view(-1).narrow(0, transfer.source.hf_offset, transfer.source.numel)
                target.copy_(source)
                for offset in range(transfer.source.hf_offset, transfer.source.hf_offset + transfer.source.numel):
                    key = (rank, transfer.source.hf_name, offset)
                    assert key not in coverage
                    coverage.add(key)
    for values in installed.values():
        for pp in range(2):
            torch.testing.assert_close(values[f'model.layers.{pp}.self_attn.q_proj.weight'], torch.tensor([[0, 1, 2], [6, 7, 8]], dtype=torch.bfloat16) + 16 * pp)
            assert values[f'model.layers.{pp}.mlp.router.bias'].view(torch.int32)[0].item() == -(2**31)
    assert len(coverage) == len(receivers) * sum(torch.tensor(shape).prod().item() for shape, _ in expected_shapes.values())
    broken = list(rows)
    broken[0] = replace(broken[0], slices=broken[0].slices[:-1])
    with pytest.raises(ValueError, match='geometry'):
        dense_stream_plan(broken, receivers, expected_shapes, expert_parallel_size=2)
    with pytest.raises(ValueError, match='manifest'):
        dense_stream_plan(rows, receivers, {**expected_shapes, 'missing': ((1,), 'bfloat16')}, expert_parallel_size=2)


@pytest.mark.parametrize("offset", [1, 3])
def test_dense_geometry_consistent_across_ranks_still_rejects_gap_or_overlap(offset):
    descriptor = FrozenSourceSlice("dense", 0, 2, "float32", "dense", 0, False)
    second = replace(descriptor, hf_offset=offset, source_offset=2)
    rows = [DenseSourceRank(TrainerRank(ep, 0, 0, ep), (descriptor, second)) for ep in range(2)]
    receivers = [ReceiverRank(ep, 0, ep) for ep in range(2)]
    with pytest.raises(ValueError, match="coverage"):
        dense_stream_plan(rows, receivers, {"dense": ((4,), "float32")}, expert_parallel_size=2)
