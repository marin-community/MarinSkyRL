import copy
import multiprocessing

import pytest
import torch

from skyrl_train.distributed.tensor_transfer import (
    TensorTransferEntry,
    TensorTransferManifest,
    broadcast_tensor_payload,
    transfer_tensor_operations,
)
from skyrl_train.distributed.utils import get_free_port, init_custom_process_group
from skyrl_train.utils import get_tcp_url


def _manifest_and_tensors():
    tensors = {
        "draft.layers.0.weight": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
        "draft.token_map": torch.tensor([3, 1, 4], dtype=torch.int64),
    }
    manifest = TensorTransferManifest.from_tensors(
        transfer_id="candidate-7",
        revision="draft-step-7",
        source_weights_sha256="a" * 64,
        tensors=tensors,
    )
    return manifest, tensors


def test_tensor_transfer_manifest_round_trips_and_validates_payload() -> None:
    manifest, tensors = _manifest_and_tensors()

    restored = TensorTransferManifest.from_mapping(manifest.to_mapping())

    assert restored == manifest
    assert restored.total_bytes == sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
    assert [entry.name for entry in restored.tensors] == sorted(tensors)
    restored.validate_tensors(tensors)


def test_tensor_transfer_manifest_rejects_reordered_operation_log() -> None:
    manifest, _ = _manifest_and_tensors()
    value = manifest.to_mapping()
    value["tensors"].reverse()

    with pytest.raises(ValueError, match="unique, sorted names"):
        TensorTransferManifest.from_mapping(value)


def test_tensor_transfer_manifest_rejects_corrupted_tensor() -> None:
    manifest, tensors = _manifest_and_tensors()
    corrupted = copy.deepcopy(tensors)
    corrupted["draft.layers.0.weight"][0, 0] = 99

    with pytest.raises(ValueError, match="digest mismatch"):
        manifest.validate_tensors(corrupted)


@pytest.mark.parametrize("field", ["total_bytes", "payload_sha256"])
def test_tensor_transfer_manifest_rejects_corrupted_envelope(field: str) -> None:
    manifest, _ = _manifest_and_tensors()
    value = manifest.to_mapping()
    value[field] = value[field] + 1 if field == "total_bytes" else "0" * 64

    with pytest.raises(ValueError, match="does not match"):
        TensorTransferManifest.from_mapping(value)


class _SingletonGroup:
    def rank(self):
        return 0

    def size(self):
        return 1


def test_capture_transfer_rejects_receiver_as_a_source_before_collectives() -> None:
    tensor = torch.ones(1)
    operations = [
        {
            "key": "invalid",
            "source_rank": 0,
            "tensor": TensorTransferEntry.from_tensor("value", tensor).to_mapping(),
        }
    ]

    with pytest.raises(ValueError, match="invalid source ranks"):
        transfer_tensor_operations(
            operations,
            tensor_provider=None,
            group=_SingletonGroup(),
            device=torch.device("cpu"),
        )


def _run_collective_transfer(rank: int, world_size: int, default_ports: list[int], custom_port: int, queue) -> None:
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=get_tcp_url("127.0.0.1", default_ports[rank]),
        world_size=1,
        rank=0,
    )
    group = init_custom_process_group(
        backend="gloo",
        init_method=get_tcp_url("127.0.0.1", custom_port),
        world_size=world_size,
        rank=rank,
        group_name="tensor-transfer-test",
    )
    candidate = {"draft.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4)}
    manifest = TensorTransferManifest.from_tensors(
        transfer_id="candidate-1",
        revision="draft-1",
        source_weights_sha256="source-1",
        tensors=candidate,
    )
    received_candidate = broadcast_tensor_payload(
        manifest.to_mapping(),
        tensors=candidate if rank == 0 else None,
        group=group,
        device=torch.device("cpu"),
        staging_bytes=16,
    )
    source_tensors = {
        1: torch.tensor([1, 2, 3], dtype=torch.int64),
        2: torch.tensor([5, 8], dtype=torch.int64),
    }
    operations = [
        {
            "key": f"window-{source_rank}",
            "source_rank": source_rank,
            "tensor": TensorTransferEntry.from_tensor(f"tensor-{source_rank}", tensor).to_mapping(),
        }
        for source_rank, tensor in source_tensors.items()
    ]
    received_capture = transfer_tensor_operations(
        operations,
        tensor_provider=(lambda _operation: source_tensors[rank]) if rank in source_tensors else None,
        group=group,
        device=torch.device("cpu"),
        staging_bytes=8,
    )
    queue.put(
        (
            rank,
            None if received_candidate is None else float(received_candidate["draft.weight"].sum()),
            None if received_capture is None else [received_capture[key].tolist() for key in sorted(received_capture)],
        )
    )
    torch.distributed.destroy_process_group(group)
    torch.distributed.destroy_process_group()


def test_tensor_transfer_collectives_move_payload_without_ray() -> None:
    context = multiprocessing.get_context("spawn")
    queue = context.SimpleQueue()
    world_size = 3
    default_ports = [get_free_port() for _ in range(world_size)]

    torch.multiprocessing.spawn(
        _run_collective_transfer,
        args=(world_size, default_ports, get_free_port(), queue),
        nprocs=world_size,
        join=True,
    )

    results = {rank: (candidate_sum, capture) for rank, candidate_sum, capture in [queue.get() for _ in range(3)]}
    assert results == {
        0: (None, [[1, 2, 3], [5, 8]]),
        1: (66.0, None),
        2: (66.0, None),
    }
