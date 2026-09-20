"""Weight transfer buffers survive IPC release with expandable training allocations."""

import gc
import multiprocessing

import pytest
import torch

from skyrl_train.weight_sync.weight_extractor import allocate_cuda_ipc_buffer
from tests.gpu.grug_gpu_gates import require_hoppers


def _receive_weight_buffer(connection, expected_value, expected_dtype):
    tensor = connection.recv()
    assert tensor.is_cuda and tensor.dtype == expected_dtype
    assert tensor.shape == (2**24,)
    assert torch.all(tensor == expected_value).item()
    del tensor
    gc.collect()
    torch.cuda.synchronize()
    connection.send("released")
    connection.close()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_cuda_ipc_buffer_repeated_transfer_preserves_values_and_training_allocator(dtype):
    require_hoppers(1)
    original = torch._C._accelerator_getAllocatorSettings()
    settings = ",".join(filter(None, (original, "expandable_segments:True")))
    torch._C._accelerator_setAllocatorSettings(settings)
    context = multiprocessing.get_context("spawn")
    try:
        resident = torch.ones(2**23, device="cuda", dtype=dtype)
        for value in (1, 2, 3):
            sender, receiver = context.Pipe()
            child = context.Process(target=_receive_weight_buffer, args=(receiver, value, dtype))
            child.start()
            receiver.close()
            try:
                tensor = allocate_cuda_ipc_buffer(2**24, device=torch.cuda.current_device(), dtype=dtype)
                assert torch._C._accelerator_getAllocatorSettings() == settings
                tensor.fill_(value)
                sender.send(tensor)
                assert sender.poll(90), "IPC receiver did not finish"
                assert sender.recv() == "released"
                child.join(90)
                assert child.exitcode == 0
                del tensor
                gc.collect()
                torch.cuda.ipc_collect()
                # Releasing an exported fabric allocation throws here on GB200
                # in the pinned Torch runtime, even when the receiver succeeded.
                torch.cuda.empty_cache()
                assert torch.all(resident == 1).item()
            finally:
                sender.close()
                if child.is_alive():
                    child.terminate()
                child.join(10)
                if child.is_alive():
                    child.kill()
                    child.join(10)
        del resident
        torch.cuda.empty_cache()
    finally:
        torch._C._accelerator_setAllocatorSettings(original)
