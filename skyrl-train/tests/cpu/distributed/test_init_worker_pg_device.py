"""Regression guard for init_worker_process_group_with_device (the shared, backend-agnostic
device-pinned NCCL process-group init).

The bug it prevents: creating the NCCL PG without a device_id lets torch's ProcessGroupNCCL
"guess device ID based on global rank". That guess is only correct when Ray masks
CUDA_VISIBLE_DEVICES to a single device (device 0); when Ray does NOT mask CVD (every actor
sees all GPUs, e.g. the megatron+vLLM image on cw-rno2a), the guess is wrong and the first
collective (the weight-init barrier) DEADLOCKS (keep-6 / cw-rno2a, 2026-07-20). The helper must
therefore always torch.cuda.set_device(LOCAL_RANK) AND pass device_id=cuda:LOCAL_RANK.
"""

import torch

from skyrl_train.distributed.utils import init_worker_process_group_with_device


def test_pins_device_and_passes_device_id(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "3")
    seen = {}
    monkeypatch.setattr(torch.cuda, "set_device", lambda d: seen.__setitem__("set_device", d))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.distributed, "init_process_group", lambda **kw: seen.__setitem__("init", kw))

    init_worker_process_group_with_device(timeout_seconds=1800)

    # device pinned to the resolved LOCAL_RANK ...
    assert seen["set_device"] == 3
    # ... and passed EXPLICITLY as device_id so ProcessGroupNCCL never guesses.
    assert seen["init"]["device_id"] == torch.device("cuda", 3)
    assert seen["init"]["backend"] == "nccl"
    assert seen["init"]["timeout"].total_seconds() == 1800


def test_idempotent_when_already_initialized(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "2")
    seen = {}
    monkeypatch.setattr(torch.cuda, "set_device", lambda d: seen.__setitem__("set_device", d))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def _must_not_init(**kw):
        raise AssertionError("init_process_group must NOT be called when the PG already exists")

    monkeypatch.setattr(torch.distributed, "init_process_group", _must_not_init)

    init_worker_process_group_with_device(timeout_seconds=600)

    # still pins the device (idempotent), but does not re-create the PG
    assert seen["set_device"] == 2
