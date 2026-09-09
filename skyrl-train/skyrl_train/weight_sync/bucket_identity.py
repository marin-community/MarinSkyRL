"""Allowlisted identity attached to native bucket protocol receipts."""

import os
import socket

import torch


def bucket_identity(device):
    return {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "rank": torch.distributed.get_rank(),
        "world_size": torch.distributed.get_world_size(),
        "device": str(device),
        "gpu_uuid": str(torch.cuda.get_device_properties(device).uuid),
        "attempt_uid": os.environ.get("IRIS_ATTEMPT_UID"),
        "task_id": os.environ.get("IRIS_TASK_ID"),
    }
