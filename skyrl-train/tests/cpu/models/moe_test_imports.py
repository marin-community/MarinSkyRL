"""Import grouped MoE layers without requiring TorchTitan in CPU tests."""

import importlib
import sys
import types
from types import ModuleType


def import_grouped_moe_module() -> ModuleType:
    try:
        return importlib.import_module("skyrl_train.models.layers.moe")
    except ModuleNotFoundError as error:
        if error.name != "torchtitan":
            raise
    torchtitan = types.ModuleType("torchtitan")
    torchtitan_distributed = types.ModuleType("torchtitan.distributed")
    torchtitan_ep = types.ModuleType("torchtitan.distributed.expert_parallel")
    torchtitan_ep.expert_parallel = lambda function: function
    sys.modules["torchtitan"] = torchtitan
    sys.modules["torchtitan.distributed"] = torchtitan_distributed
    sys.modules["torchtitan.distributed.expert_parallel"] = torchtitan_ep
    try:
        return importlib.import_module("skyrl_train.models.layers.moe")
    finally:
        del sys.modules["torchtitan.distributed.expert_parallel"]
        del sys.modules["torchtitan.distributed"]
        del sys.modules["torchtitan"]
