"""DeepEP expert-parallel sharding style — Stage 5.

Port of prime-rl ``src/prime_rl/trainer/distributed/expert_parallel.py:7-23``.

Unlike torchtitan's ``ExpertParallel`` (used by the Stage-4 torch backend), this
style installs NO ``_token_dispatch`` / ``_token_combine`` all_to_all hooks — it
only ``Shard(0)``-s the expert params over the ep submesh and stamps the ep
``ProcessGroup`` on the module as ``_ep_group``. DeepEP dispatch/combine is driven
explicitly from ``MoE._run_deepep_routed_experts`` (which calls ``get_ep_group``),
so the comm stays outside the selective-AC checkpoint boundary while the local
expert matmuls remain checkpointable (scope §2).
"""

import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh, Shard, distribute_module, distribute_tensor
from torch.distributed.tensor.parallel import ParallelStyle


class DeepEPExpertParallel(ParallelStyle):
    """Expert-parallel style backed by DeepEP dispatch/combine.

    Only handles weight sharding (Shard(0) on expert dim) and stores the EP
    process group on the module. DeepEP dispatch/combine is driven from
    ``MoE.forward()`` so communication stays outside the selective-AC checkpoint
    boundary while local expert matmuls remain checkpointable.
    """

    @staticmethod
    def _partition_fn(name: str, mod: nn.Module, device_mesh: DeviceMesh) -> None:
        for param_name, param in mod.named_parameters(recurse=False):
            mod.register_parameter(param_name, nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)])))
        mod._ep_group = device_mesh.get_group()

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(module, device_mesh, partition_fn=self._partition_fn)


def get_ep_group(experts: nn.Module) -> ProcessGroup:
    return experts._ep_group
