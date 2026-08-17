"""Record production MoE dispatch stages on the four-node EP4/FSDP4 mesh."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torchtitan.distributed.expert_parallel as titan_expert_parallel
from torch.distributed.tensor import DeviceMesh
from torchtitan.distributed.expert_parallel import ExpertParallel

from skyrl_train.distributed.fsdp_utils import apply_ep, apply_fsdp2
from skyrl_train.models.layers.moe import MoE
from skyrl_train.models.layers.moe_swap import GroupedMoEShim
from tests.gpu.fault_injection.multi_node_geometry import EP_SIZE, FSDP_SIZE, WORLD_SIZE
from tests.gpu.fault_injection.multi_node_mesh import MeshRuntime, multi_node_mesh_runtime
from tests.moe_dispatch_stages import (
    DISPATCH_LAYERS,
    DISPATCH_MICROBATCHES,
    DispatchStage,
    DispatchStageRecord,
)
from tests.process_gang import signal_rank_ready_and_wait_for_start


DIM = 128
HIDDEN_DIM = 128
NUM_EXPERTS = EP_SIZE * FSDP_SIZE
TOP_K = 2
BATCH_SIZE = 1
SEQUENCE_LENGTHS = (17, 23, 31, 37)
SEED = 1701
PROCESS_GROUP_TIMEOUT_SECONDS = 120


class DispatchStageRecorder:
    """Wrap TorchTitan dispatch and emit externally parseable stage records."""

    def __init__(self, runtime: MeshRuntime) -> None:
        self._runtime = runtime
        self._group = runtime.mesh["ep"].get_group()
        self._ep_ranks = tuple(dist.get_process_group_ranks(self._group))
        self._microbatch = -1
        self._layer = -1
        self._routed_rows = 0

    def begin_microbatch(self, microbatch: int) -> None:
        self._microbatch = microbatch

    def enter_layer(self, layer: int, routed_rows: int) -> None:
        self._layer = layer
        self._routed_rows = routed_rows

    def emit(
        self,
        stage: DispatchStage,
        *,
        input_splits: tuple[int, ...] = (),
        output_splits: tuple[int, ...] = (),
    ) -> None:
        record = DispatchStageRecord(
            rank=self._runtime.placement.rank,
            microbatch=self._microbatch,
            layer=self._layer,
            stage=stage,
            ep_ranks=self._ep_ranks,
            sequence_number=int(self._group._get_sequence_number_for_group()),
            routed_rows=self._routed_rows,
            input_splits=input_splits,
            output_splits=output_splits,
        )
        print(record.json_line(), flush=True)

    def wrap_token_dispatch(self, original: Callable[..., object]) -> Callable[..., object]:
        def observed_dispatch(
            plan: ExpertParallel,
            module: torch.nn.Module,
            inputs: tuple[torch.Tensor, torch.Tensor],
            device_mesh: DeviceMesh,
        ) -> object:
            routed_input, _ = inputs
            self._routed_rows = routed_input.shape[0]
            self.emit(DispatchStage.ROUTING_COMPLETE)
            original_all_to_all = titan_expert_parallel.dist.all_to_all_single
            call_index = 0

            def observed_all_to_all(*args, **kwargs):
                nonlocal call_index
                group = kwargs.get("group")
                if group is not self._group:
                    return original_all_to_all(*args, **kwargs)
                if call_index == 0:
                    self.emit(DispatchStage.COUNTS_A2A_BEFORE)
                    result = original_all_to_all(*args, **kwargs)
                    self.emit(DispatchStage.COUNTS_A2A_AFTER)
                elif call_index == 1:
                    input_splits = tuple(int(value) for value in plan.input_splits)
                    output_splits = tuple(int(value) for value in plan.output_splits)
                    self.emit(
                        DispatchStage.SPLITS_CONSTRUCTED,
                        input_splits=input_splits,
                        output_splits=output_splits,
                    )
                    self.emit(
                        DispatchStage.TOKENS_A2A_BEFORE,
                        input_splits=input_splits,
                        output_splits=output_splits,
                    )
                    result = original_all_to_all(*args, **kwargs)
                    self.emit(
                        DispatchStage.TOKENS_A2A_AFTER,
                        input_splits=input_splits,
                        output_splits=output_splits,
                    )
                else:
                    raise AssertionError(f"token dispatch issued unexpected all-to-all call {call_index + 1}")
                call_index += 1
                return result

            with patch.object(titan_expert_parallel.dist, "all_to_all_single", observed_all_to_all):
                result = original(plan, module, inputs, device_mesh)
            if call_index != 2:
                raise AssertionError(f"token dispatch issued {call_index} all-to-all calls; expected 2")
            torch.cuda.synchronize(self._runtime.device)
            self.emit(
                DispatchStage.TOKENS_CUDA_COMPLETE,
                input_splits=tuple(int(value) for value in plan.input_splits),
                output_splits=tuple(int(value) for value in plan.output_splits),
            )
            return result

        return observed_dispatch


class TinyMoEBlock(torch.nn.Module):
    def __init__(self, layer: int, recorder: DispatchStageRecorder) -> None:
        super().__init__()
        self._layer = layer
        self._recorder = recorder
        self.norm = torch.nn.LayerNorm(DIM)
        self.mlp = GroupedMoEShim(
            MoE(
                dim=DIM,
                hidden_dim=HIDDEN_DIM,
                num_experts=NUM_EXPERTS,
                top_k=TOP_K,
                route_norm=True,
                use_grouped_mm=True,
            ),
            returns_tuple=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self._recorder.enter_layer(self._layer, hidden_states.shape[0] * hidden_states.shape[1] * TOP_K)
        self._recorder.emit(DispatchStage.MOE_ENTER)
        output = hidden_states + self.mlp(self.norm(hidden_states))
        self._recorder.emit(DispatchStage.MOE_EXIT)
        return output


class TinyMoEModel(torch.nn.Module):
    def __init__(self, recorder: DispatchStageRecorder) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(TinyMoEBlock(layer, recorder) for layer in range(DISPATCH_LAYERS))
        self.output = torch.nn.Linear(DIM, DIM, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.output(hidden_states)


def _build_model(runtime: MeshRuntime, recorder: DispatchStageRecorder) -> TinyMoEModel:
    torch.manual_seed(SEED)
    model = TinyMoEModel(recorder).to(device=runtime.device, dtype=torch.bfloat16)
    for module in model.modules():
        if isinstance(module, MoE):
            module.init_weights(0.02)

    fsdp_kwargs = {"mesh": runtime.mesh["fsdp"], "reshard_after_forward": True}
    original_dispatch = ExpertParallel._token_dispatch
    with patch.object(ExpertParallel, "_token_dispatch", recorder.wrap_token_dispatch(original_dispatch)):
        sharded_experts = apply_ep(model, runtime.mesh, ep_comm_backend="torch", fsdp_kwargs=fsdp_kwargs)
    if sharded_experts != DISPATCH_LAYERS:
        raise AssertionError(f"expected {DISPATCH_LAYERS} EP modules, got {sharded_experts}")
    apply_fsdp2(
        model,
        fsdp_kwargs,
        {"wrap_policy": {"transformer_layer_cls_to_wrap": ["TinyMoEBlock"]}},
    )
    return model


def _run_microbatches(runtime: MeshRuntime, recorder: DispatchStageRecorder, model: TinyMoEModel) -> None:
    sequence_length = SEQUENCE_LENGTHS[runtime.placement.fsdp_coordinate]
    for microbatch in range(DISPATCH_MICROBATCHES):
        generator = torch.Generator(device=runtime.device).manual_seed(
            SEED + runtime.placement.fsdp_coordinate * 100 + microbatch
        )
        hidden_states = torch.randn(
            BATCH_SIZE,
            sequence_length,
            DIM,
            generator=generator,
            device=runtime.device,
            dtype=torch.bfloat16,
        )
        recorder.begin_microbatch(microbatch)
        output = model(hidden_states)
        loss = output.float().square().mean()
        loss.backward()
        torch.cuda.synchronize(runtime.device)
        if not torch.isfinite(loss):
            raise AssertionError(f"non-finite loss at microbatch {microbatch}: {loss.item()}")
        model.zero_grad(set_to_none=True)


def _run(runtime: MeshRuntime, control_directory: Path) -> None:
    recorder = DispatchStageRecorder(runtime)
    model = _build_model(runtime, recorder)
    print(
        f"MOE_DISPATCH_READY rank={runtime.placement.rank} ep={runtime.placement.ep_coordinate} "
        f"fsdp={runtime.placement.fsdp_coordinate}",
        flush=True,
    )
    signal_rank_ready_and_wait_for_start(control_directory, runtime.placement.rank)
    _run_microbatches(runtime, recorder, model)
    dist.barrier()
    if runtime.placement.rank == 0:
        print(
            f"MOE_DISPATCH_OK world={WORLD_SIZE} ep={EP_SIZE} fsdp={FSDP_SIZE} "
            f"microbatches={DISPATCH_MICROBATCHES} layers={DISPATCH_LAYERS}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-directory", type=Path, required=True)
    arguments = parser.parse_args()
    with multi_node_mesh_runtime(PROCESS_GROUP_TIMEOUT_SECONDS, os.environ) as runtime:
        _run(runtime, arguments.control_directory)


if __name__ == "__main__":
    main()
