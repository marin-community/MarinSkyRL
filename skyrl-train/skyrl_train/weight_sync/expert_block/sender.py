"""The trainer side of an expert-block sync, held by each Megatron policy worker.

Its methods are named after vLLM's ``TrainerWeightTransferEngine`` on the marin
fork's main (``trainer_init``, ``send_weights``, ``shutdown``) so the two halves
read alike; the trainer half stays MarinSkyRL-owned, since that API drives a
rank-0 client rather than every rank.

Once per run the worker reports what it owns; the driver plans; the worker
binds its groups and views. Every sync then checks that the parameters are the
same storage the plan was built on and that the update the driver names is the
one this rank just finished, and runs the stream.
"""

from dataclasses import asdict

import torch

from skyrl_train.weight_sync.expert_block.groups import Rendezvous, create_groups, destroy_groups, warm_groups
from skyrl_train.weight_sync.expert_block.schedule import Schedule, TrainerRank, from_wire, to_wire
from skyrl_train.weight_sync.expert_block.source_views import local_expert_sources, local_source_slices
from skyrl_train.weight_sync.expert_block.stream import Stream, storage_identity


class ExpertBlockSender:
    def __init__(self, worker, parallel_state):
        self.worker = worker
        self.parallel_state = parallel_state
        self.trainer: TrainerRank | None = None
        self.sources: dict[str, torch.Tensor] = {}
        self.identity: dict[str, tuple] = {}
        self.expert_sources = {}
        self.groups = {}
        self.stream: Stream | None = None

    def inventory(self) -> dict:
        """This rank's coordinates and the exact expert matrices and dense slices it owns."""
        state = self.parallel_state
        if state.get_tensor_model_parallel_world_size() != 1:
            raise ValueError("Expert-block sync requires tensor-parallel size 1 on the trainer")
        self.trainer = TrainerRank(
            torch.distributed.get_rank(),
            state.get_expert_data_parallel_rank(),
            state.get_pipeline_model_parallel_rank(),
            state.get_expert_model_parallel_rank(),
        )
        provider = self.worker.provider
        expert_slices, dense_slices, sources = local_source_slices(
            self.worker.bridge.get_conversion_tasks(self.worker.actor_module), provider, pp=self.trainer.pp
        )
        experts = local_expert_sources(
            expert_slices,
            sources,
            self.trainer,
            num_experts=provider.num_moe_experts,
            expert_parallel_size=state.get_expert_model_parallel_world_size(),
            hidden_size=provider.hidden_size,
            intermediate_size=provider.moe_ffn_hidden_size,
        )
        self.sources = sources
        self.identity = storage_identity(sources)
        self.expert_sources = {item.entry.name: item for item in experts}
        return {
            "trainer": to_wire(self.trainer),
            "expert_parallel_size": state.get_expert_model_parallel_world_size(),
            "layers": sorted({item.entry.layer for item in experts}),
            "model": {
                "num_experts": provider.num_moe_experts,
                "hidden_size": provider.hidden_size,
                "intermediate_size": provider.moe_ffn_hidden_size,
            },
            "experts": [to_wire(item.entry) for item in experts],
            "dense": [to_wire(item) for item in dense_slices],
        }

    def trainer_init(self, init_info: dict) -> dict:
        """Create this rank's groups and resolve every view the plan needs; returns warm-up seconds per group."""
        if self.stream is not None:
            raise RuntimeError("Expert-block sender is already initialised")
        plan = from_wire(Schedule, init_info["schedule"])
        device = next(iter(self.sources.values())).device
        self.groups = create_groups(self.trainer.rank, plan.groups, Rendezvous(**init_info["rendezvous"]))
        try:
            warm = warm_groups(self.trainer.rank, plan.groups, self.groups, device)
            self.stream = Stream(
                self.trainer.rank,
                plan,
                self.groups,
                sources=self.sources,
                expert_sources=self.expert_sources,
                device=device,
            )
        except BaseException:
            destroy_groups(self.groups)
            raise
        return {"participant": self.trainer.rank, "warmup_seconds": warm}

    def send_weights(self, update_info: dict) -> dict:
        """Send this rank's blocks for one update; refused unless the weights are that update's."""
        if self.stream is None:
            raise RuntimeError("Expert-block sender is not initialised")
        version = update_info["version"]
        # Before the first update the weights are whatever was loaded; after one, the
        # sync must name the update this rank actually finished.
        completed = self.worker._model_version_step
        if completed is not None and completed != version:
            raise RuntimeError(f"Sync names update {version} but this rank last completed {completed}")
        # The parameters are held live, so a ``param.data`` reassignment since preparation shows here.
        if storage_identity(self.sources) != self.identity:
            raise RuntimeError("Policy parameter storage changed since the sync was prepared")
        return asdict(self.stream.run(version))

    def shutdown(self) -> None:
        destroy_groups(self.groups)
        self.stream = None
