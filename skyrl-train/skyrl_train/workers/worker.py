import asyncio
import logging
import os
import socket
from datetime import timedelta
from typing import Dict, Optional, Type, List, Any, Callable
from tqdm import tqdm
from collections import defaultdict

import ray
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
import torch.distributed
from ray import ObjectRef
from ray.util.placement_group import (
    PlacementGroup,
    PlacementGroupSchedulingStrategy,
    placement_group,
    placement_group_table,
)

from skyrl_train.utils import ray_noset_visible_devices, get_ray_pg_ready_with_timeout, get_reordered_bundle_indices
from skyrl_train.utils.constants import SKYRL_RAY_PG_TIMEOUT_IN_S, SKYRL_WORKER_NCCL_TIMEOUT_IN_S
from skyrl_train.utils.io import io
from skyrl_train.utils.ppo_utils import masked_mean
from skyrl_train.distributed.dispatch import MeshRank, ActorInfo, DispatchRegistry, Dispatch
from skyrl_train.distributed.strategy import DistributedStrategy
from transformers import PreTrainedModel
from loguru import logger
from skyrl_train.distributed.ulysses import set_ulysses_sequence_parallel_group, apply_monkey_patch
from skyrl_train.distributed.utils import init_custom_process_group
from skyrl_train.utils.ppo_utils import PolicyLossRegistry, ppo_critic_loss, compute_approx_kl
from skyrl_train.workers.worker_utils import BatchIterator, reduce_metrics
from skyrl_train.dataset.replay_buffer import Experience
from skyrl_train.training_batch import TrainingInputBatch, TrainingOutputBatch
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.utils.utils import configure_ray_worker_logging, get_tcp_url
from omegaconf import DictConfig
from pathlib import Path



# Adapted from OpenRLHF: https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/trainer/ray/launcher.py#L17
class DistributedTorchRayActor:
    def __init__(
        self,
        world_size,
        rank,
        local_rank,
        master_addr,
        master_port,
        sequence_parallel_size,
        record_memory=False,
        pin_to_ray_gpu_id=False,
        force_cvd_mask=False,
    ):
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        self._world_size = world_size
        self._rank = rank
        self._local_rank = local_rank
        self._master_addr = master_addr if master_addr else self._get_current_node_ip()
        self._master_port = master_port if master_port else self._get_free_port()
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = str(self._master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        # Device pinning. `LOCAL_RANK` is consumed by every strategy's
        # setup_distributed() as the argument to torch.cuda.set_device().
        #
        # Three cases:
        #  1. ray_noset_visible_devices() True (a RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES
        #     env var is set): Ray does NOT constrain CUDA_VISIBLE_DEVICES, every actor
        #     sees all physical GPUs, so pin to this actor's Ray-assigned physical id.
        #  2. NOSET unset AND Ray constrained CUDA_VISIBLE_DEVICES to a single device for
        #     this actor (the normal/older-Ray path, e.g. the a3 8B venv runtime): device
        #     0 within the masked view IS the assigned GPU, so LOCAL_RANK="0" is correct.
        #  3. NOSET unset BUT CUDA_VISIBLE_DEVICES is unset / empty / lists >1 device
        #     (newer Ray inside the megatron+vllm SIF no longer overrides the visible-device
        #     env var, emitting "Ray will no longer override accelerator visible devices...";
        #     in this mode ray.get_gpu_ids() is ALSO unreliable — it returned [0] for every
        #     one of the 4 per-node policy actors, so even pinning to get_gpu_ids()[0] still
        #     collapses all of them onto physical GPU 0): every actor sees all GPUs, so bind
        #     to the launcher-assigned per-node rank (rank % num_gpus_per_node = 0,1,2,3),
        #     which is deterministically distinct per node and independent of Ray's
        #     accelerator bookkeeping. This is the lever that fixes the reproducible
        #     Qwen3-Next-80B init OOM (all 4 ranks materializing the 80B on GPU 0).
        #
        # Case 2 keeps the historical behavior byte-identical (single-device mask -> "0").
        # pin_to_ray_gpu_id (case 3) engages only on the per-GPU {GPU:1}-bundle policy PG,
        # where ray.get_gpu_ids() is a reliable distinct physical id per actor. The full
        # decision lives in resolve_pinned_local_rank() (pure / unit-tested).
        from skyrl_train.utils.utils import resolve_pinned_local_rank, resolve_actor_cuda_env

        _noset = ray_noset_visible_devices()

        # Deterministic forced-CVD-mask pin (opt-in via policy_force_cvd_mask).
        # When engaged, mask this actor to its single Ray-assigned PHYSICAL GPU
        # and force PCI_BUS_ID ordering BEFORE any CUDA / EP-device-mesh init,
        # so set_device(0)/init_device_mesh/FSDP device_id can only resolve that
        # one physical GPU. This is independent of positional/LOCAL_RANK ordering
        # and of whether the SIF Ray masked CVD — closing the GH200 EP×FSDP
        # GPU-0-stacking init-OOM that set_device(LOCAL_RANK) alone could not
        # deterministically prevent. IMPORTANT: this must run before
        # torch.cuda.device_count() below touches CUDA (which would latch the
        # unmasked device set). force_cvd_mask only engages with the per-GPU
        # {GPU:1}-bundle PG, where ray.get_gpu_ids()[0] is a distinct physical id.
        if force_cvd_mask and pin_to_ray_gpu_id:
            _cuda_env = resolve_actor_cuda_env(
                noset_visible_devices=_noset,
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                ray_gpu_ids=ray.get_gpu_ids(),
            )
            for _k, _v in _cuda_env.items():
                os.environ[_k] = _v
            logging.info(
                "[device-pin] force_cvd_mask: applied %s (ray_gpu_ids=%s)",
                _cuda_env,
                ray.get_gpu_ids(),
            )

        os.environ["LOCAL_RANK"] = resolve_pinned_local_rank(
            noset_visible_devices=_noset,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            ray_gpu_ids=ray.get_gpu_ids(),
            launcher_local_rank=local_rank,
            device_count=torch.cuda.device_count(),
            pin_to_ray_gpu_id=pin_to_ray_gpu_id,
        )
        self.sequence_parallel_size: int = sequence_parallel_size

        self.record_memory = record_memory
        if record_memory:
            torch.cuda.memory._record_memory_history()
        configure_ray_worker_logging()

    def get_node_local_rank(self):
        return self._local_rank

    def init_worker_process_group(self):
        if not torch.distributed.is_initialized():
            # Default torch dist pg init timeout is 10 minutes (600 seconds)
            torch.distributed.init_process_group(
                backend="nccl", timeout=timedelta(seconds=SKYRL_WORKER_NCCL_TIMEOUT_IN_S)
            )

        # setup device mesh
        # TODO: Support TP / PP for DeepSpeed
        # NOTE (sumanthrh): Device mesh and mesh rank are rank specific attributes. For the current way the strategy is defined, it is only meant to interact with worker state; not hold worker state. Thus, this should live outside the strategy object.
        # This device mesh can be common across all the strategies we use
        dp_size = self._world_size // self.sequence_parallel_size
        device_mesh = torch.distributed.device_mesh.init_device_mesh(
            "cuda", mesh_shape=(dp_size, self.sequence_parallel_size), mesh_dim_names=("dp", "sp")
        )
        self.device_mesh = device_mesh
        self.mesh_rank = MeshRank(
            dp=self.device_mesh.get_local_rank(mesh_dim="dp"),
            sp=self.device_mesh.get_local_rank(mesh_dim="sp"),
            tp=0,
            pp=0,
            world_size=self._world_size,
            dp_size=self.device_mesh.size(0),
            pp_size=1,
        )

    def _seq_parallel_monkey_patch(self, model: PreTrainedModel, use_parent_class: bool = False):
        # NOTE (sumanthrh): This sets a global variable that is used during the forward pass for sequence parallelism
        # This works because each worker is it's own process and thus different worker types are isolated
        # TODO (sumanthrh): We should re-visit this and see if we should adopt a context-manager pattern for sequence parallelism
        if self.sequence_parallel_size > 1:
            set_ulysses_sequence_parallel_group(self.device_mesh["sp"].get_group())
            apply_monkey_patch(
                model=model, ulysses_sp_size=self.sequence_parallel_size, use_parent_class=use_parent_class
            )

    def get_mesh_rank(self):
        return self.mesh_rank

    def get_gpu_id(self):
        return ray.get_gpu_ids()[0]

    @staticmethod
    def _get_current_node_ip():
        # Debug: understand where the IP comes from
        import socket
        hostname = socket.gethostname()

        # Check if Ray has a global node set
        global_node = ray._private.worker._global_node
        global_node_ip = global_node.node_ip_address if global_node else "None (no global node)"

        # What does get_node_ip_address() return?
        address = ray._private.services.get_node_ip_address()

        logging.info(f"[ipv4-debug] hostname={hostname}")
        logging.info(f"[ipv4-debug] _global_node.node_ip_address={global_node_ip}")
        logging.info(f"[ipv4-debug] get_node_ip_address()={address}")

        # strip ipv6 address brackets if present
        result = address.strip("[]")
        return result

    def get_ray_node_id(self):
        return ray.get_runtime_context().get_node_id()

    @staticmethod
    def _get_free_port():
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    def get_master_addr_port(self):
        return self._master_addr, self._master_port

    def _set_numa_affinity(self, rank):
        """Set CPU + memory affinity to match the GPU for this rank.

        Uses shared NUMA utility that auto-detects GPU-to-CPU NUMA topology
        via nvidia-smi topo. Handles GH200 unified memory correctly.

        The NUMA binding must key off the PHYSICAL GPU id (sysfs/PCI-ordered,
        as nvidia-smi sees it), not a positional/logical index. Resolution
        order for the physical id:
          1. CUDA_VISIBLE_DEVICES[rank] — when Ray masked CVD, its entries are
             the physical ids this process can see (the historical path).
          2. LOCAL_RANK env — when CVD is unset (SIF Ray) but device pinning
             ran in __init__, LOCAL_RANK already holds the physical id we
             selected (ray.get_gpu_ids()[0] in the per-GPU-bundle path), so
             NUMA binds to the SAME physical GPU torch.cuda.set_device() chose.
             This corrects the prior `gpu_id = rank` fallback, which on GH200
             could bind a different physical socket than the device in use,
             because logical (rank) and physical ordering differ.
          3. positional rank — last resort if neither is available.
        """
        try:
            from skyrl_train.utils.numa import set_numa_affinity_for_gpu
            cuda_devs = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if cuda_devs[0]:
                gpu_id = int(cuda_devs[rank])
            else:
                _lr = os.environ.get("LOCAL_RANK")
                gpu_id = int(_lr) if _lr not in (None, "", "-1") else rank
            set_numa_affinity_for_gpu(gpu_id)
        except Exception as e:
            logger.debug(f"NUMA affinity setup skipped: {e}")


class Worker(DistributedTorchRayActor):
    def __init__(self, cfg: DictConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg

    def init_model(self, *args, **kwargs):
        """Initialize worker state (model, and optimizer if applicable) on worker."""
        raise NotImplementedError()

    def empty_cache(self) -> None:
        """Empty GPU memory cache on Worker's CUDA device"""
        torch.cuda.empty_cache()

    def get_device_placement_diag(self) -> dict:
        """Diagnostic: report the PHYSICAL device this rank actually landed on,
        plus its EP/FSDP device-mesh coordinate. Used by the EP-reproducing
        placement smoke to assert every EP×FSDP rank is on a distinct physical
        GPU. Read-only; safe to call after init_model.
        """
        import socket as _socket

        idx = torch.cuda.current_device()
        try:
            uuid = str(torch.cuda.get_device_properties(idx).uuid)
        except Exception:
            uuid = None
        diag = {
            "rank": int(os.environ.get("RANK", "-1")),
            "host": _socket.gethostname(),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER"),
            "LOCAL_RANK": os.environ.get("LOCAL_RANK"),
            "ray_gpu_ids": [str(g) for g in ray.get_gpu_ids()],
            "device_count": torch.cuda.device_count(),
            "current_device": idx,
            "phys_uuid": uuid,
        }
        # EP/FSDP mesh coordinate, when the strategy built a device mesh.
        strat = getattr(self, "strategy", None)
        mesh = getattr(strat, "device_mesh", None) if strat is not None else None
        if mesh is not None:
            try:
                diag["mesh_shape"] = tuple(mesh.mesh.shape)
                diag["mesh_dim_names"] = tuple(mesh.mesh_dim_names)
                diag["mesh_coord"] = tuple(int(c) for c in mesh.get_coordinate())
                diag["ep_size"] = int(getattr(strat, "ep_size", 1))
            except Exception as e:
                diag["mesh_error"] = repr(e)
        return diag

    def offload_to_cpu(self, pin_memory=True, non_blocking=True):
        """Offload all worker state to CPU.

        After this function runs, only temporary reserved memory and torch's pre-loaded cuda kernels (~ GB) will remain

        Args:
            pin_memory: Whether to use pinned/ paged-locked memory on CPU
            non_blocking: Whether the operation is non-blocking
        """
        raise NotImplementedError()

    def backload_to_gpu(self, non_blocking=True):
        """Backload worker state to GPU

        Args:
            non_blocking: Whether the operation is non-blocking
        """
        raise NotImplementedError()

    def get_cuda_memory(self) -> Dict[str, Any]:
        """Get CUDA memory usage on worker's CUDA device."""
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        return {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "free": free,
            "total": total,
        }

    def save_memory_snapshot(self, global_step=None, local_step=None):
        """Save a snapshot of memory usage on the Worker's CUDA device.

        .. note::
            This function should be called on all the ranks in the worker group simultaneously.
        """
        rank = torch.distributed.get_rank()
        save_path = os.path.join(self.cfg.trainer.ckpt_path, "memory_snapshots")
        if self._local_rank == 0 and not io.exists(save_path):
            io.makedirs(save_path, exist_ok=True)
        torch.distributed.barrier()
        if global_step is None or local_step is None:
            file_name = f"policy_rank_{rank}.pickle"
        else:
            file_name = f"policy_rank_{rank}_training_step_{global_step}_{local_step}.pickle"
        record_memory_path = os.path.join(save_path, file_name)
        if io.exists(record_memory_path):
            # seeing issues if we don't remove the file first
            io.remove(record_memory_path)
        torch.cuda.memory._dump_snapshot(record_memory_path)

    async def init_weight_sync_state(self, inference_engine_client: InferenceEngineClient):
        """Initialize state for weight syncing with Inference Engine Client

        Initializes a custom process group with the rank 0 Worker and all the inference engine ranks
        for weight syncing.

        .. note::
            This function should be called on all the ranks in the worker group simultaneously.
        """
        assert inference_engine_client is not None

        if torch.distributed.get_rank() == 0:
            # Debug: understand where master_addr comes from
            import socket as sock_module
            hostname = sock_module.gethostname()
            global_node = ray._private.worker._global_node
            global_node_ip = global_node.node_ip_address if global_node else "None"

            master_addr = ray._private.services.get_node_ip_address()

            logging.info(f"[weight-sync] hostname={hostname}")
            logging.info(f"[weight-sync] _global_node.node_ip_address={global_node_ip}")
            logging.info(f"[weight-sync] get_node_ip_address()={master_addr}")

            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            logging.info(f"[weight-sync] Using master_addr={master_addr}, master_port={master_port}")

            num_inference_engines, tensor_parallel_size, pipeline_parallel_size, data_parallel_size = (
                self.cfg.generator.num_inference_engines,
                self.cfg.generator.inference_engine_tensor_parallel_size,
                self.cfg.generator.inference_engine_pipeline_parallel_size,
                self.cfg.generator.inference_engine_data_parallel_size,
            )
            world_size = num_inference_engines * tensor_parallel_size * pipeline_parallel_size * data_parallel_size + 1

            backend = self.cfg.generator.weight_sync_backend

            override_existing = False if self.cfg.generator.override_existing_update_group == "disable" else True
            group_name = "skyrl"
            self._model_update_group_name = group_name

            tasks = []
            tasks.append(
                inference_engine_client.init_weight_update_communicator(
                    master_addr=master_addr,
                    master_port=master_port,
                    rank_offset=1,
                    world_size=world_size,
                    group_name=group_name,
                    backend=backend,
                    override_existing=override_existing,
                )
            )

            tasks.append(
                asyncio.to_thread(
                    init_custom_process_group,
                    backend=backend,
                    init_method=get_tcp_url(master_addr, master_port),
                    world_size=world_size,
                    rank=0,
                    group_name=group_name,
                )
            )
            results = await asyncio.gather(*tasks)
            self._model_update_group = results[-1]

            # # Register signal handlers for termination only on rank 0
            # NOTE (sumanthrh): This doesn't work yet, and is thus commented out.
            # The better way is to just have this specified in __del__, but there is
            # no guarattee that __del__ will be called in general. Ray also doesn't
            # explictly call __del__ when the actor shuts down.
            # It's commented out so that we can fix this in the future.
            # atexit.register(self._handle_termination)

        torch.distributed.barrier()

    def forward(
        self,
        data: TrainingInputBatch,
    ) -> TrainingOutputBatch:
        """Run forward pass on the input batch in inference mode.

        This is a wrapper around `_forward_micro_batch` that runs in micro batches of `cfg.trainer.micro_forward_batch_size_per_gpu`.
        """
        # run in micro batches of cfg.trainer.micro_forward_batch_size_per_gpu
        # TODO (sumanthrh): this can be in the policy/critic impl if the micro batch size can be specific to policy, critic, etc.
        micro_batches = data.chunk(self.cfg.trainer.micro_forward_batch_size_per_gpu)

        outputs = []
        for micro_batch in micro_batches:
            outputs.append(self._forward_micro_batch(micro_batch))
        output = TrainingOutputBatch.cat(outputs)
        if output.device is not None and output.device != torch.device("cpu"):
            output = output.to("cpu")
        return output

    def _forward_micro_batch(self, micro_batch: TrainingInputBatch) -> TrainingOutputBatch:
        raise NotImplementedError()


# adapted from OpenReasonerZero: https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/orz/ppo/actors.py
class PPORayActorGroup:
    """
    A group of ray actors
    Functions start with 'async' should return list of object refs

    Args:
        cfg: config object for workers
        num_nodes (int): Number of nodes for this actor group.
        num_gpus_per_node (int): Number of gpus for this actor group.
        ray_actor_type (Type[Worker]): PPO model type that this actor group serve on.
        pg (PlacementGroup, optional): Placement group to schedule actor on.
            If none, create new placement group automatically. Defaults to None.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each actor.
            If < 1.0, multiple models can share same gpu. Defaults to 1.
    """

    def __init__(
        self,
        cfg,
        num_nodes,
        num_gpus_per_node,
        ray_actor_type: Type[Worker],
        pg: Optional[PlacementGroup] = None,
        num_gpus_per_actor: float = 1.0,
        resources: Optional[Dict[str, float]] = None,
        num_resources_per_node: Optional[int] = None,
        colocate_all: bool = False,
        sequence_parallel_size: int = 1,
        record_memory: bool = False,
        pin_to_ray_gpu_id: bool = False,
        force_cvd_mask: bool = False,
    ) -> None:
        self.cfg = cfg
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self.ray_actor_type = ray_actor_type
        # When True, each actor pins its CUDA device to ray.get_gpu_ids()[0]
        # (reliable only when each actor owns a dedicated {GPU:1} bundle, i.e.
        # the per-GPU-bundle policy PG). See DistributedTorchRayActor.__init__.
        self._pin_to_ray_gpu_id = pin_to_ray_gpu_id
        # When True (and pin_to_ray_gpu_id), each actor additionally MASKS
        # CUDA_VISIBLE_DEVICES to its single physical GPU + forces PCI_BUS_ID
        # ordering before any CUDA init — the deterministic EP×FSDP pin.
        self._force_cvd_mask = force_cvd_mask

        # custom resources, see https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        self._resources = resources
        self._num_resources_per_node = num_resources_per_node

        self.colocate_all = colocate_all
        self.sequence_parallel_size = sequence_parallel_size
        self.record_memory = record_memory
        self._initiate_actors(pg, num_gpus_per_actor)

    def _initiate_actors(self, pg: Optional[PlacementGroup], num_gpus_per_actor: float):
        """Initialize Ray actors in the worker group.

        Args:
            pg: The placement group for the worker group
            num_gpus_per_actor: The number of gpus to allocate per actor.
        """
        world_size = self._num_nodes * self._num_gpus_per_node
        if self.colocate_all:
            assert (
                pg is not None
            ), "if colocate_all is True, the shared placement group must be provided to PPORayActorGroup"
            pg_data = placement_group_table(pg)
            assert (
                len(pg_data["bundles"]) == world_size
            ), "if colocate_all is True, the number of bundles in the shared placement group must match the world size"

        reordered_bundle_indices = []
        if pg is not None:
            pg_data = placement_group_table(pg)
            should_reorder_bundles = len(pg_data["bundles"]) == world_size
            if should_reorder_bundles:
                reordered_bundle_indices = get_reordered_bundle_indices(pg)

        if self._num_gpus_per_node > 1 and pg is None:
            bundles = [{"GPU": self._num_gpus_per_node, "CPU": self._num_gpus_per_node} for _ in range(self._num_nodes)]
            if self._resources:
                resources_name = list(self._resources.keys())[0]
                for i in range(len(bundles)):
                    bundles[i][resources_name] = self._num_resources_per_node

            pg = placement_group(bundles, strategy="PACK")
            get_ray_pg_ready_with_timeout(pg, timeout=SKYRL_RAY_PG_TIMEOUT_IN_S)
        if pg:
            master_actor = self.ray_actor_type.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=reordered_bundle_indices[0] if reordered_bundle_indices else 0,
                ),
            ).remote(
                cfg=self.cfg,
                world_size=world_size,
                rank=0,
                local_rank=0,
                master_addr=None,
                master_port=None,
                sequence_parallel_size=self.sequence_parallel_size,
                record_memory=self.record_memory,
                pin_to_ray_gpu_id=self._pin_to_ray_gpu_id,
                force_cvd_mask=self._force_cvd_mask,
            )
        else:
            master_actor = self.ray_actor_type.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
            ).remote(
                cfg=self.cfg,
                world_size=world_size,
                rank=0,
                local_rank=0,
                master_addr=None,
                master_port=None,
                sequence_parallel_size=self.sequence_parallel_size,
                record_memory=self.record_memory,
                pin_to_ray_gpu_id=self._pin_to_ray_gpu_id,
                force_cvd_mask=self._force_cvd_mask,
            )
        self._actor_handlers = [master_actor]
        # Create worker actors
        if world_size > 1:
            master_addr, master_port = ray.get(master_actor.get_master_addr_port.remote())
            for rank in range(1, world_size):
                local_rank = rank % self._num_gpus_per_node

                if pg:
                    worker_actor = self.ray_actor_type.options(
                        num_cpus=num_gpus_per_actor,
                        num_gpus=num_gpus_per_actor,
                        resources=self._resources,
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            placement_group_bundle_index=(
                                reordered_bundle_indices[rank]
                                if reordered_bundle_indices
                                else rank // self._num_gpus_per_node
                            ),
                        ),
                    ).remote(
                        cfg=self.cfg,
                        world_size=world_size,
                        rank=rank,
                        local_rank=local_rank,
                        master_addr=master_addr,
                        master_port=master_port,
                        sequence_parallel_size=self.sequence_parallel_size,
                        record_memory=self.record_memory,
                        pin_to_ray_gpu_id=self._pin_to_ray_gpu_id,
                        force_cvd_mask=self._force_cvd_mask,
                    )
                else:
                    worker_actor = self.ray_actor_type.options(
                        num_cpus=num_gpus_per_actor,
                        num_gpus=num_gpus_per_actor,
                        resources=self._resources,
                    ).remote(
                        cfg=self.cfg,
                        world_size=world_size,
                        rank=rank,
                        local_rank=local_rank,
                        master_addr=master_addr,
                        master_port=master_port,
                        sequence_parallel_size=self.sequence_parallel_size,
                        record_memory=self.record_memory,
                        pin_to_ray_gpu_id=self._pin_to_ray_gpu_id,
                        force_cvd_mask=self._force_cvd_mask,
                    )
                self._actor_handlers.append(worker_actor)

        # Initialize process group
        logger.info("Initializing process group for RayActorGroup")
        ray.get([actor.init_worker_process_group.remote() for actor in self._actor_handlers])
        logger.info("Initialized process group for RayActorGroup")
        self.actor_infos = [ActorInfo(actor, ray.get(actor.get_mesh_rank.remote())) for actor in self._actor_handlers]
        logger.info(f"Mesh Ranks: {[actor_info.rank for actor_info in self.actor_infos]}")

    def async_init_model(
        self,
        *args,
        **kwargs,
    ) -> List[ObjectRef]:
        """Asynchronously initialize worker state (model, and optimizer if applicable) from model path on all the workers.

        Returns:
            A list of ray object refs.
        """
        return [actor.init_model.remote(*args, **kwargs) for actor in self._actor_handlers]

    def offload_to_cpu(self, nonblocking=False, offload_optimizer=True, offload_model=True):
        """Offload all worker state to CPU.

        Args:
            nonblocking: Whether this operation is synchronous or asynchronous.
            If `nonblocking=True`, then the function returns a list of object refs.
        """
        refs = [
            actor.offload_to_cpu.remote(offload_optimizer=offload_optimizer, offload_model=offload_model)
            for actor in self._actor_handlers
        ]
        if nonblocking:
            return refs
        return ray.get(refs)

    def backload_to_gpu(self, nonblocking=False, backload_optimizer=True, backload_model=True):
        """Backload worker state to GPU

        Args:
            nonblocking: Whether this operation is synchronous or asynchronous.
            If `nonblocking=True`, then the function returns a list of ObjectRefs.
        """
        refs = [
            actor.backload_to_gpu.remote(backload_optimizer=backload_optimizer, backload_model=backload_model)
            for actor in self._actor_handlers
        ]
        if nonblocking:
            return refs
        return ray.get(refs)

    def run_method(self, dispatch_type: str, method_name: str, *args, **kwargs) -> Optional[TrainingOutputBatch]:
        """Run a method on all actors using specified dispatch type synchronously.

        The method should either return `None` or a `TrainingOutputBatch` object.

        Args:
            dispatch_type: Type of dispatch to use ("mesh" or "pass_through")
            method_name: Name of the method to call on actors
            *args: Positional arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method

        Returns:
            Collect results from all the actors.
        """
        dispatch_class: Dispatch = DispatchRegistry.get(dispatch_type)
        # validate the dispatch args to be sent to `.dispatch`
        args, kwargs = dispatch_class.validate_dispatch_args(*args, **kwargs)

        # Dispatch the method call
        object_refs = dispatch_class.dispatch(self.actor_infos, method_name, *args, **kwargs)
        # Collect results from all the actors
        ret = dispatch_class.sync_collect(self.actor_infos, object_refs)
        return ret

    def async_run_ray_method(self, dispatch_type: str, method_name: str, *args, **kwargs) -> List[ObjectRef]:
        """Run a method on all actors using specified dispatch type asynchronously.

        Args:
            dispatch_type: Type of dispatch to use ("mesh" or "pass_through")
            method_name: Name of the method to call on actors
            *args: Positional arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method

        Returns:
            List of object references
        """
        dispatch_class: Dispatch = DispatchRegistry.get(dispatch_type)
        # validate the dispatch args to be sent to `.dispatch`
        args, kwargs = dispatch_class.validate_dispatch_args(*args, **kwargs)

        # Dispatch the method call
        object_refs = dispatch_class.dispatch(self.actor_infos, method_name, *args, **kwargs)
        return object_refs

    async def async_run_method(
        self, dispatch_type: str, method_name: str, *args, **kwargs
    ) -> Optional[TrainingOutputBatch]:
        """Run a method on all actors using specified dispatch type in an asyncio-compatible way.

        Args:
            dispatch_type: Type of dispatch to use ("mesh" or "pass_through")
            method_name: Name of the method to call on actors
            *args: Positional arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method

        Returns:
            TrainingOutputBatch: concatenated results from all actors
        """
        dispatch_class: Dispatch = DispatchRegistry.get(dispatch_type)
        # validate the dispatch args to be sent to `.dispatch`
        args, kwargs = dispatch_class.validate_dispatch_args(*args, **kwargs)

        # Dispatch the method call
        object_refs = dispatch_class.dispatch(self.actor_infos, method_name, *args, **kwargs)
        return await dispatch_class.async_collect(self.actor_infos, object_refs)

    def kill_actors(self, no_restart: bool = True) -> None:
        """Kill all Ray actors in this group for proper teardown.

        Args:
            no_restart: If True, prevents Ray from restarting the actors.
        """
        for actor in self._actor_handlers:
            try:
                ray.kill(actor, no_restart=no_restart)
            except Exception:
                pass  # Actor may already be dead


class PolicyWorkerBase(Worker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model: nn.Module = None
        self.scheduler: LRScheduler = None
        self.optimizer: Optimizer = None
        self.strategy: DistributedStrategy = None
        self.record_memory: bool = False
        self.mesh_rank: MeshRank = None
        self.policy_loss_fn: Callable = PolicyLossRegistry.get(self.cfg.trainer.algorithm.policy_loss_type)

    def _normalize_mini_batch_size(self):
        """
        Normalize mini batch sizes to per-gpu mini batch sizes..
        """
        if not hasattr(self, "mesh_rank") or self.mesh_rank is None:
            raise RuntimeError("mesh_rank must be initialized before calling _normalize_mini_batch_size()")

        dp_size = self.mesh_rank.dp_size
        self.policy_mini_batch_size_per_gpu = (
            self.cfg.trainer.policy_mini_batch_size * self.cfg.generator.n_samples_per_prompt // dp_size
        )

    def ppo_train(self, train_data: TrainingInputBatch) -> TrainingOutputBatch:
        global_step = train_data.metadata["global_step"]
        # Per-batch stale_min for StaleClip (None for sync RL, populated for async).
        stale_min = train_data.metadata.get("stale_min")
        # Lazily instantiate spike-mitigation objects on first call.
        if not hasattr(self, "_stale_clip"):
            from skyrl_train.utils.stale_clip import StaleClip
            from skyrl_train.utils.zclip import ZClip

            sc_cfg = getattr(self.cfg.trainer.algorithm, "stale_clip", None)
            zc_cfg = getattr(self.cfg.trainer.algorithm, "z_clip", None)
            self._stale_clip = StaleClip(
                alpha=getattr(sc_cfg, "alpha", 0.3) if sc_cfg is not None else 0.3,
                entropy_threshold=getattr(sc_cfg, "entropy_threshold", 0.15) if sc_cfg is not None else 0.15,
                entropy_window=getattr(sc_cfg, "entropy_window", 10) if sc_cfg is not None else 10,
                min_lr_scale=getattr(sc_cfg, "min_lr_scale", 0.1) if sc_cfg is not None else 0.1,
                enabled=getattr(sc_cfg, "enabled", False) if sc_cfg is not None else False,
            )
            self._z_clip = ZClip(
                alpha=getattr(zc_cfg, "alpha", 0.97) if zc_cfg is not None else 0.97,
                z_thresh=getattr(zc_cfg, "z_thresh", 2.5) if zc_cfg is not None else 2.5,
                warmup_steps=getattr(zc_cfg, "warmup_steps", 25) if zc_cfg is not None else 25,
                max_grad_norm=self.cfg.trainer.policy.optimizer_config.max_grad_norm,
                clip_option=getattr(zc_cfg, "clip_option", "adaptive_scaling")
                if zc_cfg is not None
                else "adaptive_scaling",
                clip_factor=getattr(zc_cfg, "clip_factor", 1.0) if zc_cfg is not None else 1.0,
                mode=getattr(zc_cfg, "mode", "zscore") if zc_cfg is not None else "zscore",
                skip_update_on_spike=getattr(zc_cfg, "skip_update_on_spike", False)
                if zc_cfg is not None
                else False,
                enabled=getattr(zc_cfg, "enabled", False) if zc_cfg is not None else False,
            )
            # If load_checkpoint() stashed prior ZClip / StaleClip state from a
            # resumed run, restore it now that the objects exist. This keeps
            # warmup_buffer + EMA stats coherent across chain-restarts.
            if hasattr(self, "_z_clip_state_to_restore"):
                self._z_clip.load_state_dict(self._z_clip_state_to_restore)
                delattr(self, "_z_clip_state_to_restore")
            if hasattr(self, "_stale_clip_state_to_restore"):
                load_fn = getattr(self._stale_clip, "load_state_dict", None)
                if load_fn is not None:
                    load_fn(self._stale_clip_state_to_restore)
                delattr(self, "_stale_clip_state_to_restore")
        # Stash for training_step to consume (avoids changing its signature).
        self._current_stale_min = stale_min
        dataloader = BatchIterator(
            train_data, sample_batch_size=self.cfg.trainer.micro_train_batch_size_per_gpu, drop_last=False
        )

        # ── Global length-unbiased normalizer (seq_mean_token_sum_norm_global only) ──
        # Compute the SINGLE global denominator Z = global_num_seqs * max_seq_len once,
        # before the epoch loop, via ONE single-scalar all_reduce(op="sum"). This is the
        # crux fix for the async/grad-accum size bias: instead of dividing each
        # micro-batch by accumulation_steps (a count -> mean-of-means), every
        # micro-batch's masked loss-SUM is divided by this one global denom, so the
        # realized objective is a single global normalization over the whole DP batch.
        #
        # NCCL-safety (avoids the log-ratio v2/v3 status-dict key-mismatch deadlock):
        # a single scalar tensor, a fixed code path every rank executes, clamp(min=1)
        # so a rank with zero non-zero-advantage sequences still contributes a valid
        # reduce. This is a no-op for every other loss_reduction (gated).
        if self.cfg.trainer.algorithm.loss_reduction == "seq_mean_token_sum_norm_global":
            advantages_all = train_data["advantages"]
            # Count sequences with a non-zero advantage locally (zero-advantage seqs
            # -- excluded / k<2 / zero-variance RLOO groups -- contribute no gradient,
            # so they must not inflate Z).
            local_num_seqs = float((advantages_all.abs().sum(dim=-1) > 0).sum().item())
            global_num_seqs = self.strategy.all_reduce(
                torch.tensor(local_num_seqs, device=torch.cuda.current_device()), op="sum"
            )
            global_num_seqs = float(global_num_seqs.item())
            self.cfg.trainer.algorithm.global_loss_denom = (
                max(global_num_seqs, 1.0) * self.cfg.trainer.algorithm.max_seq_len
            )

        # Clear fragmented GPU memory before training to avoid OOM at step boundaries
        # (matches CriticWorkerBase.ppo_train behavior)
        torch.cuda.empty_cache()

        micro_batches_per_mini_batch = (
            self.policy_mini_batch_size_per_gpu // self.cfg.trainer.micro_train_batch_size_per_gpu
        )
        # The number of steps (over micro batches) to accumulate gradients before taking an optimizer step.
        accumulation_steps = micro_batches_per_mini_batch

        status_list = []
        all_metrics = defaultdict(list)
        policy_update_steps = 0

        for epoch in range(self.cfg.trainer.update_epochs_per_batch):
            pbar = tqdm(
                dataloader,
                desc=f"Policy Train epoch [{epoch + 1}/{self.cfg.trainer.update_epochs_per_batch}]",
                disable=not self.strategy.is_rank_0(),
            )
            for local_step, experience in enumerate(pbar):
                status = self.training_step(
                    experience,
                    global_step,
                    local_step,
                    accumulation_steps,
                )
                policy_update_steps += 1

                # for DP
                # TODO (sumanthrh): this assumes all workers are data parallel.
                # We assume that outputs are replicated within tp or sp group, otherwise this is not correct.
                status = self.strategy.all_reduce(status)

                # weighted mean for kl
                # TODO (sumanthrh): this weighted mean is no longer correct since we use the max response length in the batch.
                # we can log this in the driver
                # if "kl" in status:
                #     status["kl"] *= status["response_length"]
                #     status["kl"] /= status["response_length"]

                short_status = {}

                if "policy_loss" in status:
                    short_status = {
                        "pg": status["policy_loss"],
                        "glen": status["response_length"],
                        "policy_lr": status["policy_lr"],
                        "ent": status["policy_entropy"],
                    }
                    if "raw_grad_norm" in status:
                        short_status["grad_norm"] = status["raw_grad_norm"]
                    if "reward" in status:
                        short_status["rm"] = status["reward"]

                if "critic_loss" in status:
                    short_status["cri"] = status["critic_loss"]
                    short_status["vals"] = status["values"]
                    short_status["cri_lr"] = status["critic_lr"]

                if "ptx_loss" in status:
                    short_status["ptx"] = status["ptx_loss"]

                status_list.append(status)
                for k, v in status.items():
                    all_metrics[k].append(v)
                pbar.set_postfix(short_status)

        torch.distributed.barrier()
        # not needed beyond status logging
        all_metrics.pop("response_length", None)

        status_mean = reduce_metrics(all_metrics)
        status_mean["policy_update_steps"] = policy_update_steps / accumulation_steps

        # should return an `TrainingOutputBatch`
        output = TrainingOutputBatch()
        output.metadata = {"train_status": status_mean}
        return output

    def training_step(self, experience: Experience, global_step, local_step, accumulation_steps) -> Dict[str, float]:
        """
        Perform one micro-batch of training, accumulate gradients, and step the optimizer only after `accumulation_steps` micro-batches.
        """
        self.model.train()
        experience.to_device(torch.cuda.current_device())

        sequences = experience.sequences
        old_action_log_probs = experience.action_log_probs
        base_action_log_probs = (
            experience.base_action_log_probs if experience.base_action_log_probs is not None else None
        )
        advantages = experience.advantages
        num_actions = experience.num_actions
        attention_mask = experience.attention_mask
        loss_mask = experience.loss_mask
        rollout_action_logprobs = experience.rollout_logprobs
        rollout_routed_experts = experience.rollout_routed_experts

        # TODO (sumanthrh): don't think this does anything for deepspeed or fsdp rn because autocast happens internally
        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            # actor loss
            action_log_probs, output = self.model(
                sequences,
                num_actions,
                attention_mask=attention_mask,
                temperature=self.cfg.generator.sampling_params.temperature,
                return_output=True,
                compute_entropy=True,
                entropy_requires_grad=self.cfg.trainer.algorithm.use_entropy_loss,
                rollout_routed_experts=rollout_routed_experts,
            )
            # loss function
            # TODO: recompute advantages
            policy_loss, clip_ratio = self.policy_loss_fn(
                action_log_probs,
                old_action_log_probs,
                advantages,
                config=self.cfg.trainer.algorithm,
                loss_mask=loss_mask,
                rollout_logprobs=rollout_action_logprobs,
            )

        # entropy loss
        with torch.set_grad_enabled(self.cfg.trainer.algorithm.use_entropy_loss):
            # batch_size, seqlen
            entropy_BS = output["entropy"]
            entropy_BS = entropy_BS[:, -num_actions - 1 : -1]
            entropy = masked_mean(entropy_BS, loss_mask)

        if self.cfg.trainer.algorithm.use_entropy_loss:
            entropy_loss_term = entropy * self.cfg.trainer.algorithm.entropy_loss_coef
        else:
            entropy_loss_term = torch.tensor(0.0)

        # kl loss
        if self.cfg.trainer.algorithm.use_kl_loss:
            kl_loss = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                loss_mask=loss_mask,
                kl_estimator_type=self.cfg.trainer.algorithm.kl_estimator_type,
            )
            kl_loss = masked_mean(kl_loss, loss_mask, dim=-1).mean()
        else:
            kl_loss = torch.tensor(0.0)
        kl_loss_term = kl_loss * self.cfg.trainer.algorithm.kl_loss_coef

        if self.cfg.trainer.algorithm.loss_reduction == "seq_mean_token_sum_norm_global":
            # The policy term is already normalized by the SINGLE global denominator
            # Z = global_num_seqs * max_seq_len (set on the driver before the epoch loop),
            # so dividing it again by accumulation_steps would double-normalize it. The
            # KL / entropy auxiliary terms are per-micro-batch means and DO still need the
            # /accumulation_steps to average correctly across the gradient-accumulation window.
            loss = policy_loss + (kl_loss_term - entropy_loss_term) / accumulation_steps
        else:
            loss = policy_loss + kl_loss_term - entropy_loss_term
            loss = loss / accumulation_steps
        self.strategy.backward(loss, self.model, self.optimizer)

        # Stage-7 P3 recompute-safety: the training forward DEFERS the router-replay
        # teardown to here (after backward) so gradient-checkpoint recompute still
        # sees the installed controller -> no CheckpointError. Idempotent + a no-op
        # when replay is disabled (flag-off / a3 path byte-identical).
        teardown_replay = getattr(self.model, "teardown_router_replay", None)
        if teardown_replay is not None:
            teardown_replay()

        # Per-token log-ratio diagnostics — v5 accumulates across all micro-batches
        # of the global_step (sum + count + concat-of-topk) and finalizes once at
        # the end. v4 ran only on the LAST micro-batch, which with
        # `update_epochs_per_batch=1` gave a noisy single-sample view (action_log_probs
        # is always computed BEFORE optimizer_step within a step, so the per-batch
        # ratio reflects only vLLM↔FSDP precision noise — averaging across all
        # micro-batches makes that signal more representative). The final scalar
        # dict has the same wandb keys as v4 so the downstream per-key
        # all_reduce(status) stays keyset-compatible.
        from skyrl_train.utils.ppo_utils import (
            _empty_log_ratio_accumulator,
            compute_log_ratio_partial,
            merge_log_ratio_partial,
            finalize_log_ratio_metrics,
            _log_ratio_diag_zero_metrics,
        )
        if local_step % accumulation_steps == 0 or getattr(self, "_ratio_diag_acc", None) is None:
            self._ratio_diag_acc = _empty_log_ratio_accumulator(device=action_log_probs.device)
        try:
            partial = compute_log_ratio_partial(
                log_probs=action_log_probs,
                old_log_probs=old_action_log_probs,
                loss_mask=loss_mask,
            )
            merge_log_ratio_partial(self._ratio_diag_acc, partial)
        except Exception as _e:
            logger.warning(f"compute_log_ratio_partial failed at local_step={local_step}: {_e!r}; skipping this micro-batch")

        grad_norm = None
        ratio_diag = {}
        spike_diag = {}
        if (local_step + 1) % accumulation_steps == 0:
            # StaleClip: read rolling entropy from prior steps' history; decide LR scale
            # for THIS step's optimizer.step(). The current micro-batch entropy is pushed
            # to the history AFTER the step so it informs the next step (decoupling
            # the decision from the value that step itself produced).
            stale_clip = getattr(self, "_stale_clip", None)
            z_clip = getattr(self, "_z_clip", None)
            stale_min = getattr(self, "_current_stale_min", None)
            lr_scale = stale_clip.compute_lr_scale(stale_min) if stale_clip is not None else 1.0

            grad_norm = self.strategy.optimizer_step(
                self.optimizer,
                self.model,
                self.scheduler,
                name="actor",
                z_clip=z_clip,
                stale_clip_lr_scale=lr_scale,
            )
            if grad_norm is not None:
                grad_norm = grad_norm.detach().cpu().item()

            # Now push this step's entropy to the rolling window for next step.
            if stale_clip is not None:
                # All-reduce the entropy across DP ranks so every rank pushes
                # an IDENTICAL value into its rank-local entropy_history. Without
                # this, each rank's rolling_entropy diverges from the others,
                # ranks straddle the entropy_threshold differently, and they
                # make different `triggered`/`scale` decisions inside the same
                # optimizer step — drifting the parameter shards apart.
                # Smoking gun (Perlmutter 52905223): metrics show
                #   triggered=0.625 / scale=0.8125 at stale_min=1
                # = 5/8 ranks applying scale=0.7 and 3/8 applying 1.0.
                entropy_global = self.strategy.all_reduce(entropy.item(), op="mean")
                stale_clip.update_entropy(entropy_global)

            # Surface decisions for logging.
            if stale_clip is not None and stale_clip.enabled:
                for k, v in stale_clip.last_decision.items():
                    if isinstance(v, (int, float)):
                        spike_diag[f"stale_clip/{k}"] = float(v)
            if z_clip is not None and z_clip.enabled:
                for k, v in z_clip.last_decision.items():
                    if isinstance(v, (int, float)):
                        spike_diag[f"z_clip/{k}"] = float(v)

            # Finalize the accumulated diagnostics. Every rank must emit identical
            # keys (the full set from _log_ratio_diag_zero_metrics) — the per-key
            # all_reduce(status) deadlocks otherwise (killed v2/v3 of this diag).
            try:
                ratio_diag = finalize_log_ratio_metrics(self._ratio_diag_acc)
            except Exception as _e:
                logger.warning(f"finalize_log_ratio_metrics failed: {_e!r}; emitting zeros")
                ratio_diag = _log_ratio_diag_zero_metrics()
            self._ratio_diag_acc = None  # reset for next global_step

        if self.record_memory:
            self.save_memory_snapshot(global_step, local_step)

        # status
        status = {
            "final_loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "policy_lr": self.scheduler.get_last_lr()[0],
            "ppo_clip_ratio": clip_ratio,
            "policy_entropy": entropy.item(),
        }
        # Per-token log-ratio diagnostics — visibility into which tokens
        # carry the gradient signal (heaviest-hit token, fraction of tokens
        # with large probability changes, per-position aggregations).
        # Trainer prefixes these with "policy/" before sending to wandb.
        status.update(ratio_diag)
        # Spike-mitigation decisions (StaleClip / ZClip). Empty dict when disabled.
        status.update(spike_diag)
        if self.cfg.trainer.algorithm.use_kl_loss:
            status["policy_kl"] = kl_loss.item()

        if grad_norm is not None:
            status["raw_grad_norm"] = grad_norm

        for k, v in experience.info.items():
            if k == "kl":
                # just use the same value as loss if available
                status[k] = kl_loss.item() if isinstance(kl_loss, torch.Tensor) else status["policy_kl"]
            else:
                status[k] = v.mean().item() if isinstance(v, torch.Tensor) else v

        status["response_length"] = num_actions
        return status

    def save_checkpoint(self, ckpt_dir: Path, tokenizer=None):
        # Persist ZClip / StaleClip state alongside the model so warmup
        # counters and EMA stats survive chain-restarts. Without this,
        # warmup_buffer resets to [] on every resume and (with default
        # warmup_steps=25 + 60-80 step ablations) ZClip never engages.
        client_state = {}
        if hasattr(self, "_z_clip") and self._z_clip is not None:
            client_state["z_clip_state"] = self._z_clip.state_dict()
        if hasattr(self, "_stale_clip") and self._stale_clip is not None:
            sc_state = getattr(self._stale_clip, "state_dict", lambda: None)()
            if sc_state is not None:
                client_state["stale_clip_state"] = sc_state
        self.strategy.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            ckpt_dir=ckpt_dir,
            node_local_rank=self.get_node_local_rank(),
            tokenizer=tokenizer,
            client_state=client_state,
        )

    def load_checkpoint(
        self, ckpt_dir: Path, load_optimizer_states: bool = True, load_lr_scheduler_states: bool = True
    ):
        _, states = self.strategy.load_checkpoint(
            model=self.model,
            optimizer=self.optimizer if load_optimizer_states else None,
            scheduler=self.scheduler if load_lr_scheduler_states else None,
            ckpt_dir=ckpt_dir,
            load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_lr_scheduler_states,
        )
        # Restore ZClip / StaleClip state if present. The actual ZClip/StaleClip
        # objects are lazy-instantiated on the first ppo_train() call, so we
        # stash the loaded state on self and apply it inside ppo_train.
        client_state = (states or {}).get("client_state") or {}
        if "z_clip_state" in client_state:
            self._z_clip_state_to_restore = client_state["z_clip_state"]
        if "stale_clip_state" in client_state:
            self._stale_clip_state_to_restore = client_state["stale_clip_state"]
        return states

    def save_hf_model(self, export_dir: str, tokenizer):
        # Save model in HuggingFace safetensors format
        self.strategy.save_hf_model(
            self.model,
            export_dir,
            tokenizer=tokenizer,
        )

    def _forward_micro_batch(self, micro_batch: TrainingInputBatch) -> TrainingOutputBatch:
        device = torch.cuda.current_device()
        micro_batch.to(device)
        self.model.eval()
        sequences = micro_batch["sequences"]
        response_length = micro_batch.metadata["response_length"]
        attention_mask = micro_batch["attention_mask"]
        # MoE router-replay (R3): replay the SAME captured routing as the training
        # forward so the old-logprob (pi_old) pass and the training pass share the
        # identical forward path. Without this the old-logprob pass falls through to
        # NATIVE top-k routing while training uses REPLAY routing -> different
        # experts -> a pathological step-1 importance ratio. Absent key (8B /
        # router-replay off) -> None -> stock native forward, unchanged.
        rollout_routed_experts = (
            micro_batch["rollout_routed_experts"]
            if "rollout_routed_experts" in micro_batch.keys()
            else None
        )

        with torch.no_grad(), torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            policy_logprob = self.model(
                sequences,
                response_length,
                attention_mask,
                return_output=False,
                temperature=self.cfg.generator.sampling_params.temperature,
                rollout_routed_experts=rollout_routed_experts,
            )
        policy_logprob = policy_logprob.to("cpu")
        output = TrainingOutputBatch(
            {"output": policy_logprob},
        )
        output.metadata = micro_batch.metadata
        return output

    def process_sequences(self, sequences, input_len, eos_token_id, pad_token_id):
        return self.model.process_sequences(sequences, input_len, eos_token_id, pad_token_id)


class CriticWorkerBase(Worker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model: nn.Module = None
        self.scheduler: LRScheduler = None
        self.optimizer: Optimizer = None
        self.strategy: DistributedStrategy = None
        self.record_memory: bool = False
        self.mesh_rank: MeshRank = None
        self.critic_loss_fn: Callable = ppo_critic_loss

    def _normalize_mini_batch_size(self):
        """
        Normalize batch sizes based on device mesh and generation parameters.
        """
        if not hasattr(self, "mesh_rank") or self.mesh_rank is None:
            raise RuntimeError("mesh_rank must be initialized before calling _normalize_mini_batch_size()")

        dp_size = self.mesh_rank.dp_size
        self.critic_mini_batch_size_per_gpu = (
            self.cfg.trainer.critic_mini_batch_size * self.cfg.generator.n_samples_per_prompt // dp_size
        )

    def _forward_micro_batch(
        self,
        micro_batch: TrainingInputBatch,
    ) -> TrainingOutputBatch:
        """Generates critic values."""
        device = torch.cuda.current_device()
        micro_batch.to(device)
        sequences = micro_batch["sequences"]
        response_length = micro_batch.metadata["response_length"]
        attention_mask = micro_batch["attention_mask"]
        self.model.eval()
        with torch.no_grad(), torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            value = self.model(
                sequences,
                response_length,
                attention_mask,
            )
        self.model.train()  # reset model state
        value = value.to("cpu")
        output = TrainingOutputBatch(
            {"output": value},
        )
        output.metadata = micro_batch.metadata
        return output

    def save_hf_model(self, export_dir: str, tokenizer):
        # Save model in HuggingFace safetensors format
        self.strategy.save_hf_model(
            self.model,
            export_dir,
            tokenizer=tokenizer,
        )

    def ppo_train(self, train_data: TrainingInputBatch) -> TrainingOutputBatch:
        global_step = train_data.metadata["global_step"]
        dataloader = BatchIterator(
            train_data, sample_batch_size=self.cfg.trainer.micro_train_batch_size_per_gpu, drop_last=False
        )

        torch.cuda.empty_cache()
        self.model.train()

        micro_batches_per_mini_batch = (
            self.critic_mini_batch_size_per_gpu // self.cfg.trainer.micro_train_batch_size_per_gpu
        )
        # The number of steps (over micro batches) to accumulate gradients before taking an optimizer step.
        accumulation_steps = micro_batches_per_mini_batch

        all_metrics = defaultdict(list)
        critic_update_steps = 0
        for epoch in range(self.cfg.trainer.update_epochs_per_batch):
            pbar = tqdm(
                dataloader,
                desc=f"Critic Train epoch [{epoch + 1}/{self.cfg.trainer.update_epochs_per_batch}]",
                disable=not self.strategy.is_rank_0(),
            )
            for local_step, experience in enumerate(pbar):
                status = self.training_step(experience, global_step, local_step, accumulation_steps)
                critic_update_steps += 1

                # for DP
                # TODO (sumanthrh): this assumes all workers are data parallel.
                # We should get more accurate metrics with seq parallel or TP.
                # There are metrics like entropy where we get average over local data size
                status = self.strategy.all_reduce(status)

                for k, v in status.items():
                    all_metrics[k].append(v)
                pbar.set_postfix(status)

        torch.distributed.barrier()

        status_mean = reduce_metrics(all_metrics)
        status_mean["critic_update_steps"] = critic_update_steps / accumulation_steps

        output = TrainingOutputBatch()
        output.metadata = {"train_status": status_mean}
        return output

    def training_step(self, experience: Experience, global_step, local_step, accumulation_steps) -> Dict[str, float]:
        """
        Perform one micro-batch of training, accumulate gradients, and step the optimizer only after `accumulation_steps` micro-batches.
        """
        experience.to_device(torch.cuda.current_device())

        sequences = experience.sequences
        old_values = experience.values
        returns = experience.returns
        num_actions = experience.num_actions
        attention_mask = experience.attention_mask
        loss_mask = experience.loss_mask

        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            # critic loss
            values, output = self.model(
                sequences,
                num_actions=num_actions,
                attention_mask=attention_mask,
                return_output=True,
            )
            # loss function
            loss, clipfrac = self.critic_loss_fn(
                values,
                old_values,
                returns,
                config=self.cfg.trainer.algorithm,
                loss_mask=loss_mask,
            )
        loss = loss / accumulation_steps
        self.strategy.backward(loss, self.model, self.optimizer)
        grad_norm = None
        if (local_step + 1) % accumulation_steps == 0:
            grad_norm = self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler, name="critic")
            if grad_norm is not None:
                grad_norm = grad_norm.detach().cpu().item()

        # status
        status = {
            "critic_loss": loss.item(),
            "values_mean": masked_mean(values, loss_mask).item(),
            "critic_lr": self.scheduler.get_last_lr()[0],
            "values_clipfrac": clipfrac,
        }
        if grad_norm is not None:
            status["raw_grad_norm"] = grad_norm
        return status

    def save_checkpoint(self, ckpt_dir: str, tokenizer=None):
        self.strategy.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            ckpt_dir=ckpt_dir,
            node_local_rank=self.get_node_local_rank(),
            tokenizer=tokenizer,
        )

    def load_checkpoint(self, ckpt_dir=None, load_optimizer_states=True, load_lr_scheduler_states=True):
        _, states = self.strategy.load_checkpoint(
            model=self.model,
            optimizer=self.optimizer if load_optimizer_states else None,
            scheduler=self.scheduler if load_lr_scheduler_states else None,
            ckpt_dir=ckpt_dir,
            load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_lr_scheduler_states,
        )
        return states


class RefWorkerBase(Worker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model: nn.Module = None

    def _forward_micro_batch(self, micro_batch: TrainingInputBatch) -> TrainingOutputBatch:
        device = torch.cuda.current_device()
        micro_batch.to(device)
        sequences = micro_batch["sequences"]
        response_length = micro_batch.metadata["response_length"]
        attention_mask = micro_batch["attention_mask"]
        # MoE router-replay (R3): ref model also replays captured routing (it is
        # constructed with moe_router_replay=true), so its KL-reference logprobs
        # are computed on the same forward path as the policy. Absent key -> None
        # -> stock native forward (8B / flag-off unchanged).
        rollout_routed_experts = (
            micro_batch["rollout_routed_experts"]
            if "rollout_routed_experts" in micro_batch.keys()
            else None
        )
        with torch.no_grad(), torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            log_probs = self.model(
                sequences,
                response_length,
                attention_mask,
                return_output=False,
                rollout_routed_experts=rollout_routed_experts,
            )
        log_probs = log_probs.to("cpu")
        output = TrainingOutputBatch(
            {"output": log_probs},
        )
        output.metadata = micro_batch.metadata
        return output
