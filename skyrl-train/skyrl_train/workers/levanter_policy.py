"""Ray compatibility facade for an external Marin Levanter policy service."""

import asyncio
import io
from dataclasses import dataclass

import numpy as np
import ray
import requests
import torch

from skyrl_train.distributed.dispatch import ActorInfo, DispatchRegistry, DispatchSettings, MeshRank
from skyrl_train.training_batch import TrainingInputBatch, TrainingOutputBatch
from marinskyrl.runtime_options import R3Transport


def _encode_batch(data: TrainingInputBatch, *, training: bool) -> bytes:
    action_count = int(data.metadata["response_length"])
    arrays: dict[str, np.ndarray] = {
        "sequences": data["sequences"].detach().cpu().numpy(),
        "attention_mask": data["attention_mask"].detach().cpu().numpy(),
        "action_count": np.asarray(action_count, dtype=np.int32),
    }
    if training:
        arrays.update(
            old_action_log_probs=data["action_log_probs"].detach().cpu().numpy(),
            advantages=data["advantages"].detach().cpu().numpy(),
            loss_mask=data["loss_mask"].detach().cpu().numpy(),
        )
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def _decode_array(payload: bytes) -> np.ndarray:
    return np.load(io.BytesIO(payload), allow_pickle=False)


@dataclass(frozen=True)
class _TrainResponse:
    action_log_probs: np.ndarray
    loss: float
    step: int


def _decode_train_response(payload: bytes) -> _TrainResponse:
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        return _TrainResponse(arrays["action_log_probs"], float(arrays["loss"]), int(arrays["step"]))


@ray.remote(num_cpus=0)
class LevanterPolicyProxy:
    """One Ray-visible rank forwarding calls to one Levanter service process."""

    def __init__(self, endpoint: str, mesh_rank: MeshRank, timeout_seconds: float, generator_config) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.mesh_rank = mesh_rank
        self.timeout_seconds = timeout_seconds
        self._generator_config = generator_config

    def get_mesh_rank(self) -> MeshRank:
        return self.mesh_rank

    def init_model(self, *_args, **_kwargs) -> None:
        return None

    def forward(self, data: TrainingInputBatch) -> TrainingOutputBatch:
        response = requests.post(
            f"{self.endpoint}/forward",
            data=_encode_batch(data, training=False),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        output = TrainingOutputBatch({"output": torch.from_numpy(_decode_array(response.content))})
        output.metadata = data.metadata
        return output

    def ppo_train(self, data: TrainingInputBatch) -> TrainingOutputBatch:
        response = requests.post(
            f"{self.endpoint}/ppo_train",
            data=_encode_batch(data, training=True),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        result = _decode_train_response(response.content)
        output = TrainingOutputBatch()
        output.metadata = {
            "train_status": {
                "policy_loss": result.loss,
                "policy_update_steps": 1.0,
                "levanter_step": float(result.step),
            }
        }
        return output

    def broadcast_to_inference_engines(self, _inference_engine_client) -> int:
        if not self._is_source_rank():
            return 0
        response = requests.post(f"{self.endpoint}/broadcast_weights", timeout=self.timeout_seconds)
        response.raise_for_status()
        return int(response.json()["step"])

    def empty_cache(self) -> None:
        return None

    def barrier_all(self) -> None:
        return None

    def _is_source_rank(self) -> bool:
        return self.mesh_rank.dp == 0 and self.mesh_rank.sp == 0 and self.mesh_rank.tp == 0 and self.mesh_rank.pp == 0

    def init_weight_sync_state(self, inference_engine_client) -> None:
        if not self._is_source_rank():
            return
        if not inference_engine_client.enable_http_endpoint:
            raise ValueError("Levanter weight sync requires generator.enable_http_endpoint=true")
        if inference_engine_client.backend != "vllm":
            raise ValueError("Levanter weight sync currently requires the vLLM generator backend")

        rendezvous = requests.get(
            f"{self.endpoint}/weight_sync_address",
            timeout=self.timeout_seconds,
        )
        rendezvous.raise_for_status()
        master_addr = str(rendezvous.json()["master_addr"])
        master_port = int(rendezvous.json()["master_port"])
        world_size = (
            int(self._generator_config.num_inference_engines)
            * int(self._generator_config.inference_engine_tensor_parallel_size)
            * int(self._generator_config.inference_engine_pipeline_parallel_size)
            * int(self._generator_config.inference_engine_data_parallel_size)
            + 1
        )
        override_existing = self._generator_config.override_existing_update_group != "disable"
        payload = {
            "backend": self._generator_config.weight_sync_backend,
            "master_addr": master_addr,
            "master_port": master_port,
            "world_size": world_size,
            "group_name": "skyrl",
            "bridge_url": (
                f"http://{inference_engine_client.http_endpoint_advertise_host}:"
                f"{inference_engine_client.http_endpoint_port}"
            ),
        }

        async def initialize() -> None:
            communicator = inference_engine_client.init_weight_update_communicator(
                master_addr=master_addr,
                master_port=master_port,
                rank_offset=1,
                world_size=world_size,
                group_name="skyrl",
                backend=self._generator_config.weight_sync_backend,
                override_existing=override_existing,
            )
            configure = asyncio.to_thread(
                requests.post,
                f"{self.endpoint}/configure_weight_sync",
                json=payload,
                timeout=self.timeout_seconds,
            )
            _, response = await asyncio.gather(communicator, configure)
            response.raise_for_status()

        asyncio.run(initialize())

    def _set_pad_token_id(self, _pad_token_id: int) -> None:
        return None

    def offload_to_cpu(self, **_kwargs) -> None:
        return None

    def backload_to_gpu(self, **_kwargs) -> None:
        return None


class LevanterPolicyActorGroup:
    """Duck-typed ``PPORayActorGroup`` backed by Levanter HTTP processes."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        endpoints = tuple(cfg.trainer.policy.levanter.endpoint_urls)
        data_parallel_size = int(cfg.trainer.policy.levanter.data_parallel_size)
        timeout_seconds = float(cfg.trainer.policy.levanter.request_timeout_seconds)
        if not endpoints:
            raise ValueError("trainer.policy.levanter.endpoint_urls must not be empty")
        if len(endpoints) % data_parallel_size:
            raise ValueError("Levanter endpoint count must be divisible by data_parallel_size")
        model_parallel_size = len(endpoints) // data_parallel_size
        self._actor_handlers = []
        self.actor_infos = []
        for rank, endpoint in enumerate(endpoints):
            mesh_rank = MeshRank(
                dp=rank // model_parallel_size,
                sp=0,
                tp=rank % model_parallel_size,
                pp=0,
                world_size=len(endpoints),
                dp_size=data_parallel_size,
                pp_size=1,
            )
            actor = LevanterPolicyProxy.remote(endpoint, mesh_rank, timeout_seconds, cfg.generator)
            self._actor_handlers.append(actor)
            self.actor_infos.append(ActorInfo(actor, mesh_rank))

    def async_init_model(self, *args, **kwargs):
        return [actor.init_model.remote(*args, **kwargs) for actor in self._actor_handlers]

    def async_run_ray_method(self, dispatch_type: str, method_name: str, *args, **kwargs):
        dispatch_class = DispatchRegistry.get(dispatch_type)
        args, kwargs = dispatch_class.validate_dispatch_args(*args, **kwargs)
        settings = DispatchSettings(
            r3_transport=R3Transport(self.cfg.generator.r3_transport),
            r3_dispatch_put_timeout_seconds=float(self.cfg.generator.r3_dispatch_put_timeout_seconds),
        )
        return dispatch_class.dispatch(self.actor_infos, method_name, *args, settings=settings, **kwargs)

    def offload_to_cpu(self, nonblocking=False, **kwargs):
        refs = [actor.offload_to_cpu.remote(**kwargs) for actor in self._actor_handlers]
        return refs if nonblocking else ray.get(refs)

    def backload_to_gpu(self, nonblocking=False, **kwargs):
        refs = [actor.backload_to_gpu.remote(**kwargs) for actor in self._actor_handlers]
        return refs if nonblocking else ray.get(refs)

    def kill_actors(self, no_restart: bool = True) -> None:
        for actor in self._actor_handlers:
            ray.kill(actor, no_restart=no_restart)
