"""CPU policy worker, strategy, and inference engine for running the production trainer locally.

These classes stand in for the Megatron workers and vLLM engines so the real entrypoint, trainer,
trajectory runners, and weight-sync protocol can run end to end on a tiny Hugging Face model.
"""

import asyncio
import os
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from skyrl_train.distributed.dispatch import MeshRank
from skyrl_train.distributed.strategy import DistributedStrategy
from skyrl_train.inference_engines.base import (
    InferenceEngineInput,
    InferenceEngineInterface,
    InferenceEngineOutput,
    NamedWeightsUpdateRequest,
)
from skyrl_train.io import io
from skyrl_train.utils.torch_utils import chunked_entropy_from_logits, logprobs_from_logits
from skyrl_train.workers.worker import PolicyWorkerBase

ABORT_STOP_REASON = "abort"
CHECKPOINT_FILE_TEMPLATE = "rank_{rank}.pt"
# Sampling controls the CPU engine implements; any other non-neutral control fails fast.
NEUTRAL_SAMPLING_PARAMS = {"top_p": 1.0, "top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0}


class CausalLMPolicy(nn.Module):
    """Adapt a Hugging Face causal LM to the policy worker's log-probability forward contract."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        sequences: torch.Tensor,
        num_actions: int,
        attention_mask: torch.Tensor,
        temperature: float = 1.0,
        return_output: bool = False,
        compute_entropy: bool = False,
        entropy_requires_grad: bool = True,
        rollout_routed_experts: torch.Tensor | None = None,
    ):
        if rollout_routed_experts is not None:
            raise ValueError("router replay requires a mixture-of-experts policy")
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        # Callers read only the trailing response-aligned positions, so skip the vocabulary
        # projection for the prompt; it dominates the cost of a tiny model with a full vocabulary.
        kept = num_actions + 1
        logits = self.model(
            sequences, attention_mask=attention_mask, position_ids=position_ids, logits_to_keep=kept
        ).logits
        logits = logits.float() / temperature
        labels = torch.roll(sequences, shifts=-1, dims=1)[:, -kept:]
        action_log_probs = logprobs_from_logits(logits, labels)[:, :-1]
        if not return_output:
            return action_log_probs
        output = {"logits": logits}
        if compute_entropy:
            output["entropy"] = chunked_entropy_from_logits(
                logits, requires_grad=entropy_requires_grad, attention_mask=attention_mask[:, -kept:]
            )
        return action_log_probs, output


class CPUStrategy(DistributedStrategy):
    """Plain PyTorch optimization over a gloo process group, averaging gradients across ranks."""

    def __init__(self, max_grad_norm: float):
        self.max_grad_norm = max_grad_norm
        self.world_size = dist.get_world_size()

    def setup_distributed(self):
        """The worker creates the gloo group before the strategy exists."""

    def collective_device(self) -> torch.device:
        return torch.device("cpu")

    def backward(self, loss: torch.Tensor, model, optimizer, **kwargs):
        loss.backward()

    def optimizer_step(self, optimizer, model, scheduler, name="model", stale_clip_lr_scale=1.0, **kwargs):
        if stale_clip_lr_scale != 1.0:
            raise ValueError("the CPU strategy does not implement StaleClip learning-rate scaling")
        if self.world_size > 1:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    dist.all_reduce(parameter.grad)
                    parameter.grad /= self.world_size
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        return grad_norm

    def save_checkpoint(
        self, model, ckpt_dir, node_local_rank, optimizer, scheduler, client_state=None, tokenizer=None
    ):
        io.makedirs(ckpt_dir, exist_ok=True)
        states = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "client_state": client_state or {},
            "rng": self.get_rng_state(),
        }
        torch.save(states, os.path.join(ckpt_dir, CHECKPOINT_FILE_TEMPLATE.format(rank=dist.get_rank())))
        dist.barrier()
        return None

    def load_checkpoint(
        self, model, ckpt_dir, optimizer=None, scheduler=None, load_module_strict=True, load_training_state=True
    ):
        path = os.path.join(ckpt_dir, CHECKPOINT_FILE_TEMPLATE.format(rank=dist.get_rank()))
        states = torch.load(path, weights_only=False)
        model.load_state_dict(states["model"], strict=load_module_strict)
        if load_training_state:
            optimizer.load_state_dict(states["optimizer"])
            scheduler.load_state_dict(states["scheduler"])
            self.load_rng_state(states["rng"])
        return ckpt_dir, states

    def save_hf_model(self, model, output_dir: str, tokenizer=None, **kwargs):
        if self.is_rank_0():
            model.model.save_pretrained(output_dir)
            if tokenizer is not None:
                tokenizer.save_pretrained(output_dir)
        dist.barrier()


class CPUPolicyWorker(PolicyWorkerBase):
    """Policy worker that trains a Hugging Face model on CPU and ships weights through Ray."""

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def init_worker_process_group(self):
        dist.init_process_group("gloo", rank=self._rank, world_size=self._world_size)
        self.mesh_rank = MeshRank(
            dp=self._rank,
            sp=0,
            tp=0,
            pp=0,
            world_size=self._world_size,
            dp_size=self._world_size,
            pp_size=1,
        )

    def init_model(self, model_path, num_training_steps: int | None = None):
        optimizer_config = self.cfg.trainer.policy.optimizer_config
        self.model = CausalLMPolicy(AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=optimizer_config.lr,
            betas=tuple(optimizer_config.adam_betas),
            weight_decay=optimizer_config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0, total_iters=0)
        self.strategy = CPUStrategy(max_grad_norm=optimizer_config.max_grad_norm)
        self._normalize_mini_batch_size()

    def _set_pad_token_id(self, pad_token_id):
        """The Hugging Face model reads its pad token from the saved config."""

    def offload_to_cpu(self, pin_memory=True, non_blocking=True, **kwargs):
        """State already lives on the CPU."""

    def backload_to_gpu(self, non_blocking=True, **kwargs):
        """State already lives on the CPU."""

    async def init_weight_sync_state(self, inference_engine_client):
        """Weights travel through the Ray object store, which needs no communicator."""

    async def broadcast_to_inference_engines(self, inference_engine_client):
        if dist.get_rank() == 0:
            state = {name: tensor.detach().clone() for name, tensor in self.model.model.state_dict().items()}
            await inference_engine_client.update_named_weights(
                {
                    "names": list(state),
                    "dtypes": [str(tensor.dtype) for tensor in state.values()],
                    "shapes": [list(tensor.shape) for tensor in state.values()],
                    "extras": [{"tensor": tensor} for tensor in state.values()],
                }
            )
        dist.barrier()


class CPUInferenceEngine(InferenceEngineInterface):
    """Sample from a Hugging Face causal LM with vLLM's pause, abort, and weight-update semantics.

    Pausing aborts in-flight requests at their next token, returning the tokens generated so far with
    stop reason ``abort``; requests that arrive while paused wait for the resume.
    """

    def __init__(self, model_path: str, seed: int):
        torch.manual_seed(seed)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.eos_token_id = self.model.config.eos_token_id
        self.max_model_len = self.model.config.max_position_embeddings
        self._paused = False
        self._resumed = asyncio.Event()
        self._resumed.set()

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        await self._resumed.wait()
        sampling_params = input_batch["sampling_params"]
        _check_sampling_params(sampling_params)
        results = [
            await self._generate_one(prompt_ids, sampling_params) for prompt_ids in input_batch["prompt_token_ids"]
        ]
        return InferenceEngineOutput(
            responses=[self.tokenizer.decode(ids, skip_special_tokens=True) for ids, _, _ in results],
            response_ids=[ids for ids, _, _ in results],
            stop_reasons=[stop_reason for _, _, stop_reason in results],
            response_logprobs=(
                [logprobs for _, logprobs, _ in results] if sampling_params.get("logprobs") is not None else None
            ),
            prompt_logprobs=None,
        )

    async def _generate_one(self, prompt_ids: list[int], sampling_params: dict[str, Any]):
        temperature = float(sampling_params["temperature"])
        min_tokens = int(sampling_params.get("min_tokens", 0))
        response_ids: list[int] = []
        response_logprobs: list[float] = []
        input_ids = torch.tensor([prompt_ids])
        past_key_values = None
        stop_reason = "length"
        for _ in range(int(sampling_params["max_tokens"])):
            if self._paused:
                stop_reason = ABORT_STOP_REASON
                break
            with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                output = self.model(input_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = output.past_key_values
            logits = output.logits[0, -1].float()
            if len(response_ids) < min_tokens:
                logits[self.eos_token_id] = float("-inf")
            if temperature == 0.0:
                log_probs = torch.log_softmax(logits, dim=-1)
                token_id = int(torch.argmax(logits))
            else:
                log_probs = torch.log_softmax(logits / temperature, dim=-1)
                token_id = int(torch.multinomial(log_probs.exp(), 1))
            response_ids.append(token_id)
            response_logprobs.append(float(log_probs[token_id]))
            if token_id == self.eos_token_id:
                stop_reason = "stop"
                break
            input_ids = torch.tensor([[token_id]])
            # Yield so a pause or weight update can interleave between tokens.
            await asyncio.sleep(0)
        return response_ids, response_logprobs, stop_reason

    async def pause_generation(self) -> None:
        self._paused = True
        self._resumed.clear()

    async def resume_generation(self) -> None:
        self._paused = False
        self._resumed.set()

    async def update_named_weights(self, request: NamedWeightsUpdateRequest):
        state = self.model.state_dict()
        with torch.no_grad():
            for name, extra in zip(request["names"], request["extras"], strict=True):
                state[name].copy_(extra["tensor"])

    async def chat_completion(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("the CPU engine serves token-in-token-out generation only")

    async def completion(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("the CPU engine serves token-in-token-out generation only")

    async def wake_up(self, *args: Any, **kwargs: Any):
        """The CPU engine never releases its memory."""

    async def sleep(self, *args: Any, **kwargs: Any):
        """The CPU engine never releases its memory."""

    async def init_weight_update_communicator(
        self, master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing: bool = False
    ):
        """Weights arrive through the Ray object store."""

    async def begin_weight_reload(self):
        """Weights load directly into the Hugging Face parameters."""

    async def finish_weight_reload(self):
        """Weights load directly into the Hugging Face parameters."""

    async def reset_prefix_cache(self):
        """The CPU engine keeps no prefix cache across requests."""

    async def teardown(self):
        """Ray releases the actor's resources."""

    def tp_size(self) -> int:
        return 1

    def pp_size(self) -> int:
        return 1

    def dp_size(self) -> int:
        return 1


def _check_sampling_params(sampling_params: dict[str, Any]) -> None:
    for key, neutral in NEUTRAL_SAMPLING_PARAMS.items():
        if sampling_params.get(key, neutral) != neutral:
            raise ValueError(f"the CPU engine does not implement {key}={sampling_params[key]}")
    if sampling_params.get("stop"):
        raise ValueError("the CPU engine does not implement stop strings")
