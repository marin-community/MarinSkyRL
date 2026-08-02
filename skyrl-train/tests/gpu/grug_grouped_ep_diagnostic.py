"""Native Grug grouped-MM and FSDP2 expert-parallel parity on four H100s.

This is a torchrun executable, not a pytest test. It proves two contracts:

* eager and native grouped EP1 agree in BF16 for expert output and representative
  input, router, and expert gradients;
* one logical batch gives the same output, loss, expert/router/dense gradients,
  AdamW result, and FP32 query-bias update under ``(fsdp=4, ep=1)`` and
  ``(fsdp=2, ep=2)``.

Run on one whole Hopper node::

    torchrun --standalone --nproc_per_node=4 tests/gpu/grug_grouped_ep_diagnostic.py
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

from skyrl_train.distributed.fsdp_utils import (
    apply_ep,
    apply_fsdp2,
    create_device_mesh,
    fsdp2_load_full_state_dict,
    gather_dtensor_strided_safe,
)
from skyrl_train.models.grug_moe import (
    GrugMoeConfig,
    GrugMoeForCausalLM,
    GrugMoeSparseMoeBlock,
    enable_grug_grouped_mm,
)
from skyrl_train.models.grug_query_bias import (
    GrugQueryBiasAccumulator,
    next_query_bias,
    query_bias_candidate_count,
)
from skyrl_train.workers.worker import _grug_query_bias_virtual_shard_mask


SEED = 31415
EXPERT_NAME = "model.layers.0.mlp.experts.gate_proj.weight"
ROUTER_NAME = "model.layers.0.mlp.router.weight"
DENSE_NAME = "model.layers.0.shared_expert.gate_proj.weight"
BIAS_NAME = "model.layers.0.mlp.router.bias"
REPRESENTATIVE_NAMES = (EXPERT_NAME, ROUTER_NAME, DENSE_NAME)


@dataclass(frozen=True)
class _TopologyResult:
    logits: torch.Tensor
    loss: torch.Tensor
    gradients: dict[str, torch.Tensor]
    weights: dict[str, torch.Tensor]
    query_bias: torch.Tensor


def _config() -> GrugMoeConfig:
    return GrugMoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=64,
        shared_expert_intermediate_size=64,
        num_local_experts=8,
        num_experts_per_tok=2,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=32,
        sliding_window=8,
        initializer_range=0.02,
        qk_mult=1.37,
    )


def _same_tokens(device: torch.device) -> torch.Tensor:
    tokens = (torch.arange(96, device=device).reshape(8, 12) * 17 + 5) % 128
    return tokens.long()


def _assert_close(label: str, actual: torch.Tensor, expected: torch.Tensor, *, rtol: float, atol: float) -> None:
    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as error:
        max_abs = (actual.float() - expected.float()).abs().max().item()
        raise AssertionError(f"{label}: max_abs={max_abs:.6e}") from error


def _gate_ep1_eager_grouped(device: torch.device) -> None:
    torch.manual_seed(SEED)
    eager = GrugMoeSparseMoeBlock(_config()).to(device=device, dtype=torch.bfloat16)
    # A bare sparse block does not run GrugMoePreTrainedModel.post_init(), so
    # initialize its stacked expert tensors explicitly before comparing paths.
    with torch.no_grad():
        for parameter in eager.parameters():
            parameter.normal_(mean=0.0, std=_config().initializer_range)
    grouped = GrugMoeSparseMoeBlock(_config()).to(device=device, dtype=torch.bfloat16)
    grouped.load_state_dict(eager.state_dict())
    grouped.enable_grouped_mm()

    torch.manual_seed(SEED + 1)
    eager_input = torch.randn(2, 12, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    grouped_input = eager_input.detach().clone().requires_grad_(True)
    target = torch.randn_like(eager_input)

    eager_output = eager(eager_input)
    grouped_output = grouped(grouped_input)
    eager_loss = (eager_output.float() * target.float()).mean()
    grouped_loss = (grouped_output.float() * target.float()).mean()
    eager_loss.backward()
    grouped_loss.backward()

    _assert_close("EP1 eager/grouped output", grouped_output, eager_output, rtol=4e-2, atol=4e-3)
    _assert_close("EP1 eager/grouped input grad", grouped_input.grad, eager_input.grad, rtol=7e-2, atol=7e-5)
    _assert_close(
        "EP1 eager/grouped router grad",
        grouped.router.weight.grad,
        eager.router.weight.grad,
        rtol=7e-2,
        atol=7e-5,
    )
    for projection in ("gate_proj", "up_proj", "down_proj"):
        _assert_close(
            f"EP1 eager/grouped {projection} grad",
            getattr(grouped.experts, projection).weight.grad,
            getattr(eager.experts, projection).weight.grad,
            rtol=8e-2,
            atol=7e-5,
        )


def _materialize(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach()
    if isinstance(tensor, DTensor):
        if len(tensor.placements) > 1:
            tensor = gather_dtensor_strided_safe(tensor)
        else:
            tensor = tensor.full_tensor()
    return tensor.float().cpu().contiguous()


def _run_topology(*, fsdp_size: int, ep_size: int, device: torch.device) -> _TopologyResult | None:
    torch.manual_seed(SEED + 2)
    model = GrugMoeForCausalLM(_config()).to(device=device, dtype=torch.bfloat16)
    assert enable_grug_grouped_mm(model) == model.config.num_hidden_layers
    full_state = (
        {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if dist.get_rank() == 0
        else {}
    )

    mesh = create_device_mesh(world_size=dist.get_world_size(), fsdp_size=fsdp_size, ep_size=ep_size)
    fsdp_mesh = mesh["fsdp"]
    mixed_precision = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        cast_forward_inputs=True,
    )
    fsdp_kwargs = {
        "mesh": fsdp_mesh,
        "mp_policy": mixed_precision,
        "offload_policy": None,
        "reshard_after_forward": True,
    }
    if ep_size > 1:
        num_sharded = apply_ep(model, mesh, ep_comm_backend="torch", fsdp_kwargs=fsdp_kwargs)
        assert num_sharded == model.config.num_hidden_layers
    apply_fsdp2(model, fsdp_kwargs, {"cpu_offload": False})
    if ep_size > 1:
        fsdp2_load_full_state_dict(
            model,
            full_state,
            cpu_offload=False,
            ep_enabled=True,
        )
    del full_state

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=1e-2)

    # Emulate MeshDispatch: FSDP ranks get distinct pieces of one global batch,
    # while ranks differing only in EP get the same piece.
    global_tokens = _same_tokens(device)
    data_parallel_size = dist.get_world_size() // ep_size
    data_rank = dist.get_rank() // ep_size
    tokens = global_tokens.chunk(data_parallel_size, dim=0)[data_rank]

    # Observe the actual asymmetric training tokens. An EP rank owns one virtual
    # contiguous shard of the replicated optimizer window, matching EP=1's
    # logical rank geometry without changing the loss forward.
    ep_rank = mesh["ep"].get_local_rank() if ep_size > 1 else 0
    token_mask = _grug_query_bias_virtual_shard_mask(
        torch.ones_like(tokens, dtype=torch.bool),
        local_step=0,
        micro_batch_size=tokens.shape[0],
        accumulation_steps=1,
        ep_size=ep_size,
        ep_rank=ep_rank,
    )
    candidate_count = query_bias_candidate_count(
        int(token_mask.sum().item()),
        model.config.num_experts_per_tok,
        model.config.num_local_experts,
    )
    assert candidate_count > 1
    model.begin_query_bias_capture(candidate_count=candidate_count, token_mask=token_mask)

    output = model(tokens, labels=tokens)
    observation = model.take_query_bias_observation(candidate_count=candidate_count)
    accumulator = GrugQueryBiasAccumulator(
        candidate_count=candidate_count,
        num_layers=model.config.num_hidden_layers,
        num_experts=model.config.num_local_experts,
    )
    accumulator.observe(observation)
    gathered_logits = [torch.empty_like(output.logits) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_logits, output.logits.detach())
    global_loss = output.loss.detach().clone()
    dist.all_reduce(global_loss)
    global_loss.div_(dist.get_world_size())
    output.loss.backward()

    parameters = dict(model.named_parameters())
    gradients = {}
    for name in REPRESENTATIVE_NAMES:
        gradient = parameters[name].grad
        assert gradient is not None, f"{name} gradient is missing for fsdp={fsdp_size}, ep={ep_size}"
        gradients[name] = _materialize(gradient)

    optimizer.step()
    model.set_query_bias(next_query_bias(accumulator.finalize_betas()))
    state = model.state_dict()
    weights = {name: _materialize(state[name]) for name in REPRESENTATIVE_NAMES}
    query_bias = _materialize(state[BIAS_NAME])

    result = None
    if dist.get_rank() == 0:
        # One representative from each EP replica pair reconstructs the logical
        # batch in data-rank order.
        if ep_size > 1:
            for start in range(0, dist.get_world_size(), ep_size):
                for replica in range(start + 1, start + ep_size):
                    _assert_close(
                        "EP-replicated logits",
                        gathered_logits[replica],
                        gathered_logits[start],
                        rtol=0,
                        atol=0,
                    )
        logical_logits = torch.cat(gathered_logits[::ep_size], dim=0)
        result = _TopologyResult(
            logits=logical_logits.float().cpu(),
            loss=global_loss.float().cpu(),
            gradients=gradients,
            weights=weights,
            query_bias=query_bias,
        )

    del state, parameters, optimizer, output, model
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    return result


def _assert_topology_parity(ep1: _TopologyResult, ep2: _TopologyResult) -> None:
    _assert_close("topology logits", ep2.logits, ep1.logits, rtol=4e-2, atol=4e-3)
    _assert_close("topology loss", ep2.loss, ep1.loss, rtol=2e-3, atol=2e-3)
    for name in REPRESENTATIVE_NAMES:
        _assert_close(
            f"topology gradient {name}",
            ep2.gradients[name],
            ep1.gradients[name],
            rtol=8e-2,
            atol=1e-4,
        )
        _assert_close(
            f"topology post-step weight {name}",
            ep2.weights[name],
            ep1.weights[name],
            # AdamW updates BF16 parameters in place. Different grouped-reduction
            # orders can move a few values by one update-scale rounding step.
            rtol=4e-3,
            atol=4e-3,
        )
    assert ep1.query_bias.dtype == torch.float32
    assert ep2.query_bias.dtype == torch.float32
    torch.testing.assert_close(ep2.query_bias, ep1.query_bias, rtol=0, atol=0)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("native Grug grouped-MM validation requires CUDA")
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    assert dist.get_world_size() == 4, "run with torchrun --nproc_per_node=4"
    assert torch.cuda.get_device_properties(device).major >= 9, "native grouped-MM validation requires Hopper"

    _gate_ep1_eager_grouped(device)
    dist.barrier()
    if dist.get_rank() == 0:
        print("[Grug H100] eager vs native grouped EP1 forward/backward: PASS", flush=True)

    ep1 = _run_topology(fsdp_size=4, ep_size=1, device=device)
    ep2 = _run_topology(fsdp_size=2, ep_size=2, device=device)
    if dist.get_rank() == 0:
        assert ep1 is not None and ep2 is not None
        _assert_topology_parity(ep1, ep2)
        print("[Grug H100] (fsdp=4,ep=1) vs (fsdp=2,ep=2) AdamW parity: PASS", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
