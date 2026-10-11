"""Shared pieces for the DPO equivalence experiments (issue 969).

The experiments compare three stages of the production DPO update path against independent
references: the distributed backward pass (gradients across micro-batches, pipeline stages,
and experts, including the batch-major output projection), the optimizer update (clipping,
Adam moments, step counters, parameter deltas), and the transfer of trained tensors into
vLLM. The independent backward reference is eager HF GrugMoE in float32 driven by a literal
TRL-style DPO formula, computed outside the skyrl_train objective stack.
"""

from dataclasses import dataclass
from importlib import util
import hashlib

import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import nn
from transformers import AutoConfig

from skyrl_train.config.utils import get_default_config
from skyrl_train.distributed.megatron.megatron_utils import materialize_megatron_params
from skyrl_train.models.grug_moe import GRUG_ROUTER_BIAS_SUFFIX
from skyrl_train.models.grug_moe import GrugMoeForCausalLM
from skyrl_train.workers.megatron.output_projection_layout import install_dpo_batch_major_output_projection
from tests.cpu.util import DPO_OVERRIDES
from tests.gpu.grug_tiny_checkpoint import write_tiny_checkpoint

BETA = 0.1
LABEL_SMOOTHING = 0.0
# Router weights are zeroed in the equivalence checkpoint so biased expert selection is
# decided by the constant router bias alone: the selection carries no data-dependent term,
# so no amount of cross-implementation hidden-state drift can flip a route (scaling the
# weights instead would scale the score noise as much as the margins). Routing mechanics
# are still exercised end to end — per-token top-k dispatch, stacked experts, combine
# weights, expert-parallel partitions — and the router weight still receives gradients
# through the combine-weight softmax. ROUTE_MARGIN_MIN is the smallest constant bias gap.
ROUTE_DECISIVE_BATCH_SEED = 27
ROUTE_MARGIN_MIN = 5e-3


@dataclass(frozen=True)
class PairBatch:
    """Deterministic preference rows: adjacent chosen (+1) / rejected (-1) pairs."""

    sequences: torch.Tensor
    attention_mask: torch.Tensor
    loss_mask: torch.Tensor
    pair_roles: torch.Tensor
    num_actions: int

    @property
    def num_pairs(self) -> int:
        return self.sequences.shape[0] // 2

    def dp_shard(self, dp_rank: int, dp_size: int, micro_batch_size: int) -> list[tuple[int, "PairBatch"]]:
        """Slice this batch into pair-aligned micro-batches for one DP rank, with row offsets."""
        rows = self.sequences.shape[0]
        if rows % (2 * dp_size):
            raise ValueError(f"{rows} rows do not split into whole pairs across {dp_size} DP ranks")
        rows_per_rank = rows // dp_size
        if rows_per_rank % micro_batch_size:
            raise ValueError(f"{rows_per_rank} rows per rank do not split into {micro_batch_size}-row micro-batches")
        start = dp_rank * rows_per_rank
        micros = []
        for offset in range(start, start + rows_per_rank, micro_batch_size):
            stop = offset + micro_batch_size
            micros.append(
                (
                    offset,
                    PairBatch(
                        sequences=self.sequences[offset:stop],
                        attention_mask=self.attention_mask[offset:stop],
                        loss_mask=self.loss_mask[offset:stop],
                        pair_roles=self.pair_roles[offset:stop],
                        num_actions=self.num_actions,
                    ),
                )
            )
        return micros

    def sha256(self) -> str:
        digest = hashlib.sha256()
        for tensor in (self.sequences, self.attention_mask, self.loss_mask, self.pair_roles):
            digest.update(tensor.contiguous().numpy().tobytes())
        return digest.hexdigest()


def fixed_pair_batch(
    pad_token_id: int,
    *,
    num_pairs: int = 4,
    prompt_length: int = 12,
    response_length: int = 8,
    seed: int = 5,
) -> PairBatch:
    """Build a fixed preference batch with left padding, varied bodies, and masked tails.

    Every pair shares a prompt; the chosen and rejected completions differ in content, and
    the rejected row of every second pair is shorter so its loss mask has an empty tail.
    """
    generator = torch.Generator().manual_seed(seed)
    rows = 2 * num_pairs
    left_pad = [3, 0, 5, 1]
    total_length = prompt_length + response_length + 5
    sequences, masks, action_masks = [], [], []
    for row in range(rows):
        body_length = prompt_length + response_length - (row % 2)
        body = torch.randint(10, 500, (body_length,), generator=generator).tolist()
        before = left_pad[row % len(left_pad)]
        after = total_length - body_length - before
        sequences.append([pad_token_id] * before + body + [pad_token_id] * after)
        masks.append([0] * before + [1] * body_length + [0] * after)
        # The trainer's action window is the final `response_length` columns; a slot counts
        # when a real response token sits there, so padded tails drop out of the loss.
        window = [0] * total_length
        body_start = before
        response_end = body_start + body_length
        for column in range(total_length - response_length, total_length):
            if column < response_end and masks[-1][column] == 1:
                window[column] = 1
        assert sum(window) >= 1, (row, body_length, before)
        action_masks.append(window)
    pair_roles = torch.tensor([1.0, -1.0], dtype=torch.float32).repeat(num_pairs)
    # The loss mask covers exactly the action window (the final `response_length` columns).
    loss_window = torch.tensor(action_masks, dtype=torch.float32)[:, -response_length:]
    return PairBatch(
        sequences=torch.tensor(sequences, dtype=torch.long),
        attention_mask=torch.tensor(masks, dtype=torch.long),
        loss_mask=loss_window,
        pair_roles=pair_roles,
        num_actions=response_length,
    )


@dataclass(frozen=True)
class DPOReference:
    """Independent DPO loss, metrics, gradients, and log-probabilities from eager HF GrugMoE."""

    loss: float
    metrics: dict[str, float]
    gradients: dict[str, torch.Tensor]
    action_log_probs: torch.Tensor
    ref_action_log_probs: torch.Tensor


def hf_dpo_reference(
    model_path: str,
    batch: PairBatch,
    *,
    beta: float = BETA,
    label_smoothing: float = LABEL_SMOOTHING,
    device: str = "cpu",
    ref_log_prob_offsets: torch.Tensor | None = None,
) -> DPOReference:
    """Compute the DPO loss, its gradients, and both log-prob streams with eager HF GrugMoE.

    The formula is written out from the TRL "sigmoid" definition rather than calling any
    skyrl_train objective code: pair log-ratios are masked token sums of policy minus frozen
    reference log-probabilities, and the loss averages ``-(1-eps) log sigmoid(beta*delta) -
    eps log sigmoid(-beta*delta)`` over pairs. ``ref_log_prob_offsets`` deterministically
    separates the frozen reference from the initial policy, as the production reference
    worker does; without it the fixture is degenerate (delta == 0, accuracy == 0).
    """
    model = GrugMoeForCausalLM.from_pretrained(model_path, dtype=torch.float32, attn_implementation="eager")
    model.to(device).train()
    sequences = batch.sequences.to(device)
    attention_mask = batch.attention_mask.to(device)
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids = position_ids.masked_fill(attention_mask == 0, 0)

    def response_log_probs() -> torch.Tensor:
        logits = model(sequences, attention_mask=attention_mask, position_ids=position_ids).logits
        log_probs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        targets = sequences[:, 1:]
        token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return token_log_probs[:, -batch.num_actions - 1 : -1]

    action_log_probs = response_log_probs()
    with torch.no_grad():
        ref_action_log_probs = response_log_probs()
        if ref_log_prob_offsets is not None:
            ref_action_log_probs = ref_action_log_probs + ref_log_prob_offsets.to(device)
    mask = batch.loss_mask.to(device)
    sums = ((action_log_probs - ref_action_log_probs) * mask).sum(-1)
    pair_logratios = sums.reshape(-1, 2)
    chosen, rejected = pair_logratios[:, 0], pair_logratios[:, 1]
    delta = chosen - rejected
    loss = (
        -(1 - label_smoothing) * torch.nn.functional.logsigmoid(beta * delta)
        - label_smoothing * torch.nn.functional.logsigmoid(-beta * delta)
    ).mean()
    loss.backward()
    gradients = {
        name: parameter.grad.detach().float().cpu()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    metrics = {
        "dpo/loss": float(loss.detach()),
        "dpo/accuracy": float((delta > 0).float().mean()),
        "dpo/margin": float((beta * (chosen - rejected)).detach().mean()),
        "dpo/chosen_reward": float((beta * chosen).detach().mean()),
        "dpo/rejected_reward": float((beta * rejected).detach().mean()),
    }
    return DPOReference(
        loss=float(loss.detach()),
        metrics=metrics,
        gradients=gradients,
        action_log_probs=action_log_probs.detach().float().cpu(),
        ref_action_log_probs=ref_action_log_probs.detach().float().cpu(),
    )


def dpo_wrapper_config(seq_len: int) -> DictConfig:
    """Full trainer config for MegatronModelWrapper with the shipped DPO recipe applied."""
    cfg = OmegaConf.merge(get_default_config(), OmegaConf.create(DPO_OVERRIDES))
    with open_dict(cfg):
        cfg.trainer.algorithm.max_seq_len = seq_len
    return cfg


def build_grug_megatron_chunks(
    model_path: str,
    *,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    expert_tensor_parallel_size: int = 1,
    bf16: bool = False,
    flash_attn: bool = False,
    attention_backend: str | None = None,
    wrap_with_ddp: bool = True,
    policy_loss_type: str | None = "dpo",
):
    """Build DDP-wrapped Megatron chunks through the same provider path as the training worker.

    Mirrors MegatronWorker's bridge/provider setup, then installs the shared batch-major
    output projection so experiments and the worker provably run the same adaptation.
    ``attention_backend`` defaults to the worker's choice; the fp32 equivalence experiment
    overrides it with "unfused" because Transformer Engine's fused kernels are fp16/bf16-only,
    while the production bf16 kernels stay covered by the parity gates.
    """
    from megatron.bridge import AutoBridge
    from megatron.core.distributed import DistributedDataParallelConfig

    import skyrl_train.models.grug_megatron_bridge  # noqa: F401  # registers the Grug bridge with Megatron-Bridge

    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    bridge = AutoBridge.from_hf_pretrained(model_path, trust_remote_code=True)
    provider = bridge.to_megatron_provider()
    provider.tensor_model_parallel_size = tensor_model_parallel_size
    provider.pipeline_model_parallel_size = pipeline_model_parallel_size
    provider.pipeline_dtype = torch.bfloat16 if bf16 else torch.float32
    provider.context_parallel_size = context_parallel_size
    provider.expert_model_parallel_size = expert_model_parallel_size
    provider.expert_tensor_parallel_size = expert_tensor_parallel_size
    provider.sequence_parallel = tensor_model_parallel_size > 1
    provider.attention_backend = attention_backend or ("flash" if flash_attn else "fused")
    provider.variable_seq_lengths = True
    provider.masked_softmax_fusion = True
    provider.moe_token_dispatcher_type = "alltoall"
    provider.gradient_accumulation_fusion = util.find_spec("fused_weight_gradient_mlp_cuda") is not None
    provider.finalize()

    ddp_config = DistributedDataParallelConfig()
    if wrap_with_ddp:
        ddp_config.use_distributed_optimizer = True
    chunks = provider.provide_distributed_model(ddp_config=ddp_config, wrap_with_ddp=wrap_with_ddp, bf16=bf16)
    install_dpo_batch_major_output_projection(chunks, provider, policy_loss_type)
    return bridge, provider, chunks, hf_config


def export_parameter_gradients(bridge, chunks: list[nn.Module]) -> dict[str, torch.Tensor]:
    """Export DDP-reduced gradients in HF layout by temporarily swapping param data for main_grad.

    The bridge conversion tasks define the named-tensor alignment across pipeline partitions
    and expert shards; reusing them for gradients guarantees the same alignment the weight
    sync uses. Returns rank 0's complete HF-named gradient export.
    """
    materialize_megatron_params(chunks)
    swapped = []
    for chunk in chunks:
        for _, parameter in chunk.named_parameters():
            main_grad = getattr(parameter, "main_grad", None)
            if main_grad is None:
                continue
            swapped.append((parameter, parameter.data))
            parameter.data = main_grad
    try:
        tasks = bridge.get_conversion_tasks(chunks)
        exports = {}
        for name, tensor in bridge.export_hf_weights(chunks, show_progress=False, conversion_tasks=tasks):
            exports[name] = tensor.detach().float().cpu().clone()
    finally:
        for parameter, data in swapped:
            parameter.data = data
    return exports


def dpo_gradient_names(num_layers: int) -> list[str]:
    """Router-bias buffers carry no gradient; everything else the export emits must."""
    return [f"model.layers.{index}{GRUG_ROUTER_BIAS_SUFFIX}" for index in range(num_layers)]


def write_route_decisive_checkpoint(path) -> None:
    """Write the tiny checkpoint with router weights zeroed for bias-decided selection."""
    from safetensors.torch import load_file, save_file

    write_tiny_checkpoint(path)
    tensors = load_file(path / "model.safetensors")
    for name, tensor in tensors.items():
        if name.endswith("mlp.router.weight"):
            tensors[name] = torch.zeros_like(tensor)
    save_file(tensors, path / "model.safetensors", metadata={"format": "pt"})


def dpo_backward_fixture(path):
    """Route-decisive checkpoint plus its preference batch, validated for stable routing."""
    from transformers import AutoTokenizer

    write_route_decisive_checkpoint(path)
    pad_token_id = AutoTokenizer.from_pretrained(path).pad_token_id
    batch = fixed_pair_batch(pad_token_id, seed=ROUTE_DECISIVE_BATCH_SEED)
    margin = min_routing_margin(str(path), batch)
    print(f"fixture minimum biased-router margin: {margin:.6g}", flush=True)
    if margin <= ROUTE_MARGIN_MIN:
        raise AssertionError(f"fixture route margin {margin:.6g} must exceed {ROUTE_MARGIN_MIN:g}")
    return path, batch


def min_routing_margin(model_path: str, batch: PairBatch, device: str = "cpu") -> float:
    """Smallest biased-router selection margin over unmasked tokens across all layers.

    The margin is the gap between the k-th selected and the (k+1)-th biased router score,
    matching the selection rule in GrugMoeRouter.forward; a flip of any top-k decision
    changes expert gradients discretely, so the fixture must keep every decision far from
    the tie boundary relative to cross-implementation score noise.
    """
    model = GrugMoeForCausalLM.from_pretrained(model_path, dtype=torch.float32, attn_implementation="eager")
    model.to(device).eval()
    margins: list[torch.Tensor] = []
    hooks = []

    def capture(module, args, _output):
        hidden = args[0].detach().float()
        scores = torch.nn.functional.linear(hidden, module.weight.detach().float())
        scores = scores + module.bias.detach().float()
        top = scores.topk(module.top_k + 1, dim=-1).values
        margins.append((top[:, -2] - top[:, -1]).reshape(batch.attention_mask.shape))

    for module in model.modules():
        if module.__class__.__name__ == "GrugMoeRouter":
            hooks.append(module.register_forward_hook(capture))
    attention_mask = batch.attention_mask.to(device)
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids = position_ids.masked_fill(attention_mask == 0, 0)
    with torch.no_grad():
        model(batch.sequences.to(device), attention_mask=attention_mask, position_ids=position_ids)
    for hook in hooks:
        hook.remove()
    valid = attention_mask.bool().flatten()
    return min(margin.flatten()[valid].min().item() for margin in margins)
