from pathlib import Path

import pytest
import ray
import torch
from megatron.core import parallel_state as mpu
from ray.util.placement_group import placement_group
from transformers import AutoTokenizer

from skyrl_train.config.numerics import Numerics
from skyrl_train.distributed.dispatch import concatenate_outputs_after_mesh_dispatch
from skyrl_train.models.grug_moe import GrugMoeConfig, GrugMoeForCausalLM
from skyrl_train.training_batch import ENGINE_DP_RANKS_KEY, TrainingBatchIterator, TrainingInputBatch
from skyrl_train.utils import get_ray_pg_ready_with_timeout
from skyrl_train.utils.utils import prepare_runtime_environment, sync_registries, validate_cfg
from skyrl_train.workers.megatron.megatron_model_wrapper import MegatronPolicyMicroBatch
from skyrl_train.workers.megatron.megatron_worker import MegatronPolicyWorkerBase
from skyrl_train.workers.worker import PPORayActorGroup
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.utils import get_test_actor_config

TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"
POLICY_WORLD_SIZE = 2
# One sequence per micro-batch, from under half the 2,048-token sliding window to over twice it. Two consecutive
# micro-batches share a length: a freed tensor's storage can come back for the next micro-batch's same-shaped tensor.
BODY_LENGTHS = (900, 1500, 2200, 2200, 3000, 4300)
RESPONSE_LENGTH = 256
ENGINE_EXPERT_PARALLEL_SIZE = 2


class GradientProbeWorkerBase(MegatronPolicyWorkerBase):
    def forward_backward_gradients(self, train_data: TrainingInputBatch) -> dict[str, float]:
        """One mini-batch's forward-backward with no optimizer step. Saves this pipeline rank's main gradients under
        ``train_data.metadata["output_dir"]`` and returns the mini-batch's last micro-batch metrics."""
        self.model.train()
        for chunk in self.actor_module:
            chunk.zero_grad_buffer()
        device = torch.cuda.current_device()
        micro_buffer = []
        for experience in TrainingBatchIterator(train_data, self.cfg.trainer.micro_train_batch_size_per_gpu):
            experience.to_device(device)
            position_ids = experience.attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(experience.attention_mask == 0, 0)
            micro_buffer.append(
                MegatronPolicyMicroBatch(
                    sequences=experience.sequences,
                    attention_mask=experience.attention_mask,
                    position_ids=position_ids,
                    num_actions=experience.num_actions,
                    old_action_log_probs=experience.action_log_probs,
                    base_action_log_probs=experience.base_action_log_probs,
                    advantages=experience.advantages,
                    loss_mask=experience.loss_mask,
                    rollout_action_logprobs=experience.rollout_logprobs,
                    response_span_tags=experience.response_span_tags,
                    rollout_engine_dp_ranks=experience.rollout_engine_dp_ranks,
                )
            )
        metrics = self.model.forward_backward_mini_batch(
            micro_batches=micro_buffer,
            seq_len=micro_buffer[0].sequences.shape[1],
            micro_batch_size=micro_buffer[0].sequences.shape[0],
            temperature=self.scoring_temperature,
        )
        gradients = {
            name: param.main_grad.detach().clone().cpu()
            for chunk in self.actor_module
            for name, param in chunk.named_parameters()
            if param.requires_grad
        }
        output_dir = Path(train_data.metadata["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(gradients, output_dir / f"pp{mpu.get_pipeline_model_parallel_rank()}.pt")
        return {"losses": [micro["final_loss"] for micro in metrics], **metrics[-1]}


GradientProbeWorker = ray.remote(num_gpus=1)(GradientProbeWorkerBase)


def _write_checkpoint(path: Path) -> None:
    """A Snowball-shaped Grug model cut to four layers with narrow experts, random weights and non-trivial gates."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    config = GrugMoeConfig(
        vocab_size=len(tokenizer),
        num_hidden_layers=4,
        intermediate_size=128,
        max_position_embeddings=8192,
        initializer_range=0.02,
    )
    torch.manual_seed(17)
    model = GrugMoeForCausalLM(config)
    with torch.no_grad():
        for module in model.modules():
            if module.__class__.__name__ == "GrugMoeGatedNorm":
                module.down_proj.weight.normal_(std=0.2)
                module.up_proj.weight.normal_(std=0.2)
        for layer in model.model.layers:
            layer.self_attn.attn_gate.weight.normal_(std=0.2)
            layer.mlp.router.bias.copy_(torch.linspace(-0.3, 0.3, config.num_local_experts))
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)


def _config(model_path: str, *, recompute: bool):
    cfg = get_test_actor_config()
    cfg.trainer.policy.model.path = model_path
    cfg.trainer.critic.model.path = None
    cfg.trainer.strategy = "megatron"
    cfg.trainer.flash_attn = False
    cfg.trainer.bf16 = True
    cfg.trainer.gradient_checkpointing = recompute
    cfg.trainer.use_sample_packing = False
    cfg.trainer.train_batch_size = len(BODY_LENGTHS)
    cfg.trainer.policy_mini_batch_size = len(BODY_LENGTHS)
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.update_epochs_per_batch = 1
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_entropy_loss = False
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.policy_num_gpus_per_node = POLICY_WORLD_SIZE
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 2
    cfg.trainer.policy.megatron_config.context_parallel_size = 1
    cfg.trainer.policy.megatron_config.expert_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = 1
    cfg.generator.backend = "vllm"
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.generator.inference_engine_data_parallel_size = ENGINE_EXPERT_PARALLEL_SIZE
    cfg.generator.inference_engine_expert_parallel_size = ENGINE_EXPERT_PARALLEL_SIZE
    cfg.generator.num_inference_engines = 1
    cfg.generator.n_samples_per_prompt = 1
    cfg.generator.sampling_params.temperature = 1.0
    cfg.generator.sampling_params.top_p = 1.0
    cfg.generator.sampling_params.top_k = -1
    validate_cfg(cfg)
    assert cfg.trainer.algorithm.resolved_numerics == Numerics.EXACT, cfg.trainer.algorithm.numerics_fallback_reasons
    return cfg


def _batch(pad_token_id: int, vocab_size: int) -> TrainingInputBatch:
    """One left-padded row per body length, served alternately by the two engine ranks."""
    generator = torch.Generator().manual_seed(5)
    total_length = max(BODY_LENGTHS)
    rows, masks = [], []
    for body_length in BODY_LENGTHS:
        body = torch.randint(10, vocab_size, (body_length,), generator=generator).tolist()
        before = total_length - body_length
        rows.append([pad_token_id] * before + body)
        masks.append([0] * before + [1] * body_length)
    sequences = torch.tensor(rows, dtype=torch.long)
    attention_mask = torch.tensor(masks, dtype=torch.long)
    response_mask = attention_mask[:, -RESPONSE_LENGTH:]
    zeros = torch.zeros(len(BODY_LENGTHS), RESPONSE_LENGTH, dtype=torch.float32)
    advantages = torch.linspace(-1.0, 1.0, RESPONSE_LENGTH).unsqueeze(0).repeat(len(BODY_LENGTHS), 1)
    batch = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "action_log_probs": zeros.clone(),
            "base_action_log_probs": zeros.clone(),
            "rollout_logprobs": zeros.clone(),
            "values": zeros.clone(),
            "returns": zeros.clone(),
            "advantages": advantages,
            "loss_mask": response_mask.clone(),
            "response_mask": response_mask.clone(),
            ENGINE_DP_RANKS_KEY: torch.arange(len(BODY_LENGTHS)) % ENGINE_EXPERT_PARALLEL_SIZE,
        }
    )
    batch.metadata = {"response_length": RESPONSE_LENGTH, "global_step": 0}
    return batch


def _init_ray(cfg) -> None:
    env_vars = prepare_runtime_environment(cfg)
    # Transformer Engine then picks deterministic attention backward algorithms, so two runs can agree bit for bit.
    env_vars["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
    ray.init(
        runtime_env={
            "env_vars": env_vars,
            "worker_process_setup_hook": "skyrl_train.worker_setup.configure_worker_process",
        }
    )
    sync_registries()


def _init_policy(cfg, model_path: str) -> PPORayActorGroup:
    pg = placement_group([{"GPU": POLICY_WORLD_SIZE, "CPU": POLICY_WORLD_SIZE}], strategy="PACK")
    get_ray_pg_ready_with_timeout(pg, timeout=30)
    policy = PPORayActorGroup(
        cfg,
        num_nodes=1,
        num_gpus_per_node=POLICY_WORLD_SIZE,
        ray_actor_type=GradientProbeWorker,
        pg=pg,
        num_gpus_per_actor=0.75,
        colocate_all=False,
        sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
        record_memory=cfg.trainer.policy.record_memory,
    )
    ray.get(policy.async_init_model(model_path))
    return policy


def _eval_logprobs(policy: PPORayActorGroup, batch: TrainingInputBatch) -> torch.Tensor:
    outputs = ray.get(policy.async_run_ray_method("mesh", "forward", data=batch))
    return concatenate_outputs_after_mesh_dispatch(policy.actor_infos, outputs)["output"].float()


def _gradients(policy: PPORayActorGroup, batch: TrainingInputBatch, output_dir: Path) -> dict[str, float]:
    """Run the mini-batch through ``policy`` and return the metrics; the gradients land in ``output_dir``."""
    batch.metadata = {**batch.metadata, "output_dir": str(output_dir)}
    return ray.get(policy.async_run_ray_method("mesh", "forward_backward_gradients", batch))[0]


def _load_gradients(output_dir: Path) -> dict[str, torch.Tensor]:
    gradients = {}
    for rank in range(POLICY_WORLD_SIZE):
        for name, grad in torch.load(output_dir / f"pp{rank}.pt", map_location="cpu", mmap=True).items():
            gradients[f"pp{rank}/{name}"] = grad
    return gradients


def _distances(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> dict[str, tuple[float, float]]:
    """Per tensor, the L2 norm and the largest absolute entry of ``left - right``."""
    distances = {}
    for name in left:
        difference = (left[name].double() - right[name].double()).flatten()
        distances[name] = (difference.norm().item(), difference.abs().max().item())
    return distances


def _pooled(distances: dict[str, tuple[float, float]], prefix: str = "") -> tuple[float, float]:
    """The L2 norm over every tensor whose name starts with ``prefix`` and their largest entry."""
    selected = [value for name, value in distances.items() if name.startswith(prefix)]
    return sum(norm**2 for norm, _ in selected) ** 0.5, max(largest for _, largest in selected)


def test_exact_gradients_agree_with_and_without_recompute(tmp_path):
    """Under exact numerics, full activation recompute in one-layer units must give the gradients of the same
    forward-backward without recompute, with six micro-batches in flight on two pipeline stages.

    The recompute rebuilds each layer's graph from values its first forward kept per micro-batch (input-norm
    statistics and FA3 outputs), so a recompute that read another micro-batch's values, or recomputed a value instead
    of taking the kept one, changes the gradient. The forward values are the same in every arm: the training
    log-probabilities equal the eval forward's bit for bit. Two runs of the recompute arm bound the backward's
    run-to-run noise. A repeatable backward must agree with the no-recompute arm bit for bit. Otherwise the noise is
    pooled over every gradient, since one tensor's repeat can agree by chance while another run's does not (28 and
    1,074 of 2,104 tensors differed between repeats in two runs with the attention backward's non-deterministic
    algorithms allowed), and each stage's gradients must lie within twice the pooled L2 distance and twice the largest
    entry; handing a recompute another in-flight micro-batch's kept values moved the first stage by 427 times the
    pooled L2 distance.
    """
    require_hoppers(2 * POLICY_WORLD_SIZE)
    model_path = tmp_path / "model"
    model_path.mkdir()
    _write_checkpoint(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    batch = _batch(tokenizer.pad_token_id, vocab_size=1000)
    with_recompute = _config(str(model_path), recompute=True)
    without_recompute = _config(str(model_path), recompute=False)
    _init_ray(with_recompute)
    try:
        recompute_policy = _init_policy(with_recompute, str(model_path))
        plain_policy = _init_policy(without_recompute, str(model_path))
        eval_logprobs = _eval_logprobs(recompute_policy, batch)
        batch["action_log_probs"] = (eval_logprobs * batch["response_mask"]).float()
        metrics = {
            "a": _gradients(recompute_policy, batch, tmp_path / "a"),
            "a_repeat": _gradients(recompute_policy, batch, tmp_path / "a_repeat"),
            "b": _gradients(plain_policy, batch, tmp_path / "b"),
        }
    finally:
        ray.shutdown()

    for arm, arm_metrics in metrics.items():
        print(f"{arm}: losses {arm_metrics['losses']}, log-ratio max abs {arm_metrics['log_ratio_abs_max']}")
        # The training forward reproduces the eval forward's log-probabilities bit for bit.
        assert arm_metrics["log_ratio_abs_max"] == 0.0, (arm, arm_metrics["log_ratio_abs_max"])
    assert metrics["a"]["losses"] == metrics["a_repeat"]["losses"] == metrics["b"]["losses"], metrics

    a, a_repeat, b = (_load_gradients(tmp_path / arm) for arm in ("a", "a_repeat", "b"))
    assert a.keys() == a_repeat.keys() == b.keys()
    noise, deviation = _distances(a, a_repeat), _distances(a, b)
    for name in a:
        (noise_norm, noise_largest), (norm, largest) = noise[name], deviation[name]
        print(f"{name}: |a-a'| L2 {noise_norm:.3e} max {noise_largest:.3e}; |a-b| L2 {norm:.3e} max {largest:.3e}")
    repeatable = all(torch.equal(a[name], a_repeat[name]) for name in a)
    print(f"{len(a)} gradients; the recompute arm is {'' if repeatable else 'not '}repeatable bit for bit")
    if repeatable:
        unequal = [name for name in a if not torch.equal(a[name], b[name])]
        assert not unequal, unequal
        return
    noise_norm, noise_largest = _pooled(noise)
    for rank in range(POLICY_WORLD_SIZE):
        norm, largest = _pooled(deviation, f"pp{rank}/")
        print(f"pp{rank}: |a-b| L2 {norm:.3e} vs pooled noise {noise_norm:.3e}; max {largest:.3e} vs {noise_largest:.3e}")
        assert norm <= 2 * noise_norm and largest <= 2 * noise_largest, (rank, norm, largest, noise_norm, noise_largest)
