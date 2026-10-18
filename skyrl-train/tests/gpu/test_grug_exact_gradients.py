from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import ray
import torch
from megatron.core import parallel_state as mpu
from ray.util.placement_group import placement_group

from skyrl_train.config.numerics import NumericsMode, read_numerics
from skyrl_train.distributed.dispatch import concatenate_outputs_after_mesh_dispatch
from skyrl_train.distributed.megatron.megatron_utils import get_model_config
from skyrl_train.training_batch import TrainingBatchIterator, TrainingInputBatch
from skyrl_train.trajectory_runners.types import ROLLOUT_ENGINE_DP_RANKS_KEY
from skyrl_train.utils import get_ray_pg_ready_with_timeout
from skyrl_train.utils.utils import prepare_runtime_environment, sync_registries, validate_cfg
from skyrl_train.workers.megatron.megatron_model_wrapper import MegatronPolicyMicroBatch
from skyrl_train.workers.megatron.megatron_worker import MegatronPolicyWorkerBase
from skyrl_train.workers.worker import PPORayActorGroup
from tests.gpu.grug_exact_checkpoint import exact_numerics_config, write_snowball_checkpoint
from tests.gpu.grug_gpu_gates import require_hoppers

POLICY_WORLD_SIZE = 2
PAD_TOKEN_ID = 0
# One sequence per micro-batch, from under half the 2,048-token sliding window to over twice it.
BODY_LENGTHS = (900, 1500, 2200, 2200, 3000, 4300)
RESPONSE_LENGTH = 256
ENGINE_RANKS = 2
# With every token's experts fixed, exact and native gradients differ by their bf16 roundings alone; a gradient that is
# missing or computed from the wrong value differs by about its own size.
NATIVE_RELATIVE_TOLERANCE = 0.5


class GradientProbeWorkerBase(MegatronPolicyWorkerBase):
    def forward_backward_gradients(self, train_data: TrainingInputBatch) -> dict[str, float]:
        """One mini-batch's forward-backward with no optimizer step, with full activation recompute when
        ``train_data.metadata["recompute"]``. Saves this pipeline rank's main gradients under
        ``train_data.metadata["output_dir"]`` and returns the mini-batch's last micro-batch metrics."""
        get_model_config(self.actor_module[0]).recompute_granularity = (
            "full" if train_data.metadata["recompute"] else None
        )
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


def _config(model_path: str, mode: NumericsMode):
    cfg = exact_numerics_config(
        model_path, policy_gpus=POLICY_WORLD_SIZE, pipeline_stages=2, expert_parallel=1, engine_ranks=ENGINE_RANKS
    )
    cfg.trainer.numerics.mode = mode.value
    cfg.trainer.gradient_checkpointing = True
    cfg.trainer.train_batch_size = len(BODY_LENGTHS)
    cfg.trainer.policy_mini_batch_size = len(BODY_LENGTHS)
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.update_epochs_per_batch = 1
    validate_cfg(cfg)
    assert read_numerics(cfg).mode is mode
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
    # Positive advantages, so the tokens' gradients add up instead of cancelling.
    advantages = torch.linspace(0.5, 1.5, RESPONSE_LENGTH).unsqueeze(0).repeat(len(BODY_LENGTHS), 1)
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
            ROLLOUT_ENGINE_DP_RANKS_KEY: torch.arange(len(BODY_LENGTHS)) % ENGINE_RANKS,
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


@contextmanager
def _policy(cfg, model_path: str) -> Iterator[PPORayActorGroup]:
    """The gradient probe policy of ``cfg`` on its own Ray instance, shut down on exit."""
    _init_ray(cfg)
    try:
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
        yield policy
    finally:
        ray.shutdown()


def _eval_logprobs(policy: PPORayActorGroup, batch: TrainingInputBatch) -> torch.Tensor:
    outputs = ray.get(policy.async_run_ray_method("mesh", "forward", data=batch))
    return concatenate_outputs_after_mesh_dispatch(policy.actor_infos, outputs)["output"].float()


def _gradients(
    policy: PPORayActorGroup, batch: TrainingInputBatch, output_dir: Path, *, recompute: bool
) -> dict[str, float]:
    """Run the mini-batch through ``policy`` and return the metrics; the gradients land in ``output_dir``."""
    batch.metadata = {**batch.metadata, "output_dir": str(output_dir), "recompute": recompute}
    return ray.get(policy.async_run_ray_method("mesh", "forward_backward_gradients", batch))[0]


def _load_gradients(output_dir: Path) -> dict[str, torch.Tensor]:
    gradients = {}
    for rank in range(POLICY_WORLD_SIZE):
        for name, grad in torch.load(output_dir / f"pp{rank}.pt", map_location="cpu", mmap=True).items():
            gradients[f"pp{rank}/{name}"] = grad
    return gradients


def _relative_difference(value: torch.Tensor, reference: torch.Tensor) -> float:
    """The L2 norm of ``value - reference`` over that of ``reference``; 0 when both are zero."""
    difference = (value.double() - reference.double()).norm().item()
    return difference / max(reference.double().norm().item(), torch.finfo(torch.float64).tiny)


def test_exact_gradients_equal_with_and_without_recompute_and_follow_native_gradients(tmp_path):
    """Under exact numerics on two pipeline stages, the same forward-backward gives the same gradients bit for bit
    with full activation recompute in one-layer units and without recompute, and every gradient is within ``NATIVE_RELATIVE_TOLERANCE`` of native numerics' gradient on the same weights, batch and expert routes,
    each mode's ratios taken against its own eval forward."""
    require_hoppers(POLICY_WORLD_SIZE)
    model_path = tmp_path / "model"
    model_path.mkdir()
    # Zero router weights send every token to the four experts of highest bias under both numerics.
    write_snowball_checkpoint(model_path, router_weight_scale=0.0)
    batch = _batch(PAD_TOKEN_ID, vocab_size=1000)
    with _policy(_config(str(model_path), NumericsMode.EXACT), str(model_path)) as policy:
        eval_logprobs = _eval_logprobs(policy, batch)
        batch["action_log_probs"] = (eval_logprobs * batch["response_mask"]).float()
        metrics = {
            "a": _gradients(policy, batch, tmp_path / "a", recompute=True),
            "b": _gradients(policy, batch, tmp_path / "b", recompute=False),
        }
    with _policy(_config(str(model_path), NumericsMode.NATIVE), str(model_path)) as policy:
        batch["action_log_probs"] = (_eval_logprobs(policy, batch) * batch["response_mask"]).float()
        metrics["native"] = _gradients(policy, batch, tmp_path / "native", recompute=False)

    for arm, arm_metrics in metrics.items():
        print(f"{arm}: losses {arm_metrics['losses']}, log-ratio max abs {arm_metrics['log_ratio_abs_max']}")
    for arm in ("a", "b"):
        # The training forward reproduces the eval forward's log-probabilities bit for bit.
        assert metrics[arm]["log_ratio_abs_max"] == 0.0, (arm, metrics[arm]["log_ratio_abs_max"])
    assert metrics["a"]["losses"] == metrics["b"]["losses"], metrics

    a, b, native = (_load_gradients(tmp_path / arm) for arm in ("a", "b", "native"))
    assert a.keys() == b.keys() == native.keys()
    recompute_differs = [name for name in a if not torch.equal(a[name], b[name])]
    assert not recompute_differs, f"{len(recompute_differs)} of {len(a)} gradients differ: {recompute_differs}"
    relative = {name: _relative_difference(a[name], native[name]) for name in a}
    for name in sorted(relative, key=relative.get, reverse=True)[:10]:
        print(f"{name}: exact vs native relative difference {relative[name]:.3e}")
    far = {name: value for name, value in relative.items() if value > NATIVE_RELATIVE_TOLERANCE}
    assert not far, far
