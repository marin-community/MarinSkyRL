import argparse
from pathlib import Path

import ray
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marinskyrl.checkpoint_paths import POLICY_CHECKPOINT_SUBDIRECTORY
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils import validate_cfg
from skyrl_train.workers.worker import PPORayActorGroup
from tests.cpu.tiny_training.cpu_backend import CausalLMPolicy, CPUPolicyWorker
from tests.cpu.tiny_training.experiment import (
    LOGICAL_CPUS,
    WORKER_ENV_VARS,
    RolloutShape,
    TrainingMode,
    tiny_training_config,
)
from tests.cpu.tiny_training.tiny_model import build_tiny_policy


def fixed_training_batch(model_dir: str) -> TrainingInputBatch:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    answer = tokenizer.encode("123", add_special_tokens=False)
    eos = tokenizer.eos_token_id
    response = [*answer, eos]
    assert len(response) == 4
    mask = torch.tensor([[1, 0, 0, 0]] * 4 + [[1, 1, 1, 1]] * 4, dtype=torch.int64).repeat(2, 1)
    sequences = torch.tensor([[eos, answer[1], answer[0], *response]] * 16)
    attention_mask = torch.cat([torch.ones((16, 3), dtype=torch.int64), mask], dim=1)
    sequences[:, -4:] = torch.where(mask.bool(), sequences[:, -4:], tokenizer.pad_token_id)
    model = CausalLMPolicy(AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)).eval()
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        old_log_probs = model(sequences, num_actions=4, attention_mask=attention_mask)
    advantages = torch.tensor([1, 1, -0.5, 1, -1, -1, 0.25, -1], dtype=torch.float32).repeat(2)
    batch = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "response_mask": mask,
            "loss_mask": mask,
            "action_log_probs": old_log_probs,
            "base_action_log_probs": None,
            "advantages": advantages[:, None].expand(-1, 4).clone(),
            "values": None,
            "returns": None,
        }
    )
    batch.metadata = {"global_step": 1, "response_length": 4}
    return batch


def run_fixed_update(root: Path, model_dir: Path, micro_batch_size: int) -> None:
    cfg = tiny_training_config(
        root,
        model_dir,
        TrainingMode.SYNC,
        RolloutShape.SINGLE_TURN,
        max_steps=1,
        checkpoint_interval=1,
        dp_size=2,
        micro_batch_size=micro_batch_size,
        max_in_flight=1,
    )
    validate_cfg(cfg)
    batch = fixed_training_batch(cfg.trainer.policy.model.path)
    dump_dir = root / "exports/dumped_data"
    dump_dir.mkdir(parents=True, exist_ok=True)
    batch.save(dump_dir / "global_step_1_training_input.pkl")
    ray.init(num_cpus=LOGICAL_CPUS, num_gpus=2, runtime_env={"env_vars": WORKER_ENV_VARS}, include_dashboard=False)
    try:
        workers = PPORayActorGroup(cfg, 1, 2, ray.remote(num_gpus=1)(CPUPolicyWorker))
        ray.get(workers.async_init_model(cfg.trainer.policy.model.path, num_training_steps=1))
        output = workers.run_method("mesh", "ppo_train", batch)
        assert output.metadata["train_status"]["raw_grad_norm"] > 0
        workers.run_method(
            "pass_through",
            "save_checkpoint",
            ckpt_dir=str(root / "ckpts/global_step_1" / POLICY_CHECKPOINT_SUBDIRECTORY),
        )
        workers.kill_actors()
    finally:
        ray.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one CPU policy update on fixed unequal-length rows.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--micro-batch-size", type=int, required=True)
    args = parser.parse_args()
    run_fixed_update(args.root, build_tiny_policy(args.root / "model"), args.micro_batch_size)


if __name__ == "__main__":
    main()
