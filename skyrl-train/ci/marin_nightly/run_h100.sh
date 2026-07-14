#!/usr/bin/env bash
# The nightly end-to-end training run. Executes INSIDE the Iris job, on one H100, with the
# repo bundled as the working directory; .github/workflows/marin-nightly.yaml submits it.
#
# Runs the smallest real thing the trainer can do -- colocated GRPO on GSM8K with a 0.6B
# policy over a few prompts -- and then gates the metrics it logged. The point is that the
# user-facing training path works end to end: rollouts generate, rewards score, the policy
# takes a step, and weights sync back into the inference engine. It says nothing about
# model quality, and is not meant to.
#
# Every uv call is --frozen: the nightly installs exactly what the lockfile pins, so a failure
# here is the trainer's, not a resolver's. The Hydra flags below spell out what
# examples/gsm8k/run_gsm8k.sh would have passed, sized for one GPU.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
MAX_STEPS="${MAX_STEPS:-2}"
DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k_nightly}"
LOG="${LOG:-$PWD/nightly-run.log}"
SPEC="${SPEC:-ci/marin_nightly/specs/gsm8k-qwen3-0.6b.json}"

# train_batch_size(8) * MAX_STEPS prompts get consumed; keep some margin. Evaluation is
# off, but data.val_data still has to resolve, so a handful of rows is enough.
TRAIN_ROWS="${TRAIN_ROWS:-64}"
VAL_ROWS="${VAL_ROWS:-8}"

cd "$(dirname "$0")/../.."   # skyrl-train/

# torch comes from the cu130 index, so the pod needs an NVIDIA driver >= 580. Log what we got:
# if a CUDA-13 wheel ever lands on an older driver, the failure is otherwise cryptic.
echo "::: GPU and driver"
nvidia-smi --query-gpu=name,driver_version --format=csv

echo "::: syncing skyrl-train (vllm extra)"
uv sync --frozen --extra vllm --extra dev

echo "::: preparing a ${TRAIN_ROWS}-prompt GSM8K slice"
uv run --frozen python examples/gsm8k/gsm8k_dataset.py --output_dir "$DATA_DIR"
DATA_DIR="$DATA_DIR" TRAIN_ROWS="$TRAIN_ROWS" VAL_ROWS="$VAL_ROWS" uv run --frozen python - <<'PY'
import os
import pathlib

import polars as pl

data_dir = pathlib.Path(os.environ["DATA_DIR"])
for name, rows in (("train", int(os.environ["TRAIN_ROWS"])), ("validation", int(os.environ["VAL_ROWS"]))):
    path = data_dir / f"{name}.parquet"
    frame = pl.read_parquet(path).head(rows)
    frame.write_parquet(path)
    print(f"{path}: {frame.height} rows")
PY

echo "::: training ${MODEL} for ${MAX_STEPS} steps on one GPU"
START=$(date +%s)
uv run --frozen --extra vllm -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.use_kl_loss=true \
  trainer.policy.model.path="$MODEL" \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.strategy=fsdp2 \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_gpus_per_node=1 \
  trainer.placement.critic_num_gpus_per_node=1 \
  trainer.placement.ref_num_gpus_per_node=1 \
  trainer.epochs=1 \
  trainer.max_steps="$MAX_STEPS" \
  trainer.train_batch_size=8 \
  trainer.policy_mini_batch_size=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.update_epochs_per_batch=1 \
  trainer.max_prompt_length=256 \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.ckpt_interval=-1 \
  trainer.hf_save_interval=-1 \
  trainer.resume_mode=null \
  trainer.dump_eval_results=false \
  trainer.logger=console \
  trainer.project_name=marin_nightly \
  trainer.run_name=gsm8k_h100 \
  generator.backend=vllm \
  generator.num_inference_engines=1 \
  generator.inference_engine_tensor_parallel_size=1 \
  generator.n_samples_per_prompt=4 \
  generator.sampling_params.max_generate_length=256 \
  generator.gpu_memory_utilization=0.7 \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  environment.env_class=gsm8k \
  2>&1 | tee "$LOG"
ELAPSED=$(( $(date +%s) - START ))

echo "::: gating (run took ${ELAPSED}s)"
uv run --frozen python -m ci.marin_nightly.gate \
  --log "$LOG" --spec "$SPEC" --wall-clock-seconds "$ELAPSED"
