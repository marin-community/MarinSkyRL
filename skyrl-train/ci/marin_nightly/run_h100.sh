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
# The scheduled workflow runs in the cluster's standard Iris task image. This script creates the
# same frozen root environment used by normal image-free launches before exercising the trainer.
# The flags below spell out what examples/gsm8k/run_gsm8k.sh would have passed, sized for one GPU.
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
MAX_STEPS="${MAX_STEPS:-30}"
DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k_nightly}"
LOG="${LOG:-$PWD/nightly-run.log}"
SPEC="${SPEC:-ci/marin_nightly/specs/gsm8k-qwen3-0.6b.json}"
NIGHTLY_RL_ENV="${NIGHTLY_RL_ENV:-$REPOSITORY_ROOT/.iris-nightly-env}"
RUNTIME_ENV_FILE="$NIGHTLY_RL_ENV/marinskyrl-runtime.sh"
PYTHON="$NIGHTLY_RL_ENV/bin/python"

# The defaults below are the behavioural shape the shipped gate spec is calibrated against: 30
# GRPO steps at batch 32 with 8 samples/prompt is enough for reward to visibly climb on a 0.6B
# policy (~0.04 -> ~0.20 over the run), which is what lets the gate check learning and not just
# liveness. The shape stays env-overridable so the same script can be driven as a quicker smoke
# check (smaller batch, fewer steps) without editing it -- but then point it at a spec whose
# thresholds match, since the shipped spec expects this shape.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
POLICY_MINI_BATCH_SIZE="${POLICY_MINI_BATCH_SIZE:-16}"
MICRO_BATCH="${MICRO_BATCH:-8}"
N_SAMPLES="${N_SAMPLES:-8}"
MAX_GEN_LEN="${MAX_GEN_LEN:-512}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-512}"
LR="${LR:-2.0e-6}"

# train_batch_size * MAX_STEPS prompts get consumed; keep some margin. Evaluation is off, but
# data.val_data still has to resolve, so a handful of rows is enough.
TRAIN_ROWS="${TRAIN_ROWS:-2000}"
VAL_ROWS="${VAL_ROWS:-16}"

echo "::: resolving the frozen MarinSkyRL runtime"
bash "$REPOSITORY_ROOT/cloud/iris/bootstrap_runtime.sh" \
  "$REPOSITORY_ROOT" \
  "$NIGHTLY_RL_ENV" \
  "$RUNTIME_ENV_FILE" \
  fsdp \
  production
source "$RUNTIME_ENV_FILE"

cd "$REPOSITORY_ROOT/skyrl-train"

# Ray's zip runtime cannot preserve the repository symlink, so materialize the sibling
# environment package before setting the source paths.
rm -rf skyrl-gym
cp -R ../skyrl-gym skyrl-gym
export PYTHONPATH="$PWD/skyrl-gym:$PWD${PYTHONPATH:+:$PYTHONPATH}"

echo "::: GPU and driver"
nvidia-smi --query-gpu=name,driver_version --format=csv

test -x "$PYTHON"
echo "::: using the frozen root environment at ${NIGHTLY_RL_ENV}"
"$PYTHON" -c "import torch, vllm; print(f'torch {torch.__version__} | vllm {vllm.__version__}')"

echo "::: preparing a ${TRAIN_ROWS}-prompt GSM8K slice"
"$PYTHON" examples/gsm8k/gsm8k_dataset.py --output_dir "$DATA_DIR"
DATA_DIR="$DATA_DIR" TRAIN_ROWS="$TRAIN_ROWS" VAL_ROWS="$VAL_ROWS" "$PYTHON" - <<'PY'
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
echo "::: shape: batch=${TRAIN_BATCH_SIZE} samples=${N_SAMPLES} gen_len=${MAX_GEN_LEN} lr=${LR}"
# vLLM warms up DeepGEMM FP8 kernels whenever the GPU supports them (is_deep_gemm_supported() is
# true on Hopper) regardless of whether the `deep_gemm` package actually imported -- and it is not
# in this environment, so the warmup hard-fails at engine start. This is a bf16 model that never
# uses FP8, so disable DeepGEMM outright. Exported so the Ray-spawned vLLM workers inherit it.
export VLLM_USE_DEEP_GEMM=0
# With no compiled flash-attn (see the sync step) the policy trains on eager attention
# (trainer.flash_attn=false), and sample packing has to be disabled alongside it: packing
# sequences into one relies on flash-attn's varlen kernel, so model_wrapper asserts
# flash_attention_2 whenever use_sample_packing is true.
START=$(date +%s)
"$PYTHON" -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.use_kl_loss=true \
  trainer.policy.model.path="$MODEL" \
  trainer.policy.optimizer_config.lr="$LR" \
  trainer.strategy=fsdp2 \
  trainer.flash_attn=false \
  trainer.use_sample_packing=false \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_gpus_per_node=1 \
  trainer.placement.critic_num_gpus_per_node=1 \
  trainer.placement.ref_num_gpus_per_node=1 \
  trainer.epochs=1 \
  trainer.max_steps="$MAX_STEPS" \
  trainer.train_batch_size="$TRAIN_BATCH_SIZE" \
  trainer.policy_mini_batch_size="$POLICY_MINI_BATCH_SIZE" \
  trainer.micro_train_batch_size_per_gpu="$MICRO_BATCH" \
  trainer.micro_forward_batch_size_per_gpu="$MICRO_BATCH" \
  trainer.update_epochs_per_batch=1 \
  trainer.max_prompt_length="$MAX_PROMPT_LEN" \
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
  generator.n_samples_per_prompt="$N_SAMPLES" \
  generator.sampling_params.max_generate_length="$MAX_GEN_LEN" \
  generator.gpu_memory_utilization=0.7 \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  environment.env_class=gsm8k \
  2>&1 | tee "$LOG"
ELAPSED=$(( $(date +%s) - START ))

echo "::: gating (run took ${ELAPSED}s)"
"$PYTHON" -m ci.marin_nightly.gate \
  --log "$LOG" --spec "$SPEC" --wall-clock-seconds "$ELAPSED"

echo "::: checking Grug PyTorch parity against the committed Levanter fixture"
"$PYTHON" -m tests.grug_training_parity
