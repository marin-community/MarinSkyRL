set -x

# Best-of-N distillation without teacher logits.
# Generates N completions per prompt, selects best by reward, trains via SFT.
#
# NOTE: train_batch_size should be N× larger than usual since selection
# reduces the effective batch by factor N.
#
# Setup:
# bash examples/algorithms/dapo/prepare_dapo_data.sh
# bash examples/best_of_n_distillation/run_best_of_n.sh

DATA_DIR="$HOME/data/dapo"
TRAIN_FILE="$DATA_DIR/dapo-math-17k-cleaned.parquet"
TEST_FILE="$DATA_DIR/aime-2024-cleaned.parquet"
LOGGER=wandb

STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen3-1.7B-Base}"

# Best-of-N args
N_SAMPLES_PER_PROMPT=16
ADVANTAGE_ESTIMATOR="uniform"
POLICY_LOSS="sft"
USE_KL_IN_REWARD=false
USE_KL_LOSS=false

# Placement args
NUM_GPUS_PER_NODE=8
NUM_INFERENCE_ENGINES=8
INFERENCE_ENGINE_TP_SIZE=1

# Sampling params
TEMPERATURE=1.0
TOP_P=1.0
EVAL_TOP_P=0.7

# Training params — batch_size is N× larger to compensate for selection
TRAIN_BATCH_SIZE=512
MINI_BATCH_SIZE=512
EVAL_N_SAMPLES_PER_PROMPT=32
LR=1e-5

python -m examples.best_of_n_distillation.main_best_of_n \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="['$TEST_FILE']" \
  trainer.algorithm.advantage_estimator=$ADVANTAGE_ESTIMATOR \
  trainer.algorithm.policy_loss_type=$POLICY_LOSS \
  trainer.policy.model.path=$STUDENT_MODEL \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  generator.num_inference_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine_tensor_parallel_size=$INFERENCE_ENGINE_TP_SIZE \
  trainer.epochs=20 \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=2048 \
  generator.enforce_eager=true \
  generator.sampling_params.max_generate_length=8192 \
  generator.sampling_params.temperature=$TEMPERATURE \
  generator.sampling_params.top_p=$TOP_P \
  generator.eval_sampling_params.temperature=$TEMPERATURE \
  generator.eval_sampling_params.top_p=$EVAL_TOP_P \
  generator.eval_sampling_params.max_generate_length=8192 \
  generator.eval_n_samples_per_prompt=$EVAL_N_SAMPLES_PER_PROMPT \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.policy.optimizer_config.num_warmup_steps=0 \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.use_kl_in_reward=$USE_KL_IN_REWARD \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.async_engine=false \
  generator.batched=true \
  environment.env_class=aime \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="best_of_n_distillation" \
  trainer.run_name="best_of_${N_SAMPLES_PER_PROMPT}_${STUDENT_MODEL##*/}" \
  trainer.resume_mode=latest \
  trainer.export_path="$HOME/exports/best_of_n" \
  trainer.max_ckpts_to_keep=3 \
  trainer.ckpt_interval=10 \
  trainer.ckpt_path="$HOME/ckpts/best_of_n" \
  $@
