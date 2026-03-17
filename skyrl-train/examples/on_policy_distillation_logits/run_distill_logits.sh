set -x

# On-policy distillation with teacher logits.
# Uses a vLLM-served teacher model (supports AWQ/GPTQ quantization).
#
# Setup:
# bash examples/algorithms/dapo/prepare_dapo_data.sh
# bash examples/on_policy_distillation_logits/run_distill_logits.sh

DATA_DIR="$HOME/data/dapo"
TRAIN_FILE="$DATA_DIR/dapo-math-17k-cleaned.parquet"
TEST_FILE="$DATA_DIR/aime-2024-cleaned.parquet"
LOGGER=wandb

# Teacher model (set this to your teacher model path)
# For AWQ quantized models, also add: +teacher.engine_init_kwargs.quantization=awq
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-4B}"
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen3-1.7B-Base}"

# On-policy distillation args
ADVANTAGE_ESTIMATOR="no_op"
POLICY_LOSS="importance_sampling"
USE_KL_IN_REWARD=true
USE_KL_LOSS=false

# Placement args (adjust to your GPU setup)
NUM_GPUS_PER_NODE=8
NUM_STUDENT_ENGINES=8
STUDENT_ENGINE_TP_SIZE=1
NUM_TEACHER_ENGINES=1
TEACHER_ENGINE_TP_SIZE=1

# Sampling params
TEMPERATURE=1.0
TOP_P=1.0
EVAL_TOP_P=0.7

# Training params
TRAIN_BATCH_SIZE=512
MINI_BATCH_SIZE=512
N_SAMPLES_PER_PROMPT=16
EVAL_N_SAMPLES_PER_PROMPT=32
LR=1e-5

python -m examples.on_policy_distillation_logits.main_on_policy_distill_logits \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="['$TEST_FILE']" \
  trainer.algorithm.advantage_estimator=$ADVANTAGE_ESTIMATOR \
  trainer.algorithm.policy_loss_type=$POLICY_LOSS \
  trainer.policy.model.path=$STUDENT_MODEL \
  trainer.ref.model.path=$TEACHER_MODEL \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  generator.num_inference_engines=$NUM_STUDENT_ENGINES \
  generator.inference_engine_tensor_parallel_size=$STUDENT_ENGINE_TP_SIZE \
  teacher.model_path=$TEACHER_MODEL \
  teacher.num_inference_engines=$NUM_TEACHER_ENGINES \
  teacher.inference_engine_tensor_parallel_size=$TEACHER_ENGINE_TP_SIZE \
  teacher.top_k_logprobs=256 \
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
  trainer.project_name="on_policy_distillation_logits" \
  trainer.run_name="distill_logits_${STUDENT_MODEL##*/}_from_${TEACHER_MODEL##*/}" \
  trainer.resume_mode=latest \
  trainer.export_path="$HOME/exports/distill_logits" \
  trainer.max_ckpts_to_keep=3 \
  trainer.ckpt_interval=10 \
  trainer.ckpt_path="$HOME/ckpts/distill_logits" \
  $@
