set -x

# Fully async GRPO training+generation for Qwen2.5-1.5B-Instruct on GSM8K.
# Training and generation run on separate GPUs (colocate_all=false), and generation may run up to
# MAX_STALENESS_STEPS policy steps ahead of training.

# uv run examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export WANDB_API_KEY=<your_key_here>
# bash examples/fully_async/async_run_gsm8k.sh

# NOTE (sumanthrh): `micro_train_batch_size_per_gpu` and `micro_forward_batch_size_per_gpu` can be tuned

# You can override the default values with e.g.: `MAX_STALENESS_STEPS=1 bash examples/fully_async/async_run_gsm8k.sh`.

: "${DATA_DIR:="$HOME/data/gsm8k"}"
: "${NUM_INFERENCE_GPUS:=2}"
: "${NUM_POLICY_GPUS:=2}"
: "${LOGGER:=wandb}" # change to "console" to print to stdout / or use wandb

: "${INFERENCE_BACKEND:=vllm}"
# : "${INFERENCE_BACKEND:=sglang}"

# Rollout buffer knobs; see docs/tutorials/fully_async.rst.
: "${MINI_BATCH_SIZE:=256}"
: "${MAX_STALENESS_STEPS:=4}"

uv run --isolated --extra megatron --extra $INFERENCE_BACKEND -m skyrl_train.entrypoints.gym_worker_pool \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.rollout_buffer.max_staleness_steps=${MAX_STALENESS_STEPS} \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.policy_loss_type="behavior_clip" \
  trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct" \
  trainer.placement.colocate_all=false \
  trainer.strategy=megatron \
  trainer.placement.policy_num_gpus_per_node=$NUM_POLICY_GPUS \
  trainer.placement.critic_num_gpus_per_node=$NUM_POLICY_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_POLICY_GPUS \
  generator.num_inference_engines=$NUM_INFERENCE_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=20 \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=false \
  trainer.eval_interval=4 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=${MINI_BATCH_SIZE} \
  trainer.policy_mini_batch_size=${MINI_BATCH_SIZE} \
  trainer.micro_forward_batch_size_per_gpu=8 \
  trainer.micro_train_batch_size_per_gpu=8 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=5 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="gsm8k-async" \
  trainer.run_name="gsm8k-test-fully_async-4xl4" \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$HOME/ckpts/gsm8k_1.5B_ckpt" \
  generator.chat_template.source=name \
  generator.chat_template.name_or_path="qwen2_5_with_generation_tag_simplified" \
  $@
