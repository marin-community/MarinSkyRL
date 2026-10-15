set -x

# Colocated GRPO training+generation for Qwen/Qwen3-Coder-30B-A3B-Instruct on the SWE-Bench task.
# Uses 2 node with 8 GPUs each.
# See README.md for task materialization with pinned dataset revisions.
# bash examples/mini_swe_agent/run_mini_swe_30B.sh

# ensure that all worker nodes can access this data directory
DATA_DIR="$DATA/data/swe_gym_subset"

CKPT_PATH="$DATA/ckpts/llm_mini_swe"


NUM_GPUS=8
NNODES=2
NUM_INFERENCE_ENGINES=4
TP_SIZE=4
LOGGER=wandb

# We use a small batch size here for demonstration
uv run --project .. --extra megatron --extra vllm -m skyrl_train.entrypoints.taskcompendium \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen3-Coder-30B-A3B-Instruct" \
  trainer.placement.colocate_all=true \
  trainer.strategy=megatron \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.policy_num_nodes=$NNODES \
  trainer.placement.ref_num_nodes=$NNODES \
  trainer.policy.sequence_parallel_size=4 \
  generator.num_inference_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine_tensor_parallel_size=$TP_SIZE \
  trainer.epochs=20 \
  trainer.eval_batch_size=16 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=16 \
  trainer.policy_mini_batch_size=16 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=4096 \
  generator.sampling_params.max_generate_length=4096 \
  generator.max_input_length=30720 \
  generator.max_turns=50 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=vllm \
  generator.run_engines_locally=True \
  generator.weight_sync_backend=nccl \
  generator.n_samples_per_prompt=4 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="mini_swe" \
  trainer.run_name="mini_swe_32B_swe_gym" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$CKPT_PATH" \
  "$@"
