set -x

# DPO training on dataset-supplied preference pairs (no inference engines are launched).
# Single-node Megatron policy + colocated frozen reference; the completions come from the
# parquet, so the whole GPU budget goes to training.
#
# uv run examples/dpo/ultrafeedback_dataset.py --output_dir "$HOME/data/ultrafeedback"
# bash examples/dpo/run_dpo.sh
#
# Override with e.g.: NUM_GPUS=8 MODEL=alignment-handbook/zephyr-7b-sft-full BETA=0.01 LR=5e-7 bash examples/dpo/run_dpo.sh

: "${DATA_DIR:="$HOME/data/ultrafeedback"}"
: "${MODEL:="Qwen/Qwen2.5-0.5B-Instruct"}"
: "${NUM_GPUS:=1}"
: "${LOGGER:=console}"
: "${BETA:=0.1}"
: "${LR:=1e-6}"
: "${TRAIN_BATCH_SIZE:=128}"
: "${MAX_LEN:=1024}"
: "${MAX_PROMPT_LEN:=512}"
: "${OUTPUT_TAG:="dpo"}"

uv run --isolated --extra megatron -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train_prefs.parquet']" \
  data.val_data="[]" \
  +algorithm_recipe=dpo \
  environment.env_class=preference_pair \
  trainer.strategy=megatron \
  trainer.placement.colocate_all=false \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  trainer.policy.model.path="$MODEL" \
  trainer.ref.model.path="$MODEL" \
  trainer.epochs=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$TRAIN_BATCH_SIZE \
  trainer.micro_train_batch_size_per_gpu=8 \
  trainer.micro_forward_batch_size_per_gpu=8 \
  trainer.use_sample_packing=false \
  trainer.max_prompt_length=$MAX_PROMPT_LEN \
  trainer.ckpt_interval=50 \
  trainer.algorithm.dpo.beta=$BETA \
  trainer.policy.optimizer_config.lr=$LR \
  generator.n_samples_per_prompt=2 \
  generator.sampling_params.max_generate_length=$MAX_LEN \
  generator.max_input_length=$MAX_PROMPT_LEN \
  generator.trajectory_retention.enabled=false \
  trainer.eval_interval=-1 \
  trainer.logger="$LOGGER" \
  trainer.project_name="dpo" \
  trainer.run_name="${OUTPUT_TAG}_$(basename "$MODEL")" \
  trainer.ckpt_path="$HOME/ckpts/${OUTPUT_TAG}_ckpt" \
  trainer.export_path="$HOME/exports/${OUTPUT_TAG}" \
  $@
