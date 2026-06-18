set -x

# ---------------------------------------------------------------------------
# Long-context Context-Parallel (CP) GRPO recipe — FSDP2 ring-SDPA.
#
# This is the Stage-6 example recipe for torch-native Context Parallel on the
# FSDP2 backend. CP shards the SEQUENCE dimension across `context_parallel_size`
# GPUs and runs ring SDPA, so a single training forward can hold a sequence that
# would OOM a single GPU. Use it for EXTREME-long-context RL (long prompts +
# long generations) where the per-GPU activation memory of the full sequence is
# the bottleneck — not for short sequences (ring SDPA has comm overhead that is
# only worth paying when the sequence otherwise does not fit).
#
# Cloned from examples/gsm8k/run_gsm8k.sh; the CP-specific knobs are flagged
# inline. NUM_GPUS must be a multiple of CONTEXT_PARALLEL_SIZE.
#
#   uv run examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
#   NUM_GPUS=4 CONTEXT_PARALLEL_SIZE=2 bash examples/cp_longctx/run_cp_longctx.sh
#
# ===========================================================================
# CP REQUIREMENTS / CONSTRAINTS (read before changing the seq-len knobs)
# ===========================================================================
#  * strategy MUST be fsdp2 (CP is FSDP2-only).
#  * attn_backend MUST be sdpa or flex — NEVER flash_attention_2 (flash-attn
#    varlen is CP-incompatible; G2 rejects it). We force sdpa here.
#  * sequence_parallel_size MUST be 1 — CP and Ulysses SP both shard the seq dim
#    and are mutually exclusive (G2).
#  * use_sample_packing MUST be false — CP runs the dense [B,S] path (G2). SkyRL's
#    flash-varlen packing is CP-incompatible (a packed-CP SDPA path is future work).
#  * G4 DIVISIBILITY: the PADDED sequence length must satisfy
#        seq_len % (2 * context_parallel_size) == 0
#    for torch's built-in CP zigzag load balancer. The model_wrapper RIGHT-pads
#    the batch up to this multiple automatically (and strips the pad after the
#    per-token unshard), so you do NOT have to hand-align (max_prompt_length +
#    max_generate_length); but keeping them already-divisible avoids the pad.
#  * LEFT-PAD AUTO-REALIGN (Stage 5/5b): CP ring SDPA runs pure-causal with
#    attention_mask=None inside the context. Causality only masks TRAILING
#    (right) pads; LEFT-padding would corrupt attention (real tokens attend back
#    across the leading pads → ~1.0 logprob error vs cp=1). SkyRL's collator
#    LEFT-pads prompts, so the CP forward DETECTS the per-row leading-pad count
#    and ROLLS each row left (left-flush) before the CP context, then INVERTS the
#    roll on the per-token logprobs/entropy afterward — the returned tensors are
#    byte-identical column order to cp=1. Gated by SKYRL_CP_REQUIRE_RIGHT_ALIGN=1
#    (default); set 0 only if alignment is already guaranteed upstream (skips the
#    per-step realign). cp_size==1 is an unconditional no-op.
#  * Production precision is bf16 (fp16 was diagnostic only). Only the
#    "allgather" rotate method works on torch 2.11 (all_to_all → NotImplemented).
# ===========================================================================

: "${DATA_DIR:="$HOME/data/gsm8k"}"
: "${NUM_GPUS:=4}"
: "${CONTEXT_PARALLEL_SIZE:=2}"   # CP ring degree. NUM_GPUS % this must be 0.
: "${LOGGER:=wandb}"              # "console" to print to stdout
: "${INFERENCE_BACKEND:=vllm}"

# Long-context knobs: large prompt + generation. These are what make CP pay off;
# at these lengths a single-GPU dense forward of the full sequence is the OOM.
: "${MAX_PROMPT_LENGTH:=8192}"
: "${MAX_GENERATE_LENGTH:=8192}"

uv run --isolated --extra $INFERENCE_BACKEND -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen3-1.7B" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  `# ---- CP knobs (the whole point of this recipe) ----` \
  trainer.policy.fsdp_config.context_parallel_size=$CONTEXT_PARALLEL_SIZE \
  trainer.policy.fsdp_config.cp_style=ring_sdpa \
  trainer.policy.fsdp_config.cp_rotate_method=allgather \
  trainer.ref.fsdp_config.context_parallel_size=$CONTEXT_PARALLEL_SIZE \
  trainer.ref.fsdp_config.cp_rotate_method=allgather \
  trainer.policy.sequence_parallel_size=1 \
  trainer.ref.sequence_parallel_size=1 \
  trainer.flash_attn=false \
  trainer.attn_backend=sdpa \
  trainer.use_sample_packing=false \
  `# ---------------------------------------------------` \
  trainer.epochs=20 \
  trainer.eval_batch_size=256 \
  trainer.eval_before_train=false \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=64 \
  trainer.policy_mini_batch_size=32 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_GENERATE_LENGTH \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=5 \
  generator.gpu_memory_utilization=0.7 \
  trainer.logger="$LOGGER" \
  trainer.project_name="cp_longctx" \
  trainer.run_name="cp_longctx_test" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/cp_longctx_ckpt" \
  $@
