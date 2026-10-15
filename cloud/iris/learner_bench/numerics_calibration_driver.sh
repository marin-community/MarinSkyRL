#!/usr/bin/env bash
# Rank-0 driver (run by ray_gang.py) for the numerics calibration of the 8-layer Hero on 4 H100 nodes:
# calibrate on three layouts (one a micro-batch change), record a held-out layout and the must-fail controls,
# then compare. LB_LR sets both MuonH learning rates. Steps whose output exists are skipped on a rerun.
set -euo pipefail

py=/tmp/lb-venv/bin/python
numerics=("$py" -m skyrl_train.benchmarks.learner.numerics)
out="$LB_OUTPUT/numerics"
common=(
  --run-name "$LB_RUN_NAME" --harness-revision "$LB_REVISION" --model-path "$LB_MODEL_PATH"
  --tp 1 --cp 1 --etp 1 --num-nodes "$LB_NODES" --gpus-per-node 8
  --optimizer MuonH --lr "$LB_LR" --adam-lr "$LB_LR" --offload-momentum
  --recompute full_layers --sample-packing --no-flash-attn
  --ensemble-root "$LB_ENSEMBLE_ROOT" --ensemble-version "$LB_ENSEMBLE_VERSION" --profiles short
  --routes-checkpoint "$LB_ROUTES_CHECKPOINT" --warmup-updates 0 --measured-updates 1
  --override trainer.policy.megatron_config.ddp_config.grad_reduce_in_fp32=true
  --override trainer.policy.megatron_config.ddp_config.overlap_grad_reduce=true
  --override trainer.policy.megatron_config.ddp_config.overlap_param_gather=false
  --output "$LB_OUTPUT/unused.json"
)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exists() { "$py" -c "import sys; from skyrl_train.io import io; sys.exit(0 if io.exists(sys.argv[1]) else 1)" "$1"; }

record() {
  local name="$1"
  shift
  exists "$out/$name.json" && return
  "${numerics[@]}" record --record-output "$out/$name.json" --num-updates 2 "${common[@]}" "$@"
}
record pp2-ep16-mbs1 --pp 2 --ep 16 --micro-batch-size-per-gpu 1
record pp4-ep8-mbs1 --pp 4 --ep 8 --micro-batch-size-per-gpu 1
record pp2-ep16-mbs2 --pp 2 --ep 16 --micro-batch-size-per-gpu 2
record pp8-ep4-mbs1 --pp 8 --ep 4 --micro-batch-size-per-gpu 1
record pp2-ep16-mbs1-negate --control negate_advantages --pp 2 --ep 16 --micro-batch-size-per-gpu 1
record pp2-ep16-mbs1-double --control double_advantages --pp 2 --ep 16 --micro-batch-size-per-gpu 1

"${numerics[@]}" calibrate --records "$out/pp2-ep16-mbs1.json" "$out/pp4-ep8-mbs1.json" "$out/pp2-ep16-mbs2.json" \
  --reference "$out/reference.json"
for candidate in pp8-ep4-mbs1 pp2-ep16-mbs1-negate pp2-ep16-mbs1-double; do
  status=0
  "${numerics[@]}" compare --reference "$out/reference.json" --record "$out/$candidate.json" \
    --output "$out/compare-$candidate.json" || status=$?
  echo "numerics compare $candidate exit=$status"
done
