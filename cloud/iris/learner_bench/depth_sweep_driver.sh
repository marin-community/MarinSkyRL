#!/usr/bin/env bash
# Rank-0 driver (run by ray_gang.py) for one depth of the shallow-Hero depth sweep: the replay learner benchmark on
# one layout at each update size in LB_SIZES (sequences per update), smallest first. Each size writes
# $LB_OUTPUT/n<size>.json; sizes whose result exists are skipped on a rerun, and a failed size does not stop the
# larger ones.
set -euo pipefail

py=/tmp/lb-venv/bin/python
common=(
  --harness-revision "$LB_REVISION" --model-path "$LB_MODEL_PATH"
  --tp 1 --pp "$LB_PP" --ep "$LB_EP" --cp 1 --etp 1 --num-nodes "$LB_NODES" --gpus-per-node 8
  --optimizer MuonH --lr "$LB_LR" --adam-lr "$LB_LR" --offload-momentum
  --recompute full_layers --micro-batch-size-per-gpu "$LB_MICRO" --sample-packing --no-flash-attn
  --ensemble-root "$LB_ENSEMBLE_ROOT" --ensemble-version "$LB_ENSEMBLE_VERSION" --profiles short
  --routes-checkpoint "$LB_ROUTES_CHECKPOINT" --warmup-updates "$LB_WARMUP" --measured-updates "$LB_MEASURED"
  --score-old-logprobs --memory-breakdown
  --override trainer.policy.megatron_config.ddp_config.grad_reduce_in_fp32=true
  --override trainer.policy.megatron_config.ddp_config.overlap_grad_reduce=true
  --override trainer.policy.megatron_config.ddp_config.overlap_param_gather=false
)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exists() { "$py" -c "import sys; from skyrl_train.io import io; sys.exit(0 if io.exists(sys.argv[1]) else 1)" "$1"; }

failed=()
for size in $LB_SIZES; do
  output="$LB_OUTPUT/n${size}.json"
  exists "$output" && continue
  status=0
  "$py" -m skyrl_train.benchmarks.learner.driver "${common[@]}" --run-name "$LB_RUN_NAME-n${size}" \
    --num-sequences "$size" --output "$output" || status=$?
  echo "depth sweep size ${size} exit=${status}"
  [[ "$status" == 0 ]] || failed+=("$size")
done
if [[ ${#failed[@]} -gt 0 ]]; then
  echo "depth sweep failed sizes: ${failed[*]}" >&2
  exit 1
fi
