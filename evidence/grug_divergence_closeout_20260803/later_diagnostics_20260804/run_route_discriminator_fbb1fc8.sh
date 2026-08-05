#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:?candidate image digest is required}
MODE=${MODE:?preflight or headline mode is required}
RESULT=${RESULT:?result S3 URI is required}
RENDEZVOUS=${RENDEZVOUS:?rendezvous S3 URI is required}
SOURCE_REVISION=fbb1fc8378601e0346d00d186809f10d1ad0360d
MODEL=marin-community/grug-67b-a2b-sft-s2-thinking-step630
MODEL_REVISION=a822321c2c21af099189e7116104b3cf5142c119
MANIFEST=s3://marin-us-east-02a/iris/grug-training-perf-gap/20260731/replay-step-1-global/e81f387763177ae55faccf9a2747c2568d59c6efcee7f10d752958771e95f50d/manifest.json
MANIFEST_SHA=5d2479bbbdcd4ca04a9f7d11de82ce42830fbae878d734cdc3c4a4f123f93b74
LOGICAL_SHA=e81f387763177ae55faccf9a2747c2568d59c6efcee7f10d752958771e95f50d
PY=/opt/openthoughts/envs/rl/bin/python
DRIVER=/app/skyrl-train/scripts/grug_fixed_replay_benchmark.py
WORKER=/app/skyrl-train/skyrl_train/workers/fsdp/fsdp_worker.py
MOE=/app/skyrl-train/skyrl_train/models/grug_moe.py
BAKED_MOE=/opt/skyrl/skyrl-train/skyrl_train/models/grug_moe.py
CONTROLLER=/app/cloud/iris/start_rl_iris_controller.py
CGROUP_MEMORY_STOP_PERCENT=85
CGROUP_SWAP_STOP_PERCENT=75
HOST_MEMORY_AVAILABLE_STOP_PERCENT=10
HOST_SWAP_FREE_STOP_PERCENT=25
SAFETY_KILL_GRACE_SECONDS=30

export HF_HOME=/hf/cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_SOCKET_IFNAME='^ibs,ibp,lo,docker,veth,cilium,lxc'
export PYTHONPATH=/app:/app/skyrl-train:/opt/skyrl/skyrl-train
export PYTHONUNBUFFERED=1

run_gate() {
  target_args=()
  case "$MODE" in
    preflight)
      target_args+=(
        --route-discriminator-target 0:0:first
        --route-discriminator-target 0:0:middle
        --route-discriminator-target 0:0:last
      )
      ;;
    headline)
      target_args+=(
        --route-discriminator-target 1:28:middle
        --route-discriminator-target 8:114:first
        --route-discriminator-target 16:33:middle
      )
      ;;
    *)
      printf 'unsupported MODE=%s\n' "$MODE" >&2
      exit 2
      ;;
  esac
  exec "$PY" "$DRIVER" \
    --model "$MODEL" \
    --model-revision "$MODEL_REVISION" \
    --source-revision "$SOURCE_REVISION" \
    --image "$IMAGE" \
    --manifest-s3-uri "$MANIFEST" \
    --manifest-sha256 "$MANIFEST_SHA" \
    --logical-batch-sha256 "$LOGICAL_SHA" \
    --mode "$MODE" \
    --attention-backend flash_attention_2 \
    --expert-implementation eager \
    --objective paired_matched_ce \
    --route-discriminator \
    "${target_args[@]}" \
    --result-s3-uri "$RESULT" \
    --sample 1
}

if [[ "${1:-}" == gate ]]; then
  run_gate
fi

test "$(sha256sum "$DRIVER" | cut -d' ' -f1)" = bbdc711b3d26b5127a71b3b8e24f7f3dfb5e00ba94e1a51f28f9bf83111dd084
test "$(sha256sum "$WORKER" | cut -d' ' -f1)" = d66c1c3ee148a8aef0007d1d3e17af4ef522381c0107f836d3c0817805fc0de4
test "$(sha256sum "$MOE" | cut -d' ' -f1)" = 2dd51d842afe6f57fe00337c21945923da68664d9c23678633c578c27504ba93
test "$(sha256sum "$BAKED_MOE" | cut -d' ' -f1)" = 2dd51d842afe6f57fe00337c21945923da68664d9c23678633c578c27504ba93
cmp -s "$MOE" "$BAKED_MOE"
test "$(sha256sum "$CONTROLLER" | cut -d' ' -f1)" = f2e8484de7d5566f7e39d6d7e8ef3c03744960a53dd9a39c20a9dfba4db6d2ba
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader
free -h
for file in memory.current memory.max memory.swap.current memory.swap.max; do
  printf '%s=' "$file"
  sed -n '1p' "/sys/fs/cgroup/$file" 2>/dev/null || printf 'unavailable\n'
done

"$PY" "$CONTROLLER" \
  --rendezvous-dir "$RENDEZVOUS" \
  --prestage-model "$MODEL" \
  --model-warm-source s3://marin-us-east-02a/models/marin-community--grug-67b-a2b-sft-s2-thinking-step630 \
  -- \
  /bin/bash "$0" gate &
controller_pid=$!
safety_reason=
external_termination=0

# shellcheck disable=SC2317  # invoked indirectly by trap
forward_termination() {
  external_termination=1
  if kill -0 "$controller_pid" 2>/dev/null; then
    kill -TERM "$controller_pid" 2>/dev/null || true
  fi
}
trap forward_termination INT TERM

read_cgroup_metric() {
  sed -n '1p' "/sys/fs/cgroup/$1" 2>/dev/null || printf 'unavailable\n'
}

read_meminfo_kib() {
  awk -v name="$1:" '$1 == name { print $2; exit }' /proc/meminfo
}

while kill -0 "$controller_pid" 2>/dev/null; do
  memory_current=$(read_cgroup_metric memory.current)
  memory_max=$(read_cgroup_metric memory.max)
  swap_current=$(read_cgroup_metric memory.swap.current)
  swap_max=$(read_cgroup_metric memory.swap.max)
  host_memory_total_kib=$(read_meminfo_kib MemTotal)
  host_memory_available_kib=$(read_meminfo_kib MemAvailable)
  host_swap_total_kib=$(read_meminfo_kib SwapTotal)
  host_swap_free_kib=$(read_meminfo_kib SwapFree)
  printf '%s RESOURCE_GUARD memory.current=%s memory.max=%s swap.current=%s swap.max=%s host.MemTotal_kib=%s host.MemAvailable_kib=%s host.SwapTotal_kib=%s host.SwapFree_kib=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$memory_current" "$memory_max" "$swap_current" "$swap_max" \
    "$host_memory_total_kib" "$host_memory_available_kib" "$host_swap_total_kib" "$host_swap_free_kib"

  if [[ "$memory_current" =~ ^[0-9]+$ && "$memory_max" =~ ^[0-9]+$ ]] \
    && (( memory_current * 100 >= memory_max * CGROUP_MEMORY_STOP_PERCENT )); then
    safety_reason="cgroup_memory_ge_${CGROUP_MEMORY_STOP_PERCENT}pct"
  elif [[ "$swap_current" =~ ^[0-9]+$ && "$swap_max" == 0 ]] && (( swap_current > 0 )); then
    safety_reason=cgroup_swap_nonzero_with_zero_limit
  elif [[ "$swap_current" =~ ^[0-9]+$ && "$swap_max" =~ ^[0-9]+$ ]] \
    && (( swap_max > 0 )) \
    && (( swap_current * 100 >= swap_max * CGROUP_SWAP_STOP_PERCENT )); then
    safety_reason="cgroup_swap_ge_${CGROUP_SWAP_STOP_PERCENT}pct"
  elif [[ "$host_memory_total_kib" =~ ^[0-9]+$ && "$host_memory_available_kib" =~ ^[0-9]+$ ]] \
    && (( host_memory_total_kib > 0 )) \
    && (( host_memory_available_kib * 100 <= host_memory_total_kib * HOST_MEMORY_AVAILABLE_STOP_PERCENT )); then
    safety_reason="host_memory_available_le_${HOST_MEMORY_AVAILABLE_STOP_PERCENT}pct"
  elif [[ "$host_swap_total_kib" =~ ^[0-9]+$ && "$host_swap_free_kib" =~ ^[0-9]+$ ]] \
    && (( host_swap_total_kib > 0 )) \
    && (( host_swap_free_kib * 100 <= host_swap_total_kib * HOST_SWAP_FREE_STOP_PERCENT )); then
    safety_reason="host_swap_free_le_${HOST_SWAP_FREE_STOP_PERCENT}pct"
  fi

  if [[ -n "$safety_reason" ]]; then
    printf '%s RESOURCE_GUARD_STOP reason=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$safety_reason" >&2
    sed -n '1,40p' /sys/fs/cgroup/memory.events 2>/dev/null || true
    kill -TERM "$controller_pid" 2>/dev/null || true
    break
  fi
  sleep 5
done

if [[ -n "$safety_reason" ]]; then
  for ((second = 0; second < SAFETY_KILL_GRACE_SECONDS; second++)); do
    if ! kill -0 "$controller_pid" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 "$controller_pid" 2>/dev/null; then
    printf '%s RESOURCE_GUARD_KILL reason=%s grace_seconds=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$safety_reason" "$SAFETY_KILL_GRACE_SECONDS" >&2
    kill -KILL "$controller_pid" 2>/dev/null || true
  fi
  wait "$controller_pid" 2>/dev/null || true
  exit 70
fi

set +e
wait "$controller_pid"
controller_status=$?
set -e
if (( external_termination )); then
  exit 143
fi
exit "$controller_status"
