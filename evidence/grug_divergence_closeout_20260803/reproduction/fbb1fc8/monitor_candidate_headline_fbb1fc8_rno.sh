#!/usr/bin/env bash
set -euo pipefail

KUBECTL=(kubectl --kubeconfig /home/romain/.kube/coreweave-iris --context marin-rn02a_RNO2A)
IRIS_ROOT=/home/romain/dev/marin-wt/grug-training-perf-gap-20260731
JOB_ID=/romain/grug-paired-eager-grouped-fbb1fc8-s1-rno-20260803
LOG=/tmp/grug-final-fbb1fc8-rno-host-monitor.log
PODS=(
  iris-romain-grug-paired-eager-group-01d11a37-0-8460ccc86b5579c7
  iris-romain-grug-paired-eager-group-1243ea95-0-05528ef07f320699
  iris-romain-grug-paired-eager-group-18f5fd0b-0-e39e5feba0fad90f
  iris-romain-grug-paired-eager-group-19aa91ce-0-8bf46be354d3e624
)

stop_job() {
  local reason=$1
  printf '%s STOP reason=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$reason" | tee -a "$LOG"
  (
    cd "$IRIS_ROOT"
    MARIN_CLUSTER=coreweave .venv/bin/iris --config lib/iris/config/cw-rno2a.yaml job stop "$JOB_ID"
  ) 2>&1 | tee -a "$LOG"
  exit 2
}

: >> "$LOG"
seen_running=0
while true; do
  now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  terminal=0
  absent=0
  for pod in "${PODS[@]}"; do
    phase=$("${KUBECTL[@]}" get pod -n iris "$pod" -o jsonpath='{.status.phase}' 2>/dev/null || true)
    if [[ -z "$phase" ]]; then
      phase=absent
      ((absent += 1))
    elif [[ "$phase" == Succeeded || "$phase" == Failed ]]; then
      ((terminal += 1))
    elif [[ "$phase" == Running ]]; then
      seen_running=1
    fi

    stats=
    if [[ "$phase" == Running ]]; then
      stats=$(timeout 5 "${KUBECTL[@]}" exec -n iris "$pod" -c task -- sh -c '
        printf "memory.current=%s memory.max=%s swap.current=%s swap.max=%s " \
          "$(cat /sys/fs/cgroup/memory.current)" \
          "$(cat /sys/fs/cgroup/memory.max)" \
          "$(cat /sys/fs/cgroup/memory.swap.current)" \
          "$(cat /sys/fs/cgroup/memory.swap.max)"
        tr "\n" "," < /sys/fs/cgroup/memory.events
      ' 2>/dev/null || true)
    fi
    printf '%s pod=%s phase=%s %s\n' "$now" "$pod" "$phase" "$stats" | tee -a "$LOG"

    if [[ "$stats" =~ memory.current=([0-9]+)[[:space:]]memory.max=([0-9]+)[[:space:]]swap.current=([0-9]+)[[:space:]]swap.max=([^[:space:]]+) ]]; then
      memory_current=${BASH_REMATCH[1]}
      memory_max=${BASH_REMATCH[2]}
      swap_current=${BASH_REMATCH[3]}
      swap_max=${BASH_REMATCH[4]}
      if (( memory_current * 100 >= memory_max * 90 )); then
        stop_job "${pod}_memory_ge_90pct"
      fi
      if [[ "$swap_max" == 0 ]] && (( swap_current > 0 )); then
        stop_job "${pod}_swap_nonzero_with_zero_limit"
      fi
      if [[ "$swap_max" =~ ^[0-9]+$ ]] && (( swap_max > 0 )) && (( swap_current * 100 >= swap_max * 75 )); then
        stop_job "${pod}_swap_ge_75pct"
      fi
    fi
  done

  if (( terminal == ${#PODS[@]} )); then
    exit 0
  fi
  if (( seen_running == 1 && absent == ${#PODS[@]} )); then
    printf '%s WARN all pods absent from this read; retaining monitor because the API may be transient\n' "$now" | tee -a "$LOG"
  fi
  sleep 10
done
