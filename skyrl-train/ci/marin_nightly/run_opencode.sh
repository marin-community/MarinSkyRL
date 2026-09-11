#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

job_name="${JOB_NAME:?JOB_NAME must be set}"
target_cluster="${TARGET_CLUSTER:-cw-rno2a}"
runtime_commit="${RUNTIME_COMMIT:-$(git rev-parse HEAD)}"
model="${MODEL:-Qwen/Qwen3-8B}"
log_path="${LOG_PATH:-opencode-nightly.log}"
secrets_env="${SECRETS_ENV:-/dev/null}"
mode="${OPENCODE_MODE:-nightly}"
job_user="${IRIS_JOB_USER:-${USER:?USER must be set}}"
job_path="/${job_user}/${job_name}"
config_dir="$(uv run --frozen python -c 'from pathlib import Path; import iris; print(Path(iris.__file__).parent / "config")')"
started_at="$(date +%s)"
train_data="/app/marinskyrl/skyrl-train/ci/opencode_smoke/tasks/exact-continuation"
gate_spec="skyrl-train/ci/marin_nightly/specs/opencode-qwen3-8b.json"
extra_overrides=()

case "$mode" in
    nightly)
        ;;
    compaction-stress)
        train_data="/app/marinskyrl/skyrl-train/ci/opencode_smoke/tasks/boundary-mix"
        gate_spec="skyrl-train/ci/marin_nightly/specs/opencode-compaction-stress.json"
        extra_overrides=(
            context_budget.request_window_tokens=8192
            context_budget.max_new_tokens_per_turn=256
            context_budget.max_turns=12
            terminal_bench_config.harbor.opencode_config.compaction.auto=true
            terminal_bench_config.harbor.opencode_config.compaction.reserved=4096
            terminal_bench_config.harbor.override_timeout_sec=null
            terminal_bench_config.harbor.verifier_override_timeout_sec=null
            terminal_bench_config.harbor.max_retries=0
        )
        ;;
    overflow-stress)
        train_data="/app/marinskyrl/skyrl-train/ci/opencode_smoke/tasks/boundary-mix"
        gate_spec="skyrl-train/ci/marin_nightly/specs/opencode-overflow-stress.json"
        extra_overrides=(
            context_budget.request_window_tokens=8192
            context_budget.max_new_tokens_per_turn=256
            context_budget.max_turns=12
            terminal_bench_config.harbor.opencode_config.compaction.auto=false
            terminal_bench_config.harbor.override_timeout_sec=null
            terminal_bench_config.harbor.verifier_override_timeout_sec=null
            terminal_bench_config.harbor.max_retries=0
        )
        ;;
    *)
        echo "unknown OPENCODE_MODE: $mode" >&2
        exit 2
        ;;
esac

skyrl_override_args=()
if ((${#extra_overrides[@]})); then
    for override in "${extra_overrides[@]}"; do
        skyrl_override_args+=(--skyrl_override "$override")
    done
fi

cleanup() {
    iris --cluster marin job cancel "$job_path" >/dev/null 2>&1 || true
}
trap cleanup EXIT

PYTHONPATH=. uv run --frozen python -m cloud.iris.iris_backend \
    --rl_config cloud/iris/configs/opencode_smoke_literal.yaml \
    --model_path "$model" \
    --train_data "[\"$train_data\"]" \
    --cluster "$target_cluster" \
    --cluster-config "$config_dir/$target_cluster.yaml" \
    --target-cluster "$target_cluster" \
    --parent-cluster-config "$config_dir/marin.yaml" \
    --runtime-commit "$runtime_commit" \
    --job-name "$job_name" \
    --num-nodes 1 \
    --gpus-per-node 8 \
    --gpu-variant H100 \
    --memory 1500GB \
    --disk 1000GB \
    --priority batch \
    --max-retries 0 \
    --timeout 2400 \
    --no-wait \
    --secrets-env "$secrets_env" \
    --skyrl_override trainer.max_steps=1 \
    --skyrl_override trainer.logger=console \
    --skyrl_override trainer.algorithm.tito_full=true \
    ${skyrl_override_args[@]+"${skyrl_override_args[@]}"}

set +e
iris --cluster marin job wait "$job_path"
wait_status=$?
iris --cluster marin job logs "$job_path" --max-lines 50000 >"$log_path"
logs_status=$?
set -e

if ((wait_status != 0)); then
    echo "OpenCode Iris job failed: $job_path" >&2
    exit "$wait_status"
fi
if ((logs_status != 0)); then
    echo "Could not retrieve the OpenCode Iris job log: $job_path" >&2
    exit "$logs_status"
fi

finished_at="$(date +%s)"
uv run --frozen python skyrl-train/ci/marin_nightly/gate.py \
    --log "$log_path" \
    --spec "$gate_spec" \
    --wall-clock-seconds "$((finished_at - started_at))"
