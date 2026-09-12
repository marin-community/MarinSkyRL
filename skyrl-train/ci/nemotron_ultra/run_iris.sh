#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

if [[ -n "$(git status --short)" ]]; then
    echo "Nemotron Ultra acceptance must run from a clean, committed worktree." >&2
    exit 2
fi

job_name="${JOB_NAME:-nemotron-ultra-rlvr-$(date +%Y%m%d-%H%M%S)}"
target_cluster="${TARGET_CLUSTER:-cw-rno2a}"
runtime_commit="$(git rev-parse HEAD)"
model="${MODEL:-Qwen/Qwen3-8B}"
log_path="${LOG_PATH:-nemotron-ultra-rlvr-acceptance.log}"
secrets_env="${SECRETS_ENV:-/dev/null}"
job_user="${IRIS_JOB_USER:-${USER:?USER must be set}}"
job_path="/${job_user}/${job_name}"
config_dir="$(uv run --frozen python -c 'from pathlib import Path; import iris; print(Path(iris.__file__).parent / "config")')"

cleanup() {
    uv run --frozen iris --cluster marin job cancel "$job_path" >/dev/null 2>&1 || true
}
trap cleanup EXIT

PYTHONPATH=. uv run --frozen python -m cloud.iris.iris_backend \
    --rl_config cloud/iris/configs/nemotron_ultra_rlvr_acceptance.yaml \
    --entrypoint skyrl_train.entrypoints.nemotron_ultra_acceptance \
    --model_path "$model" \
    --train_data '[]' \
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
    --timeout 3600 \
    --no-wait \
    --secrets-env "$secrets_env"

set +e
uv run --frozen iris --cluster marin job wait "$job_path"
wait_status=$?
uv run --frozen iris --cluster marin job logs "$job_path" --max-lines 50000 >"$log_path"
logs_status=$?
set -e

if ((wait_status != 0)); then
    echo "Nemotron Ultra Iris job failed: $job_path" >&2
    exit "$wait_status"
fi
if ((logs_status != 0)); then
    echo "Could not retrieve the Nemotron Ultra Iris job log: $job_path" >&2
    exit "$logs_status"
fi

PYTHONPATH=skyrl-train:. uv run --frozen python -m ci.nemotron_ultra.gate --log "$log_path"
