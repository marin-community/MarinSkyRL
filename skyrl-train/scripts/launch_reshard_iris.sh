#!/usr/bin/env bash
set -euo pipefail

# Launch a CPU-only Iris job that converts a FSDP2 checkpoint to a Hugging Face
# safetensors repository. Set the credentials in the invoking environment; the
# job receives only the values it needs for the checkpoint store and HF upload.

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly RESHARD_SCRIPT="$SCRIPT_DIR/reshard_fsdp2_to_hf.py"

: "${MARIN_ROOT:?Set MARIN_ROOT to the Marin checkout that provides the iris CLI.}"
: "${HF_TOKEN:?Set HF_TOKEN for the destination Hugging Face repository.}"
: "${AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID for the checkpoint object store.}"
: "${AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY for the checkpoint object store.}"

# No default on purpose. The deployed digests live in cloud/iris/launch_rl_iris.py
# (DEFAULT_RL_DOCKER_IMAGE / DEFAULT_RL_MEGATRON_DOCKER_IMAGE), which is their single
# source of truth. A second copy here went stale and kept pointing at a retired image
# in the old registry org long after the launcher had moved on.
: "${TASK_IMAGE:?Set TASK_IMAGE to a gpu-rl image reference. Take the current digest from DEFAULT_RL_MEGATRON_DOCKER_IMAGE in cloud/iris/launch_rl_iris.py.}"
readonly TASK_IMAGE
readonly RL_PYTHON="${RL_PYTHON:-/opt/openthoughts/envs/rl/bin/python}"
readonly S3_PREFIX="${S3_PREFIX:-s3://marin-us-east-02a/iris/delphi-1e23-wc50m-rl-d1-rlvrmath-32gpu/checkpoints/global_step_101/policy}"
readonly HF_REPO="${HF_REPO:-laion/delphi-1e23-wc50m-rl-d1-rlvrmath-32gpu}"
readonly WORLD_SIZE="${WORLD_SIZE:-16}"
readonly JOB_NAME="${JOB_NAME:-reshard-rl-d1-rlvrmath}"
readonly IRIS_CONFIG="${IRIS_CONFIG:-lib/iris/config/cw-us-east-02a.yaml}"
readonly RESHARD_B64="$(base64 < "$RESHARD_SCRIPT" | tr -d '\n')"

read -r -d '' IN_POD <<EOF || true
set -euo pipefail
export HF_HUB_OFFLINE=0
export AWS_S3_ADDRESSING_STYLE=virtual
export AWS_ENDPOINT_URL=https://cwobject.com
export AWS_REGION=us-east-1
export AWS_DEFAULT_REGION=us-east-1
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY || true

work_dir=/tmp/reshard_work
mkdir -p "\$work_dir/checkpoint" "\$work_dir/output"
echo "\$RESHARD_B64" | base64 -d > "\$work_dir/reshard_fsdp2_to_hf.py"
"$RL_PYTHON" "\$work_dir/reshard_fsdp2_to_hf.py" \\
  --s3-prefix "$S3_PREFIX" \\
  --world-size "$WORLD_SIZE" \\
  --local-dir "\$work_dir/checkpoint" \\
  --out-dir "\$work_dir/output" \\
  --hf-repo "$HF_REPO"
EOF

cd "$MARIN_ROOT"
uv run iris --config="$IRIS_CONFIG" job run \\
  --task-image "$TASK_IMAGE" \\
  --no-sync \\
  --enable-extra-resources \\
  --cpu 16 --memory 220GB --disk 170GB \\
  --priority interactive \\
  --job-name "$JOB_NAME" \\
  --no-wait \\
  -e RESHARD_B64 "$RESHARD_B64" \\
  -e HF_TOKEN "$HF_TOKEN" \\
  -e HF_HUB_OFFLINE 0 \\
  -e CW_AKID "$AWS_ACCESS_KEY_ID" \\
  -e CW_SECRET "$AWS_SECRET_ACCESS_KEY" \\
  -- bash -lc "$IN_POD"
