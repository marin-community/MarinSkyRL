#!/bin/bash
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --reservation=reformo
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --job-name=pin-smoke
#SBATCH --output=/e/scratch/jureap59/feuer1/tmp_pinning_smoke/logs/%x_%j.out

# ==============================================================================
# GH200 per-GPU-bundle GPU-pinning smoke (2 nodes x 4 GH200, Qwen3-0.6B).
#
# Reproduces the SAME placement path the 80B exercises (SIF Ray, unmasked
# CUDA_VISIBLE_DEVICES, colocate_all=false, policy STRICT-spread -> per-GPU
# {GPU:1} bundles), at small scale, and asserts every policy actor pins a
# DISTINCT physical GPU (by UUID), NUMA matches, NVLink locality holds.
#
# - Runs INSIDE skyrl_megatron_vllm.sif (apptainer exec --nv) so the
#   ray_noset_visible_devices()/unmasked-CVD behavior the fix targets is real.
# - PYTHONPATH points at the ISOLATED worktree's skyrl-train (the fix code),
#   then sif_pydeps for hydra. The shared envs/rl + shared SkyRL clone are
#   NEVER touched.
# - Multi-node Ray: head on node 0, worker srun'd onto node 1, both in the SIF.
#
# Override knobs (env): SMOKE_TRIALS (default 3), SMOKE_PER_GPU_BUNDLES
# ("true"/"false" -> sets cfg.trainer.placement.policy_per_gpu_bundles for the
# control run via env read in the test? No -- control handled by separate var).
# ==============================================================================
set -eo pipefail
ulimit -c 0
ulimit -n 65536 2>/dev/null || true

export DCFT=/e/scratch/jureap59/feuer1/OpenThoughts-Agent
WORKDIR="$DCFT"
WT=/e/scratch/jureap59/feuer1/tmp_pinning_smoke/SkyRL
SIF=/e/scratch/jureap59/feuer1/containers/skyrl_megatron_vllm.sif
PYD=/e/scratch/jureap59/feuer1/sif_pydeps
cd "$WORKDIR"

module load GCC/14.3.0 nvidia-compilers/25.9-CUDA-13 2>/dev/null || true
if [ -f "$DCFT/hpc/dotenv/jupiter.env" ]; then
  set -a; source "$DCFT/hpc/dotenv/jupiter.env"; set +a
fi

# The smoke's source MUST be the worktree (fix code), not the shared clone.
# Put worktree skyrl-train FIRST, then pydeps (hydra/omegaconf for the SIF).
export SMOKE_PYTHONPATH="$WT/skyrl-train:$PYD"

# Networking / stability env (subset of the 80B RL preamble).
export GLOO_USE_IPV6=0
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_SOCKET_IFNAME=ib0
export VLLM_FORCE_IPV4=1
export VLLM_USE_V1=1
export UV_USE_IO_URING=0
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export RAY_DEDUP_LOGS=0
export SMOKE_NUM_NODES=2
export SMOKE_GPUS_PER_NODE=4
export SMOKE_TRIALS="${SMOKE_TRIALS:-3}"

# Apptainer prefix (no overlays needed for a placement smoke; SIF Python only).
APPT=(apptainer exec --nv --bind /e/scratch --bind /e/data1
      --env "PYTHONPATH=$SMOKE_PYTHONPATH"
      --env "WANDB_MODE=offline" "$SIF")

LOGDIR=/e/scratch/jureap59/feuer1/tmp_pinning_smoke/logs
mkdir -p "$LOGDIR"

# --- Resolve node list / head IP (IPv4) ---
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD_NODE="${NODES[0]}"
WORKER_NODE="${NODES[1]}"
HEAD_IP=$(srun --export=ALL -N1 -n1 -w "$HEAD_NODE" bash -c 'hostname -i | awk "{print \$1}"')
RAY_PORT=6379
RAY_ADDR="${HEAD_IP}:${RAY_PORT}"

echo "=== pin-smoke launch ==="
echo "JOBID=$SLURM_JOB_ID  nodes=${NODES[*]}  head=$HEAD_NODE($HEAD_IP)  worker=$WORKER_NODE"
echo "WT=$WT  SIF=$SIF"
echo "WT-HEAD-expected=722bce8"
# Prove the worktree's skyrl_train is what will be imported (worktree wins over SIF install).
"${APPT[@]}" python -c "import skyrl_train; print('skyrl_train resolved ->', skyrl_train.__file__)" 2>&1 | grep -i skyrl_train || true
echo "trials=$SMOKE_TRIALS"

# Clean any stale ray state on both nodes (inside SIF for matching runtime).
srun --export=ALL -N1 -n1 -w "$HEAD_NODE"   "${APPT[@]}" ray stop --force 2>/dev/null || true
srun --export=ALL -N1 -n1 -w "$WORKER_NODE" "${APPT[@]}" ray stop --force 2>/dev/null || true
sleep 2

# --- Start Ray HEAD on node 0 (background srun, inside SIF) ---
srun --export=ALL,VLLM_HOST_IP="$HEAD_IP" -N1 -n1 -w "$HEAD_NODE" \
  "${APPT[@]}" ray start --head --node-ip-address="$HEAD_IP" --port="$RAY_PORT" \
  --num-gpus=4 --block > "$LOGDIR/ray_head_${SLURM_JOB_ID}.log" 2>&1 &
HEAD_PID=$!

# Wait for head GCS to come up.
for i in $(seq 1 30); do
  if "${APPT[@]}" bash -c "RAY_ADDRESS=$RAY_ADDR ray status" >/dev/null 2>&1; then
    echo "Ray head up after ${i}s"; break
  fi
  sleep 2
done

# --- Start Ray WORKER on node 1 (background srun, inside SIF) ---
WORKER_IP=$(srun --export=ALL -N1 -n1 -w "$WORKER_NODE" bash -c 'hostname -i | awk "{print \$1}"')
srun --export=ALL,VLLM_HOST_IP="$WORKER_IP" -N1 -n1 -w "$WORKER_NODE" \
  "${APPT[@]}" ray start --address="$RAY_ADDR" --node-ip-address="$WORKER_IP" \
  --num-gpus=4 --block > "$LOGDIR/ray_worker_${SLURM_JOB_ID}.log" 2>&1 &
WORKER_PID=$!

# Wait until the cluster reports 8 GPUs (2 nodes x 4).
for i in $(seq 1 40); do
  GPUS=$("${APPT[@]}" bash -c "RAY_ADDRESS=$RAY_ADDR python -c 'import ray; ray.init(address=\"auto\"); print(int(ray.cluster_resources().get(\"GPU\",0)))'" 2>/dev/null | tail -1)
  echo "cluster GPU count: $GPUS"
  if [ "$GPUS" = "8" ]; then echo "Ray cluster ready (8 GPU)"; break; fi
  sleep 3
done

# --- Run the smoke (pytest) DIRECTLY in the batch script (already on the head
# node), inside the SIF. NOT via a third srun step: the ray-head/-worker srun
# steps above run `--block` (forever) and hold the nodes' single task slots
# (--ntasks-per-node=1), so a third `srun -w HEAD` deadlocks waiting for a slot.
# The batch shell itself runs on the head node and can reach Ray via address=auto. ---
RUNLOG="$LOGDIR/pytest_${SLURM_JOB_ID}.log"
echo "RUNLOG=$RUNLOG"
set +e
cd "$WT/skyrl-train"
RAY_ADDRESS="$RAY_ADDR" "${APPT[@]}" bash -c \
  "cd $WT/skyrl-train && RAY_ADDRESS=$RAY_ADDR PYTHONPATH=$SMOKE_PYTHONPATH \
  python -u -m pytest tests/gpu/test_per_gpu_bundle_pinning.py -q -s -p no:cacheprovider 2>&1" \
  | stdbuf -oL -eL tee "$RUNLOG"
RC=${PIPESTATUS[0]}
set -e
echo "=== pytest EXIT $RC ==="

# --- Teardown (--overlap: the head/worker ray-start srun steps still hold the
# task slots via `--block`, so teardown steps must overlap them). ---
srun --overlap --export=ALL -N1 -n1 -w "$HEAD_NODE"   "${APPT[@]}" ray stop --force 2>/dev/null || true
srun --overlap --export=ALL -N1 -n1 -w "$WORKER_NODE" "${APPT[@]}" ray stop --force 2>/dev/null || true
kill "$HEAD_PID" "$WORKER_PID" 2>/dev/null || true
echo "=== pin-smoke DONE rc=$RC ==="
exit $RC
