#!/bin/bash
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --job-name=skyrl-s8-gdn-torch211
#SBATCH --output=/e/scratch/jureap59/feuer1/s8_gdn_torch211_%j.out

# Stage-8 GDN parity+speed on the NATIVE torch-2.11 SIF + rebuilt fla overlay.
# 1 GPU on reformo free short tier -> does NOT touch a3.
module load Apptainer 2>/dev/null || true
B=/e/scratch/jureap59/feuer1
SIF=$B/containers/skyrl_megatron_vllm_r3_torch211.sif
OV=$B/containers/fla_tilelang_torch211_overlay.img
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
apptainer exec --nv --overlay "${OV}:ro" --bind /e/scratch --bind /e/data1 \
  "$SIF" python -u /e/scratch/jureap59/feuer1/p2_gdn_fla_tilelang.py
echo "=== GDN_TORCH211_SBATCH_DONE rc=$? ==="
