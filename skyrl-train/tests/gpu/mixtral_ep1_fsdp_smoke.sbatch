#!/bin/bash
#SBATCH --job-name=mixtral-smoke
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=00:30:00
#SBATCH --output=/e/scratch/jureap59/feuer1/mixtral_smoke_%j.out

set -x
SIF=/e/scratch/jureap59/feuer1/containers/skyrl_megatron_vllm0202rc0_r3.sif
SKYRL=/e/scratch/jureap59/feuer1/OpenThoughts-Agent/SkyRL/skyrl-train
HFCACHE=/e/data1/datasets/playground/ot-baf/hf_hub

export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_USE_IPV6=0
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29577

apptainer exec --nv \
  --bind /e/scratch --bind /e/data1 \
  --env PYTHONPATH=$SKYRL \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env HF_HUB_CACHE=$HFCACHE --env HF_HOME=$HFCACHE \
  --env VLLM_USE_FLASHINFER_SAMPLER=0 \
  --env VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  --env TRITON_LIBCUDA_PATH=/.singularity.d/libs \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env NCCL_SOCKET_FAMILY=AF_INET --env GLOO_USE_IPV6=0 \
  "$SIF" \
  torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29577 \
    "$SKYRL/tests/gpu/mixtral_ep1_fsdp_smoke.py"
