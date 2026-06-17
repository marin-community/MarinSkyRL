<div align="center">

# SkyRL: A Modular Full-stack RL Library for LLMs


[![🌐 NovaSky](https://img.shields.io/badge/-Visit%20Website-5865F2?style=for-the-badge)](https://novasky-ai.github.io/) [![Github](https://img.shields.io/badge/SkyRL-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/NovaSky-AI/SkyRL) [![Twitter](https://img.shields.io/badge/NovaSky-white?style=for-the-badge&logo=X&logoColor=000&color=000&labelColor=white)](https://x.com/NovaSkyAI) [![Hugging Face Collection](https://img.shields.io/badge/NovaSky-fcd022?style=for-the-badge&logo=huggingface&logoColor=000&labelColor)](https://huggingface.co/NovaSky-AI) [![Discord](https://img.shields.io/badge/NovaSky-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/cJF2JUaaAN) [![Documentation](https://img.shields.io/badge/Documentation-blue?style=for-the-badge&logo=readthedocs&logoColor=white)](https://skyrl.readthedocs.io/en/latest/)



<div align="center" style="font-family: Arial, sans-serif;">
  <p>
    <a href="#news" style="text-decoration: none; font-weight: bold;">News</a> •
    <a href="#links" style="text-decoration: none; font-weight: bold;">Links</a> •
    <a href="#getting-started" style="text-decoration: none; font-weight: bold;">Getting Started</a> •
    <a href="#citation" style="text-decoration: none; font-weight: bold;">Citation</a> •
    <a href="#acknowledgement" style="text-decoration: none; font-weight: bold;">Acknowledgement</a> 
  </p>
</div>

</div>

# Overview of this fork

This is a fork of SkyRL maintained for the [Marin project](https://github.com/marin-community/marin) (`marin-community`), where it powers agentic RL training (SkyRL + Harbor). It was originally developed for the [OpenThoughts-Agent project](https://github.com/open-thoughts/OpenThoughts-Agent); that line of work now continues here under Marin.

We aim to upstream these changes to the main SkyRL branch.

The walkthrough below reproduces the original OpenThoughts-Agent v1 release (kept here for reference), i.e.:
- Using [open-thoughts/OpenThinker-Agent-v1-SFT](https://huggingface.co/open-thoughts/OpenThinker-Agent-v1-SFT) as base
- GRPO with the data [open-thoughts/OpenThoughts-Agent-v1-RL](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-RL), while
- Evaluating with [open-thoughts/OpenThoughts-TB-dev](https://huggingface.co/datasets/open-thoughts/OpenThoughts-TB-dev), and 
- Getting the final [open-thoughts/OpenThinker-Agent-v1](https://huggingface.co/open-thoughts/OpenThinker-Agent-v1)

### Environment

Install SkyRL

```bash
conda create -n otagent python=3.12
conda activate otagent
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

git clone https://github.com/mlfoundations/SkyRL
cd SkyRL/skyrl-train/
pip install -e .
pip install "vllm==0.10.1.1"
cd ../..
```

Install Harbor
```bash
git clone https://github.com/CharlieFRuan/harbor
cd harbor
git checkout 112425-terminus2-messages
pip install -e .
```

Remainings
```bash
pip install fastapi uvicorn
```

We will soon make things uv-syncable.

### Data preparation

```bash
conda activate otagent
# Download the eval dataset (OTTB-dev)
hf download open-thoughts/OpenThoughts-TB-dev --repo-type=dataset
# Download the train dataset
hf download open-thoughts/OpenThoughts-Agent-v1-RL --repo-type=dataset
# cd into the downloaded folder, say /path/to/.cache/huggingface/hub/datasets--open-thoughts--OpenThoughts-Agent-v1-RL/snapshots/hash_code
cd /path/to/.cache/huggingface/hub/datasets--open-thoughts--OpenThoughts-Agent-v1-RL/snapshots/hash_code
python extract_parquet_tasks.py tasks_new.parquet ./extracted_tasks
```

### Launch

Then configure the paths and API keys at the top of the script, and run:

```bash
cd SkyRL/skyrl-train
bash run_otagent.sh
```

The script is designed to run on 8 GPUs single-node. If that is not your setup, modify these configs correspondingly:

```bash
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.ref_num_nodes=1 \
  trainer.placement.policy_num_gpus_per_node=8 \
  trainer.placement.ref_num_gpus_per_node=8 \
  generator.num_inference_engines=8 \
  generator.inference_engine_tensor_parallel_size=1 \
```
