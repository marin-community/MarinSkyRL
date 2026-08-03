# FSDP2 CPU-offload NUMA contract

`fsdp2_cpu_offload_placement.py` starts the production Ray policy-worker path on one four-GPU GH200 node,
materializes a small FSDP2 model with native CPU offload, and queries the physical NUMA node backing sampled
pinned parameter pages.

The test requires all of the following on every rank:

- the effective memory policy is `bind` over every CPU-bearing NUMA node and excludes HBM-only nodes;
- FSDP2 creates pinned CPU parameter storage;
- no sampled parameter page resides in an HBM-only node;
- at least 95 percent of sampled pages reside on the CPU node paired with that rank's GPU.

Run the image's installed Python and dependencies. Do not resolve a new `uv` environment on the offline compute
node. The model must already be present in the bound Hugging Face cache.

```bash
srun --nodes=1 --ntasks=1 --gres=gpu:4 --cpu-bind=none --gpu-bind=none \
  apptainer exec --nv --pwd / \
  --bind "$SKYRL_HOME:$SKYRL_HOME" \
  --env "PYTHONPATH=$SKYRL_HOME/skyrl-train" \
  "$RL_CONTAINER" \
  python "$SKYRL_HOME/skyrl-train/tests/gpu/numa/fsdp2_cpu_offload_placement.py"
```

Record the Jupiter node, image, MarinSkyRL commit, command, and complete result with each run.
