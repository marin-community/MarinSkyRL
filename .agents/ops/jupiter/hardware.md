# Jupiter RL hardware

Each Jupiter booster node exposes four GH200 Grace Hopper superchips:

- four Hopper GPUs with 96 GB HBM3 each;
- four Grace CPUs with 120 GB LPDDR5X each;
- NVLink within the node;
- one NDR200 InfiniBand adapter per superchip for inter-node traffic.

For the MarinSkyRL EP4/FSDP4 geometry, one torchrun agent starts four ranks per node. EP groups stay within a
node, while each FSDP group spans four nodes. A single-node collective test therefore cannot validate the
inter-node FSDP path or a divergence between EP and FSDP phases.

GH200 is `aarch64` and CUDA capability `sm_90`. A policy image used here must expose `linux/arm64` code and
include CUDA architecture 9.0; a GB200-only `sm_100` build is not compatible.
