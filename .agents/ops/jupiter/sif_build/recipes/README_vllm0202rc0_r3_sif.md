# Historical SIF inputs: `skyrl_megatron_vllm0202rc0_r3.sif`

The historical login-node script upgraded the NGC 25.09 Torch 2.9 installation to Torch 2.11.0+cu130, built the vLLM
fork against that Torch installation, merged the FlashQLA overlay, and validated the result. It is not retained as an
executable recipe. Use the compute-node build in the parent directory instead.

## Recorded inputs

| Input | Recorded identity |
| --- | --- |
| Base SIF | `containers/skyrl_megatron.sif`, NGC 25.09, CUDA 13.0, aarch64 |
| vLLM source | `v2-migration` at `1948bebd1968688f2eac8f30ecc1e418df7118b5`, described as `v0.20.2rc0-305-g1948bebd1` |
| CUTLASS | v4.4.2 staged at `vllm/.deps/cutlass-src` |
| Torch | 2.11.0 from the CUDA 13.0 PyTorch index |
| FlashAttention 3 | `3.0.0+cu130torch2.11gite2743ab`, prebuilt aarch64 wheel |
| GDN overlay | `containers/fla_tilelang_overlay.img`: tilelang 0.1.8, FlashQLA `0.1.0+6ef4858`, and apache-tvm-ffi 0.1.9 |
| GPU architecture | `TORCH_CUDA_ARCH_LIST=9.0+PTX` for GH200 |

The production image was later inventoried with a different vLLM revision, `3e3a1c45d`. Inspect
`/opt/vllm_build/.vllm_commit` in the source SIF before deriving another image. Do not assume the archived source
revision is the revision in r4 or r5.

## Historical procedure

The old workflow staged the base SIF, overlay, source trees, CUTLASS checkout, and wheelhouses on GPFS. It built a
sandbox under login-local `/tmp`, wrote the final SIF to GPFS, and ran acceptance from `/` so a host vLLM checkout
could not shadow the image package. This procedure is recorded to explain r3; do not repeat it on a login node.

The Torch 2.9 build path is not retained. The vLLM source includes
`torch/headeronly/util/shim_utils.h`, which is absent from the original NGC Torch 2.9 headers.

## r4 reconstruction inputs

The recovered r4 rebake script added these packages to r3:

| Package | Recorded identity and build constraint |
| --- | --- |
| py-spy | 0.4.2 |
| Torchtitan | `a1fdd7e` for the historical image; resolve the required revision from the MarinSkyRL source being built |
| DeepEP | `1.2.1+73b6ea4`; do not build DeepEP HEAD against Torch 2.11 |
| NVSHMEM | `nvidia-nvshmem-cu13==3.7.2`; add `libnvshmem_host.so -> libnvshmem_host.so.3` in the wheel library directory |
| CCCL | v3.0.1 (`f19d875da`), matching the CUDA 13.0.88 CMake metadata in the image |

DeepEP uses `TORCH_CUDA_ARCH_LIST=9.0`, `MAX_JOBS=16`, and `NVSHMEM_DIR` pointing at the NVSHMEM wheel root. Put the
CCCL libcudacxx headers on `CPATH`; the NGC image contains CCCL CMake metadata but not `cuda/std/*` headers.

The r4 acceptance checks must import the exact interfaces MarinSkyRL uses:

```bash
apptainer exec --pwd / --nv <sif> py-spy --version
apptainer exec --pwd / --nv <sif> python -c \
  "from deep_ep import Buffer; from deep_ep.utils import EventHandle, EventOverlap"
apptainer exec --pwd / --nv <sif> python -c \
  "from torchtitan.distributed.expert_parallel import expert_parallel"
```

Test Torchtitan with the production overlay and `PYTHONPATH` stack. An overlay can shadow the package baked into the
SIF, and package metadata can report a different version from the imported module.

## r5 reconstruction input

The recovered r5 rebake script replaced FlashAttention 2.6.3 with the aarch64 wheel
`flash_attn-2.8.3+cu130torch2.11-cp312-cp312-manylinux_2_34_aarch64.whl` from
`mjun0812/flash-attention-prebuild-wheels` release `v0.9.22`. Install it with `--no-deps` so pip does not replace the
Torch and CUDA stack.

Run `/opt/r4_accept.py` and `/opt/r5_accept.py` from the corresponding images after those programs are recovered.
The r5 image was validated for FSDP2, but MarinSkyRL's historical Megatron FlashAttention version guard rejects
FlashAttention 2.8.3. A replacement image needs separate FSDP2 and Megatron validation.

## Acceptance

The r3 build must fail unless all of these checks pass:

- Torch reports 2.11.0 with CUDA 13.0 and sees a GPU.
- vLLM imports from `/opt/vllm_build` and exposes the expected model architectures.
- Transformer Engine, Apex, Megatron Core, FlashAttention, SkyRL, and the GDN packages import against the upgraded
  Torch installation.
- The native routed-expert capture source and OpenAI response fields are present.
- All staged vLLM runtime dependencies import.
- A cached tiny model loads and generates when a suitable model is present on the build host.

The SIF may be written before validation runs. A nonzero validation result makes the artifact ineligible for
promotion; remove it or retain it under a clearly failed name for debugging.
