# Jupiter SIF builds

## Current CUDA 13.0 / GH200 build

The current build has two phases:

1. Run `stage_cuda130_gh200_build.sh` on a Jupiter login node. It resolves the current
   `marin-community/vllm:gpu` and `marin-community/harbor:main` revisions, clones every vLLM CMake dependency, and
   downloads the official aarch64 NCCL 2.31.2 wheel. It does not compile code or create an image.
2. Submit `build_cuda130_gh200.sbatch` with the staged `BUNDLE_DIR`. Compilation, sandbox mutation, SIF creation, and
   GPU acceptance all run on a Slurm compute node.

The build derives from the production r5 SIF. It intentionally preserves Torch 2.11.0+cu130 and the remaining native
stack, compiles only for GH200 (`sm_90`), replaces NCCL with 2.31.2, builds vLLM from the selected `gpu` revision, and
installs Harbor from the selected `main` revision. `validate_cuda130_gh200.py` checks both the writable sandbox and
the fresh SIF before the output is copied into `containers/`.

PyTorch 2.11's published package metadata pins NCCL 2.28.9. No published PyTorch CUDA 13 wheel pins NCCL 2.31.2, so
this image uses NCCL's compatible shared-library ABI and validates the library that `libtorch_cuda.so` actually
loads. It is not a metadata-matched PyTorch/NCCL pair.

Example:

```bash
./stage_cuda130_gh200_build.sh
# Use the BUNDLE_DIR printed by the staging command.
sbatch --export=ALL,BUNDLE_DIR=/e/scratch/jureap59/feuer1/sif_build/cuda130-gh200-<stamp> \
  ./build_cuda130_gh200.sbatch
```

Do not update `../production-runtime.env` from the build job. Promotion requires the multi-node FSDP2, EP, and
Megatron validation matrix.

## Historical archive

This directory records how the Jupiter Torch 2.11, CUDA 13, and vLLM 0.20.2rc0 SIF lineage was built. The
reconstruction notes came from OpenThoughts-Agent commit `c5cccf33d79511172ac910f36385007a123d5aa1` and the original
scripts recovered on Jupiter. The old scripts are not included because several could publish an image after a failed
validation and all performed the build on a login node.

The obsolete vLLM 0.16 HTTP serialization patch was not ported. The r3 image uses vLLM 0.20.2rc0, where routed-expert
serialization is native.

The notes are an archive, not a current release command. Never reconstruct a production SIF from them without a new,
reviewed build and acceptance recipe.

## Reconstruct the r3 base

Start with [recipes/README_vllm0202rc0_r3_sif.md](recipes/README_vllm0202rc0_r3_sif.md). It records the working Torch
2.11 build path and the FlashAttention and Python dependency repairs found during validation.

The superseded Torch 2.9 login and Slurm recipes were excluded. The vLLM source requires the Torch 2.11
`torch/headeronly/util/shim_utils.h` header and cannot compile against the original Torch 2.9 base.

## Later image lineage

The recorded production lineage is:

| Image | Change from predecessor | Reproduction status |
| --- | --- | --- |
| `skyrl_megatron_vllm0202rc0_r3.sif` | Torch 2.11.0+cu130 and vLLM 0.20.2rc0 | Recipe is present. |
| `skyrl_megatron_vllm0202rc0_r4.sif` | py-spy 0.4.2, Torchtitan `a1fdd7e`, DeepEP `1.2.1+73b6ea4`, and NVSHMEM | Provenance recovered. |
| `skyrl_megatron_vllm0202rc0_r5.sif` | FlashAttention 2.8.3 from release `v0.9.22` | Provenance recovered. |

The r4/r5 provenance was recovered from local Claude history. The original scripts and acceptance programs remain at
`/e/scratch/jureap59/feuer1/rebake_r4*.sh`, `rebake_r5.sh`, `r4_accept.py`, and `r5_accept.py`. The r5 acceptance
completed on 2026-08-01 with Torch 2.11.0+cu130, vLLM 0.20.2rc0, and FlashAttention 2.8.3+cu130torch2.11. The current
compute-node recipe replaces those login-node rebakes; the old scripts are retained on Jupiter only as primary
evidence.

The historical CP and routed-expert rebakes replaced Python source inside an editable install. That technique is valid
only when the source delta contains no compiled files or dependency changes; the new recipe always rebuilds vLLM.

## Known runtime constraints

- Run validation with `apptainer exec --pwd /`. Running from a host vLLM checkout can import an empty namespace
  package instead of the image installation.
- The r5 FlashAttention 2.8.3 build returns five values from `unpad_input`; older builds returned four.
- The historical r5 image fails MarinSkyRL's Megatron FlashAttention version guard when `trainer.flash_attn` is
  enabled. Its baked Transformer Engine extension also fails to import against its baked Torch with an undefined
  `c10::SymInt::sym_ne` symbol. The current build preserves those packages, so standalone SIF acceptance does not
  claim Megatron readiness. The production launcher overlay must supply and validate the Megatron native closure.
- The Titan overlay can shadow the Torchtitan package baked into the SIF. Validate the exact production overlay and
  `PYTHONPATH` stack, including the `expert_parallel` symbol MarinSkyRL imports.
- vLLM CUDA extensions were compiled against the image's Torch installation. A Torch or NCCL change requires ABI
  checks for vLLM, Transformer Engine, Apex, FlashAttention, FlashInfer, Torchtitan, and DeepEP.

## Required additions for the next build

Record the following before promoting a new image:

1. The base SIF checksum and labels.
2. The MarinSkyRL, vLLM, Torch, CUDA, NCCL, Transformer Engine, Megatron, FlashAttention, Torchtitan, and DeepEP
   revisions.
3. The complete build command and build log.
4. The output SIF checksum and a package inventory captured from `--pwd /`.
5. Single-GPU import checks and the multi-node FSDP2, EP, and Megatron validation results.
6. The previous image path and rollback command.

Update `../production-runtime.env` only after the new image passes the intended validation matrix.
