# Jupiter SIF build archive

This directory records how the Jupiter Torch 2.11, CUDA 13, and vLLM 0.20.2rc0 SIF lineage was built. The files were
ported from OpenThoughts-Agent commit `c5cccf33d79511172ac910f36385007a123d5aa1`. The port fixes one unmatched shell
quote in `rebake_vllm0202rc0_r3_cp.sh`; the remaining historical commands and validation checks are unchanged.

The obsolete vLLM 0.16 HTTP serialization patch was not ported. The r3 image uses vLLM 0.20.2rc0, where routed-expert
serialization is native.

The scripts are a build archive, not a current release command. They contain old source revisions, image names, and
Jupiter paths. Confirm every input against the intended MarinSkyRL revision and write a new output filename before
running a recipe. Never overwrite a production SIF in place.

## Reconstruct the r3 base

Start with [recipes/README_vllm0202rc0_r3_sif.md](recipes/README_vllm0202rc0_r3_sif.md). The archive contains three
attempts that document the transition from the original NGC Torch 2.9 base to the Torch 2.11 image:

- `build_vllm0202rc0_r3_sif.sbatch` is the original scheduled build against the NGC base.
- `build_vllm0202rc0_r3_login.sh` records the login-node rebuild path.
- `build_vllm0202rc0_r3_login_torch211.sh` upgrades the sandbox to Torch 2.11.0+cu130 before compiling vLLM.
- `fix_vllm0202rc0_r3_torch211_deps.sh` repairs the FlashAttention and Python dependency mismatches found during
  validation.

The README records the base image, vLLM revision, overlays, architecture flags, expected package versions, staging
procedure, and acceptance checks.

## Later image lineage

The recorded production lineage is:

| Image | Change from predecessor | Reproduction status |
| --- | --- | --- |
| `skyrl_megatron_vllm0202rc0_r3.sif` | Torch 2.11.0+cu130 and vLLM 0.20.2rc0 | Recipe is present. |
| `skyrl_megatron_vllm0202rc0_r4.sif` | py-spy 0.4.2, Torchtitan `a1fdd7e`, DeepEP `1.2.1+73b6ea4`, and NVSHMEM | The rebake script is missing from this archive. |
| `skyrl_megatron_vllm0202rc0_r5.sif` | FlashAttention 2.8.3 from release `v0.9.22` | The rebake script is missing from this archive. |

The missing scripts were named `rebake_r4*.sh` and `rebake_r5.sh` and were last known to exist on Jupiter. The r4 and
r5 images contain `/opt/r4_accept.py` and `/opt/r5_accept.py`. Recover the scripts and acceptance programs before
changing the production lineage.

The archive also contains the CP and routed-expert surgical rebakes used before r4:

- `rebake_vllm0202rc0_r3_cp.sh`
- `rebake_237_cp_fixb3.sh`
- `rebake_cp_fixb4_r3_rayhook.sh`
- `resume_cp_rebake_from_sandbox.sh`

These rebakes replace Python source inside an editable install. They are valid only when their stated source delta
contains no compiled files or dependency changes.

## Known runtime constraints

- Run validation with `apptainer exec --pwd /`. Running from a host vLLM checkout can import an empty namespace
  package instead of the image installation.
- The r5 FlashAttention 2.8.3 build returns five values from `unpad_input`; older builds returned four.
- The historical r5 image fails MarinSkyRL's Megatron FlashAttention version guard when `trainer.flash_attn` is
  enabled. Do not treat its FSDP2 validation as Megatron validation.
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
