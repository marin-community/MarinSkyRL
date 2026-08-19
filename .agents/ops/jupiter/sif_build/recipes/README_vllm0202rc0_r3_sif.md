# SIF recipe: `skyrl_megatron_vllm0202rc0_r3.sif`

Reconstruction recipe for the purged #208/#211 SkyRL + Megatron + TE SIF: **penfever v0.20.2rc0** vLLM with native R3 routed-experts capture and Gemma4/Qwen3-MoE/Qwen3-Next support for Jupiter GH200 (aarch64, CUDA 13.0).

This file and `build_vllm0202rc0_r3_sif.sbatch` reproduce the SIF after scratch purge. The vLLM source is rsynced per the vLLM-sync rule; use the pinned commit below.

## Output
`/e/scratch/jureap59/feuer1/containers/skyrl_megatron_vllm0202rc0_r3.sif`

## Ingredients

| Ingredient | Identity | Notes |
|---|---|---|
| **Base SIF** | `containers/skyrl_megatron.sif` | NGC 25.09 base, **NO vLLM**. torch `2.9.0a0+50eac811a6.nv25.09` (CUDA 13.0, aarch64), pytorch-triton `3.4.0+gitc817b9b6`, Megatron-core `0.14.0`, megatron-bridge `0.1.0rc4`, TE 2.7, apex, `flash_attn 2.7.4.post1`, transformers `5.10.1` (gemma4/gemma4_text/qwen3_next/qwen3_moe all present), SkyRL editable at `/opt/SkyRL` (`penfever/SkyRL@2ab513a6`), nvcc CUDA 13.0. |
| **vLLM fork source** | `/Users/benjaminfeuer/Documents/vllm`, branch `v2-migration`, HEAD **`1948bebd1968688f2eac8f30ecc1e418df7118b5`** (`git describe` = `v0.20.2rc0-305-g1948bebd1`, 2026-05-21) | Built **from source against the in-SIF torch 2.9** via `use_existing_torch.py` (the fork pins `torch==2.11.0` in `requirements/cuda.txt`; we strip that pin to keep the NGC torch the whole Megatron/TE/apex/flash_attn/flashinfer stack is built on). R3 is **native** — `vllm/model_executor/layers/fused_moe/routed_experts_capturer.py` + `routed_experts` emission in all four `entrypoints/openai/{chat_completion,completion}/{serving,protocol}.py` files (capture rail landed via PR #39917, 2026-05-07). gemma4 models present (`gemma4.py`, `gemma4_mm.py`); registry has `Gemma4ForConditionalGeneration`, `Gemma4ForCausalLM`, `Qwen3MoeForCausalLM`, `Qwen3NextForCausalLM`. **NO separate R3 patch is applied** — unlike the 0.16 SIF which needed `vllm_routed_experts_http_serialization.patch` via `vllm_http_overlay.img`, the 0.20.2rc0 fork carries it natively. |
| **GDN overlay** | `containers/fla_tilelang_overlay.img` | tilelang `0.1.8`, FlashQLA (`flash_qla 0.1.0+6ef4858`, QwenLM git `6ef4858`), apache-tvm-ffi `0.1.9`, fla `0.5.0` (masked/broken; FlashQLA is self-contained). Merged via `debugfs rdump /upper` (no fuse, no root), same as `bake_r3_sif.sbatch`. Provides fused GatedDeltaNet fwd+bwd for Qwen3-Next (Stage-8 validated, 4.8–27× speedup). |

## QUEUED for the next rebuild — do NOT build yet

Two additions requested 2026-07-31. **Gated: hold until the in-flight MarinSkyRL fixes land**, so the
rebuild carries them and the torchtitan/MoE work in one pass rather than two. Nothing below has been
added to `build_vllm0202rc0_r3_sif.sbatch`; this section is the specification for when it is.

### 1. `py-spy` — trivial, add unconditionally

```bash
pip install py-spy        # 0.4.1; pure binary wheel, aarch64 available, no CUDA coupling
```

**Why, given it already works from the host.** `py-spy` installed in the `otagent` conda attaches
straight through apptainer (shared PID namespace) and is what root-caused the 2026-07-31 MoE
`_token_dispatch` deadlock. In-image is still worth it: it removes the `srun --overlap -w <node>`
dance, works when the host env is unavailable, and makes the deep-dive skill's "capture the live
py-spy before a wedge KILL" step a one-liner. Cost is ~6 MB.

### 2. DeepEP — enables `ep_comm_backend: "deepep"`, the alternative to the deadlocking torch all-to-all

Wanted because the torchtitan `_token_dispatch` all-to-all deadlocked a live MoE run (see
`ENVIRONMENT_MAP.md` §3c and `agent_logs/2026-07-31_escalation-tasktrove-shaped-arm-async-deadlock.md`).
`skyrl_train/distributed/deepep.py` is already written against it and needs
`deep_ep.Buffer` + `deep_ep.utils.{EventHandle, EventOverlap}`.

| Ingredient | Identity | Status as of 2026-07-31 |
|---|---|---|
| DeepEP source | `github.com/deepseek-ai/DeepEP` @ **`73b6ea4`** ("support hidden-dim 3072", #458) | **PIN THIS.** `skyrl_train/distributed/deepep.py` names `1.2.1+73b6ea4`. A `--depth 1` clone gives HEAD (`dd758ca`), which **does not compile against torch 2.11** — `no matching function for call to 'empty(…, c10::TensorOptions)'`, `launch_engram_fetch_wait declared void`. Fetch with depth ≥ 200. |
| NVSHMEM SDK | `nvidia-nvshmem-cu13` **3.7.2** (aarch64 manylinux wheel) | **Verified obtainable.** Ships headers, `libnvshmem_device.a`, and `libnvshmem_device_sm_90.bc` (GH200's arch). Its `libnvshmem_host.so.3` soname matches the runtime already in the SIF at `/usr/local/cuda/targets/sbsa-linux/lib`. DeepEP's `setup.py` supports this route natively (`find_nvshmem_root()`, links `-l:libnvshmem_host.so`). |
| CCCL / libcu++ headers | `cuda/std/*` | **RESOLVED 2026-08-01 — CCCL `v3.0.1` from GitHub, on `CPATH`.** The image's CUDA 13.0.88 has its CCCL *cmake configs* (`lib64/cmake/{cccl,libcudacxx,cub,thrust}`) but its **headers stripped** — `/usr/local/cuda/include/cuda/` does not exist. NVSHMEM's `nvshmemi_common_device.cuh:9` includes `<cuda/std/array>`, hence the failure. The version is **declared, not guessed**: `libcudacxx-config-version.cmake` in the image sets `MAJOR 3 / MINOR 0 / PATCH 1`. Clone `github.com/NVIDIA/cccl` at `v3.0.1` (`f19d875da`) and export `CPATH=<cccl>/libcudacxx/include:…`. Confirms `nvidia-cuda-cccl-cu13` is a dead end — it is a 0.0.1 stub sdist. |
| NVSHMEM unversioned soname | `libnvshmem_host.so` | **Second blocker, behind the first.** With CCCL supplied, all 7 CUDA objects and the device link compile clean, then `ld: cannot find -l:libnvshmem_host.so`. The wheel ships only **`libnvshmem_host.so.3`**; DeepEP's `setup.py` links `-l:libnvshmem_host.so`, which requires the exact unversioned name. Fix: `ln -sfn libnvshmem_host.so.3 <nvshmem>/lib/libnvshmem_host.so`. |
| Build flags | `TORCH_CUDA_ARCH_LIST=9.0`, `MAX_JOBS=16`, `NVSHMEM_DIR=<nvshmem wheel root>` | Same arch convention as the vLLM step (`9.0+PTX` there). |

**Prerequisites already in the SIF** (so the gap is only CCCL): nvcc **CUDA 13.0** with `compute_90`,
torch **2.11.0+cu130**, NVSHMEM **runtime** libs + ibrc/libfabric/ucx transports, setuptools +
`torch.utils.cpp_extension`.

Work in progress at `/e/scratch/jureap59/feuer1/deepep_smoke/` — `DeepEP/` (at `73b6ea4`), `build.sh`,
`build_full.log` (HEAD failure), `build_73b6ea4.log` (pinned attempt, CCCL failure). **Reproduce the
build green there before touching the sbatch**; DeepEP is an optional perf backend and must not be
allowed to break a rebuild of the whole SIF.

### 3. torchtitan — bake the version MarinSkyRL actually pins, resolved at build time

The SIF ships **0.2.2**. MarinSkyRL `main` has been pinning **a1fdd7e (0.1.0)** and imports
`from torchtitan.distributed.expert_parallel import expert_parallel`, a symbol 0.2.2 removed — so every
MoE arm currently depends on the `sif_pydeps_titan_a1fdd7e` PYTHONPATH prefix shadowing the image
*down*. Baking the right version removes that shadow and the whole class of failure with it.

**Do NOT hardcode a1fdd7e.** The repo history has crossed this pin twice (`cdacec77` aligned to 0.2.2,
`4ba60a1f` reverted to a1fdd7e). Resolve it from the MarinSkyRL revision being baked, at build time,
and bake that. See §3a of `ENVIRONMENT_MAP.md` for the failure signature if they disagree: the arm dies
~10 min in, *after* weights load, at `model_wrapper.py:639 → moe_swap → moe.py:57`. The image's own
build assert does not catch it, because the class it imports exists in both versions.

**Do not drop the pydeps prefixes until the baked version is verified**, and note that
`sif_pydeps_longctx_titan022` shadows the image *up* to 0.2.2 for the longctx/CP configs — baking
0.1.0 leaves those configs depending on their prefix, so removing prefixes is a per-config decision,
not a global one. Sequence: bake → verify in-container → drop `sif_pydeps_titan_a1fdd7e` from
`recipe.py` → relaunch.

**Acceptance for all three** — add to the §Acceptance list when they go in:

```bash
apptainer exec --nv <sif> py-spy --version
apptainer exec --nv <sif> python -c "from deep_ep import Buffer; from deep_ep.utils import EventHandle, EventOverlap; print('deep_ep OK', Buffer)"
# torchtitan: assert the SYMBOL, not the version string -- the version is not the thing that breaks
apptainer exec --nv <sif> python -c "import torchtitan; from torchtitan.distributed.expert_parallel import expert_parallel; print('titan OK', torchtitan.__version__)"
```

The second is the exact import `skyrl_train/distributed/deepep.py` performs. Do **not** assert it by
grepping build output for a success string — a `set -x` trace echoes the command containing that string
and yields a false pass.

## Why build from source (not a prebuilt wheel)
`/e/data1/datasets/playground/ot-baf/wheels/` has `vllm-0.20.2+cu130torch2.11-cp312-...whl`, but it is non-rc0 and torch 2.11, incompatible with SIF torch 2.9/Megatron-core/TE/apex/flash_attn/flashinfer. Compile against in-SIF NGC torch 2.9.

## Build steps (encoded in `build_vllm0202rc0_r3_sif.sbatch`)
1. `apptainer build --sandbox` from `skyrl_megatron.sif`.
2. Stage fork source into `$SANDBOX/opt/vllm_build`.
3. Inside sandbox (`apptainer exec --writable`): `python use_existing_torch.py`
   (strips torch pins) → `pip install --no-build-isolation -v -e .` (compiles
   CUDA/C++ kernels with the SIF's nvcc; `TORCH_CUDA_ARCH_LIST=9.0+PTX` for GH200,
   `MAX_JOBS=48`). Editable install — source stays at `/opt/vllm_build`.
4. Merge `fla_tilelang_overlay.img` via `debugfs rdump`.
5. `apptainer build` the final SIF.
6. Validate (see below).

## How to run (on Jupiter)
```bash
# 1) rsync the fork source to the cluster (per project vLLM-sync rule).
#    Pin the exact commit first:
cd /Users/benjaminfeuer/Documents/vllm
git checkout v2-migration   # HEAD must be 1948bebd1...
git rev-parse HEAD > .vllm_commit
rsync -az --delete \
  --exclude '.git' --exclude 'build' --exclude '*.so' --exclude '.deps' \
  -e "ssh -i ~/.ssh/id_ed25519_jsc -o AddressFamily=inet" \
  /Users/benjaminfeuer/Documents/vllm/ \
  feuer1@login02.jupiter.fz-juelich.de:/e/scratch/jureap59/feuer1/sif_build/vllm_src/

# 2) place this sbatch on the cluster and submit:
ssh Jupiter 'mkdir -p /e/scratch/jureap59/feuer1/sif_build/logs'
scp sif_build/recipes/build_vllm0202rc0_r3_sif.sbatch \
  Jupiter:/e/scratch/jureap59/feuer1/sif_build/
ssh Jupiter 'sbatch /e/scratch/jureap59/feuer1/sif_build/build_vllm0202rc0_r3_sif.sbatch'
```

## Acceptance / validation (asserted in step 5 of the sbatch)
- `vllm.__version__` == `0.20.2rc0` (dev tree may render `0.20.2rc0.devN+g<sha>`).
- `ModelRegistry.get_supported_archs()` contains `Gemma4ForConditionalGeneration`,
  `Gemma4ForCausalLM`, `Qwen3MoeForCausalLM`, `Qwen3NextForCausalLM`.
- `routed_experts` present in the four chat/completion serving+protocol files and
  `routed_experts_capturer.py` exists (native R3 capture).
- transformers `gemma4`/`gemma4_text` in `CONFIG_MAPPING_NAMES` (≥ 5.10.1).
- `import skyrl_train, skyrl_gym` OK (training stack intact).
- GDN overlay: `tilelang 0.1.8`, `apache-tvm-ffi 0.1.9`, `flash_qla 0.1.0+6ef4858`.

## Notes / gotchas
- The base SIF already has nvcc (CUDA 13.0) and the full training stack; only vLLM
  and the GDN overlay are added.
- `use_existing_torch.py` MUST run before `pip install` or pip will try to pull
  torch 2.11 and break the stack.
- Build scratch lives on GPFS (`/e/scratch/.../sif_build/job_<id>`); node-local
  `/tmp` (96G) is too small for sandbox + extracts + SIF.
- vLLM kernel compile is the long pole (~1–2h on 48 cores); 5h wall is generous.
- Do NOT perturb triton/torch/flash_attn/flashinfer in the sandbox — vLLM is built
  against the exact in-SIF versions.
