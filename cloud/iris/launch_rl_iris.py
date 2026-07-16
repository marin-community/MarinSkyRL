#!/usr/bin/env python3
"""Launch a MarinSkyRL RL training job on Marin's Iris GPU cluster (CoreWeave).

This is the GPU/Iris analog of ``rl/cloud/launch_rl_cloud.py`` (the SkyPilot RL
launcher). It combines:
  - the RL-job structure from ``launch_rl_cloud.py`` (gpu-rl venv, run_rl.py
    entrypoint, rl_config / model_path / train_data / overrides), and
  - the Iris SDK submission mechanics from ``eval/cloud/launch_eval_iris.py``
    (controller tunnel, IrisClient.submit, --secrets-env injection, --no-wait,
    job-name, max-retries, workspace source-sync to /app).

The target is GPU (not TPU), and the gpu-rl image is a conda-venv image
(/opt/openthoughts/envs/rl), so this launcher drives the iris SDK's GPU helpers
(build_resources(gpu=...), gpu_device, the leafgroup-coscheduling
``resolve_multinode_defaults``) directly rather than going through a TPU-shaped
base launcher.

Multi-node / gang scheduling
----------------------------
Iris HAS a native gang mechanism for GPUs (verified via `iris job run --help`
and lib/iris/src/iris/cli/job.py):
  - ``--gpu H100x8`` requests a whole CoreWeave node (8 H100 + IB) per task.
  - ``--replicas N`` (the `--help` text: "Number of tasks for gang scheduling")
    requests N such tasks.
  - For GPUs with replicas>1, ``resolve_multinode_defaults`` returns
    ``CoschedulingConfig(group_by="leafgroup")`` — the H100/InfiniBand
    colocation level — so all N replicas are co-scheduled together on the same
    IB leaf fabric, all-or-nothing.
  - The cw-us-east-02a cluster config enables **Kueue gang admission**
    (``kueue.cluster_queue: iris-cq``, ``host_network: true`` for NCCL/IB), so
    the N-task gang is admitted atomically: either all N whole nodes are
    granted or the job queues — true exclusive, co-scheduled multi-node.

So this launcher requests ``--num-nodes N`` whole H100x8 nodes EXCLUSIVELY: one
iris task per node (``replicas=N``), each holding all 8 GPUs of its node (no
co-tenants), coscheduled by leafgroup. The RL topology (one cross-node Ray
cluster, NCCL over IB) is wired by an in-container controller
(``cloud/iris/start_rl_iris_controller.py``): rank 0 starts the Ray head and
publishes its IP to a shared rendezvous; ranks 1..N-1 join; then rank 0 runs the
MarinSkyRL driver (``cloud.iris.run_rl --num_nodes N``) attached to that cluster.

Usage
-----
    set -a; source "${DC_AGENT_SECRET_ENV:?see .claude/secret.md}"; set +a

    python -m cloud.iris.launch_rl_iris \
        --rl_config cloud/iris/configs/<config>.yaml \
        --model_path Qwen/Qwen3-8B \
        --train_data '["mlfoundations-dev/dataset"]' \
        --num-nodes 4 \
        --job-name my-rl-iris-run \
        --no-wait
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import sys
import time
from pathlib import Path
from typing import List, Optional

from cloud.iris.paths import PROJECT_ROOT
from cloud.iris.secrets_env import load_secrets_env_into_os_environ

# Defaults for the CoreWeave H100 GPU cluster.
DEFAULT_CLUSTER = "cw-us-east-02a"
# Pin the RL image by IMMUTABLE DIGEST, not the floating ``:gpu-rl`` tag.
#
# WHY (the floating-tag stale-cache trap): the iris k8s backend always stamps
# the task pod with ``imagePullPolicy: IfNotPresent``
# (marin lib/iris .../backends/k8s/tasks.py) and we cannot override it from here.
# With a FLOATING tag, IfNotPresent means a node that already has *some* image
# under that tag name will NOT re-pull when the tag is later retagged to new
# bytes — so a node that cached an OLD ``:gpu-rl`` keeps running stale code
# (observed: a launcher run executed MarinSkyRL 4c668f4 with NO flash_attn_2_cuda
# while the freshly-retagged ``:gpu-rl`` pointed at the good build).
#
# A content-addressed ``@sha256:`` reference is self-verifying: IfNotPresent only
# treats the cache as a hit when the cached bytes hash to exactly this digest, so
# it always runs the intended image regardless of node cache state — sidestepping
# the stale-tag problem entirely without needing imagePullPolicy: Always.
#
# This digest == the immutable gitsha tag ``:gpu-rl-44c06ea8`` (OT-Agent commit
# 44c06ea8, "bump gpu-rl SKYRL_COMMIT 2d9feef -> 78d83a5"): flash_attn 2.8.3 +
# flash_attn_2_cuda present, /opt/skyrl baked at MarinSkyRL 78d83a5 — which ADDS
# the two fixes that deterministically crashed CoreWeave RL at build_models:
# 518179d (default norm_topk_prob=True for Qwen3.5/3.6 MoE) + 0b2b05b (retry around
# rank-0 HF weight-index resolution). Also still includes 2d9feef's trials_dir
# raw-str fix; harbor BAKED at 342729d5 (reward-zeroing trial.paths.trial_dir fix).
# When the gpu-rl image is rebuilt, bump this digest (use the immutable
# ``:gpu-rl-<gitsha>`` tag's digest, never the floating ``:gpu-rl``).
#
# This digest BAKES torchtitan a1fdd7e (+ tyro): the `ExpertParallel` import-assert
# (step-4a of Dockerfile.gpu-rl) PASSED in-build → the EP>1 MoE unblock is proven.
# The CoreWeave EP=8 RL jobs (30B-A3B 131k, 35B) no longer hit
# `ModuleNotFoundError: torchtitan`. Also baked: vLLM-fork 76259c63 + flash-attn
# 2.8.3 (flash_attn_2_cuda present) + MarinSkyRL 39faff7d + harbor 342729d5.
# MarinSkyRL 39faff7d (bumped from 78d83a5) carries the VALIDATED MoE forward-spill
# fix + deterministic-dtype hardening. In-build asserts ran green: flash_attn_2_cuda
# OK (from cached wheel), torch 2.11.0+cu128 / vllm 0.1.dev16611+g76259c63a /
# skyrl_train import OK, torchtitan ExpertParallel import OK, baked MarinSkyRL HEAD
# == 39faff7d.
#
# BUILT IN-CLUSTER ON COREWEAVE (not the arm64 Mac): the image is amd64 + a
# from-source x86 CUDA build QEMU/Docker-Desktop can't do locally, and iris has NO
# in-cluster build primitive (`iris build` = LOCAL buildx). The build ran as an iris
# job with KANIKO (BuildKit needs CAP_SYS_ADMIN/bind-mounts the cluster denies —
# privileged is silently downgraded; nodes run gVisor). Context = the iris-synced
# /app bundle (cpu48/mem512GB/disk400GB). FAST no-nvcc PREBUILT-WHEELHOUSE path: the
# kaniko script (docker/build_gpu_rl_kaniko.sh) fetched the prebuilt vLLM-fork +
# flash-attn wheels (from laion/gpu-rl-build-wheels) into the context and ran with
# WHEEL_SOURCE=prebuilt-wheelhouse + --skip-unused-stages → ZERO nvcc (~minutes, not
# ~3h); the SKYRL_COMMIT-only bump did not change the wheel cache-key, so the wheels
# stayed ABI-correct. ghcr push via the GitHub PAT (`gh auth token`, write:packages),
# NOT the Docker-Hub DOCKER_TOKEN in secrets.env.
#
# Single-platform linux/amd64 manifest, 13 layers ~21.5 GB. The floating :gpu-rl
# tag resolves to the same digest. When the image is rebuilt, bump this digest
# (use the immutable :gpu-rl-<gitsha> tag's digest, never the floating :gpu-rl).
DEFAULT_RL_DOCKER_IMAGE = (
    "ghcr.io/open-thoughts/openthoughts-agent"
    # gpu-rl-efd77b98 (built 2026-07-03, kaniko job gpurl-kaniko-efd77b98): the PULLABLE re-layering of
    # gpu-rl-69634c0b (@sha256:d9c7e604…, harbor 0729a3e9 = poll fix + tmux-bake). Same baked contents
    # (harbor 0729a3e9, MarinSkyRL 39faff7d, vLLM-fork 76259c63, flash-attn 2.8.3, torch 2.11.0+cu128,
    # rl env pinned via rl_env_constraints.txt) but built with SINGLE_SNAPSHOT=0 (per-instruction layers)
    # + torch's nvidia-CUDA deps split into 3 pre-install RUNs, so the MAX layer is 3.46 GB (was one
    # 16.6 GB --single-snapshot layer). WHY: the 16.6 GB single layer CANNOT be pulled over the
    # CoreWeave→ghcr egress — containerd restarts the single-blob GET from 0 and it dies at 8-11 GB every
    # attempt (diagnosed restart-from-0 across all 8 r4 pods → ImagePullBackOff; the incremental-base
    # rebuild ALSO failed because the build pod had to pull the same 16.6 GB base). 48 small layers each
    # pull+retry independently. Build asserts green (baked harbor 0.8.0 @ 0729a3e9). Digest below.
    # gpu-rl-a003838c (built 2026-07-03, kaniko job gpurl-kaniko-a003838c): HARBOR_COMMIT-only bump →
    # harbor 9416d5f3 "default-OFF episode logging" (descends from a4957ef1, so keeps 1ms poll + tmux-bake
    # + persistent exec session, AND gates the 3 big synchronous per-turn S3 writes — debug.json + prompt
    # + response — behind default-OFF enable_episode_logging, the real throughput lever py-spy found: a
    # sync S3 write per LLM call blocking the shared asyncio loop). Same PULLABLE recipe (SINGLE_SNAPSHOT=0
    # + torch nvidia-CUDA split, max layer 3.46 GB, 48 layers). Everything else unchanged (MarinSkyRL
    # 39faff7d baked, vLLM-fork 76259c63, flash-attn 2.8.3, torch 2.11.0+cu128).
    # gpu-rl-722fec34 (built 2026-07-04, kaniko gpurl-kaniko-722fec34): harbor 2e42d312 (cheap reaper
    # DEFAULT-ON — fixes the n=384 O(N) coordinator-reaper bottleneck py-spy found; opt out via
    # HARBOR_CHEAP_REAPER=0) + skyrl 3caeb79f (TIS served-id splice, now baked-default — no --skyrl-ref
    # needed for 35B). Same PULLABLE recipe; vLLM-fork 76259c63, flash-attn 2.8.3, torch 2.11.0+cu128 unchanged.
    # gpu-rl-dc56d265 (built 2026-07-05, kaniko gpurl-kaniko-dc56d265): harbor 2e42d312 (unchanged, cheap
    # reaper DEFAULT-ON) + skyrl 7b7d627b (load-aware power-of-two-choices inference-engine routing — fixes
    # the sticky hash-at-birth session routing that pinned agentic-RL rollout load onto one vLLM engine
    # while siblings idled at n=384; preserves per-session stickiness for prefix-cache reuse). Same PULLABLE
    # recipe (SINGLE_SNAPSHOT=0, max layer <8 GB; pull-verified 4m0s, 22.5 GB, skyrl HEAD=7b7d627b in-pod).
    # vLLM-fork 76259c63, flash-attn 2.8.3, torch 2.11.0+cu128 unchanged.
    # gpu-rl-bd888d27 (built 2026-07-05, kaniko gpurl-kaniko-bd888d27): HARBOR_COMMIT-only bump → harbor
    # d58043c3, TWO Daytona fixes: (1) connection-pool cap 250→2048 (fleet knob
    # HARBOR_DAYTONA_CONNECTION_POOL_MAXSIZE) — the SDK's 250-connection aiohttp pool starved the verifier's
    # upload/exec/download round-trip for a socket at n_concurrent>>250 → 100% VerifierTimeoutError on the slow
    # 35B (verifier isn't slow; its HTTP calls can't get a connection). 2048 lets the grid run clean at n=768.
    # (2) auto_stop_interval_mins 0→5 — killed-job orphaned sandboxes idle-stop then auto-delete (self-clean)
    # instead of leaking forever. Same PULLABLE recipe; skyrl 7b7d627b, vLLM-fork 76259c63, flash-attn 2.8.3,
    # torch 2.11.0+cu128 unchanged (harbor-only bump — wheels + rl_env_constraints untouched).
    # gpu-rl-2712998d (built 2026-07-05, kaniko gpurl-kaniko-2712998d): PULLABLE re-layer of bd888d27 —
    # bd888d27 (@sha256:a8f76d48…) baked identical contents but with --single-snapshot → one 16.6 GB layer that
    # EOFs on the CoreWeave→ghcr pull (ImagePullBackOff + whiteout conflict). This is SINGLE_SNAPSHOT=0
    # (48 layers, max 3.5 GB), same baked harbor d58043c3 + wheels. Verified pullable; pods reached Running.
    # gpu-rl-4e505a4e (built 2026-07-06, kaniko gpurl-kaniko-4e505a4e, SINGLE_SNAPSHOT=0 pullable): SKYRL_COMMIT
    # bump 7b7d627b→cdca0b3a = EPDIAG per-phase fwd/modelfwd instrumentation (LOGGING-ONLY, EPDIAG-gated, no
    # routing/correctness change) for the EP16-vs-EP8 fwd-op diagnostic (FINDING #2). Strict superset of 861656ba
    # (instrumentation is a no-op when EPDIAG unset). wheels + harbor d58043c3 + rl_env_constraints unchanged.
    # gpu-rl-f9806065 (built 2026-07-06, kaniko gpurl-kaniko-f9806065, SINGLE_SNAPSHOT=0 pullable): SKYRL_COMMIT
    # bump cdca0b3a→b2ff8bf2 = abort_generation DRAIN fix — drains the vLLM engine (poll has_unfinished_requests
    # until idle, bounded 60s fail-loud) before the caller meta-izes params in the layerwise weight-sync reload.
    # Fixes the _C::rms_norm meta-tensor crash on eager decode (grid-30b-c) + the masked stale-weight read under
    # cudagraph replay (BOTH 35B rungs died at gs1's weight sync on 84ffafac). wheels + harbor d58043c3 +
    # rl_env_constraints unchanged (skyrl-only, prebuilt-wheelhouse).
    # NOTE: gpu-rl-f9806065 @sha256:37cdc3e6 was UN-PULLABLE (built --single-snapshot by DEFAULT = one 16 GB
    # layer -> ImagePullBackOff on CoreWeave). gpu-rl-addb348e below is the SINGLE_SNAPSHOT=0 re-layer (48
    # layers, max 3.5 GB, pull-verified) with IDENTICAL contents (SKYRL b2ff8bf2 drain fix).
    # gpu-rl-cf1ecea6 (built 2026-07-07, kaniko gpurl-kaniko-cf1ecea6, SINGLE_SNAPSHOT=0 pullable):
    # SKYRL_COMMIT bump 613e225d->822221a0 = engine-readiness gate in ray_wrapped_inference_engine.py
    # (already-validated; previously runtime-only via --skyrl-ref, now baked-default). Parent is exactly
    # 613e225d, so this ADDS one commit and preserves everything baked in gpu-rl-1a32669c. wheels + harbor
    # + rl_env_constraints UNCHANGED (skyrl-only, prebuilt-wheelhouse). Pull-verified: 48 layers, max 3.46
    # GB, 22.6 GB total. Build asserts green (skyrl_train/vllm/flash_attn/torchtitan.ExpertParallel import).
    # gpu-rl-7d15b25a (built 2026-07-08, kaniko gpurl-kaniko-7d15b25a, SINGLE_SNAPSHOT=0 pullable): the
    # InfiniBand ENABLE image. Adds `rdma-core ibverbs-providers libibverbs1 librdmacm1 ibverbs-utils` to the
    # rl-stage apt-get so NCCL's built-in IB transport can DLOPEN libibverbs.so.1 + the libmlx5 provider at
    # runtime — WITHOUT it (all prior images) NCCL silently disabled IB and fell back to NET/Socket (TCP over
    # enp157s0np0), the cross-node throughput bottleneck. Diagnosed in a live grid-30b-c-cp4-timing-v5 pod:
    # CoreWeave exposes the RDMA devices (/dev/infiniband/{uverbs0..8,rdma_cm}, 9x mlx5 ports ACTIVE @ 100Gb/s
    # EDR) but the image shipped NO verbs userspace (`find / -name libibverbs*` empty, `ibv_devices` not found).
    # NO external libnccl-net.so/OFI plugin is needed on Mellanox IB. Also SKYRL_COMMIT 822221a0->272bf011
    # (penfever/working HEAD, direct child: CP>1 _C::rms_norm Meta-kernel fix) so --skyrl-ref 272bf011 is a
    # no-op safety belt. wheels + harbor d58043c3 + rl_env_constraints UNCHANGED (fast prebuilt-wheelhouse,
    # NO nvcc). Pull-verified: 48 layers, max 3.46 GB, 22.66 GB total. Build asserts green (flash_attn_2_cuda /
    # skyrl_train / vllm / torchtitan.ExpertParallel; baked MarinSkyRL HEAD == 272bf011; harbor 0.8.0). Expected
    # next-launch signal (NCCL_DEBUG=INFO): `NET/IB : Using [0]mlx5_0:1/IB` + `GPU Direct RDMA Enabled`.
    # gpu-rl-19bd8c5e (built 2026-07-13, kaniko gpurl-kaniko-19bd8c5e, SINGLE_SNAPSHOT=0 pullable): SKYRL_COMMIT
    # bump 272bf011->de40d31c (penfever/working HEAD, linear descendant — safe superset). Substantive change: a
    # LOG-CAPTURE-SAFE tqdm fallback (skyrl_train/utils/progress.py). On a non-TTY stderr (CoreWeave/Iris captured
    # container logs, SLURM) every SkyRL progress bar (Generation Buffer / Training Step / Generating Trajectories /
    # Evaluation) now emits THROTTLED newline-terminated loguru lines instead of invisible \r-in-place frames, so
    # progress finally shows up in the captured job logs; delegates to real tqdm on a TTY (auto-gated by
    # sys.stderr.isatty(), no launcher env wiring needed). wheels + harbor + rl_env_constraints UNCHANGED (fast
    # prebuilt-wheelhouse, NO nvcc). Pull-verified: 48 layers, max 3.46 GB. Build asserts green (flash_attn_2_cuda /
    # skyrl_train / vllm / torchtitan.ExpertParallel). NOTE: floating :gpu-rl tag was deliberately NOT moved
    # (PUSH_FLOATING=0) — promote it after a live smoke: `crane tag ...@sha256:98adaa38... gpu-rl`.
    # gpu-rl-318e18ce (built 2026-07-14, kaniko gpurl-kaniko-318e18ce, SINGLE_SNAPSHOT=0 pullable): HARBOR_COMMIT
    # bump -> 793ff3fb (round-2 per-turn coordinator-offload harbor fix). ALSO unbreaks the base build: reverts
    # docker/rl_env_constraints.txt tilelang 0.1.8->0.1.9. Commit 291cfef3 had wrongly pinned the BASE constraint
    # to 0.1.8 ("tilelang is FlashQLA-only"), but the vLLM fork's requirements/cuda.txt REQUIRES tilelang==0.1.9,
    # so the rl-stage `uv pip install -r requirements/cuda.txt` under UV_CONSTRAINT died "tilelang==0.1.9 and
    # tilelang==0.1.8 unsatisfiable" (first attempt gpurl-kaniko-6697dd20 FAILED here). The 0.1.8 pin belongs ONLY
    # in the FlashQLA incremental layer (clears UV_CONSTRAINT + re-downgrades on top). SKYRL de40d31c (baked,
    # unchanged vs 19bd8c5e). wheels UNCHANGED (fast prebuilt-wheelhouse, NO nvcc). Pull-verified: 48 layers, max
    # 3.46 GB. Build asserts green (flash_attn_2_cuda / skyrl_train / vllm / torchtitan.ExpertParallel; baked harbor
    # 0.8.1 commit 793ff3fb; baked MarinSkyRL HEAD de40d31c). Floating :gpu-rl tag WAS moved to this build (PUSH_FLOATING=1).
    # gpu-rl-f9110c79 (built 2026-07-14, kaniko job gpurl-kaniko-f9110c79, SINGLE_SNAPSHOT=0 pullable): HARBOR_COMMIT
    # bump 793ff3fb -> 35fbdbcc (round-3 per-turn coordinator-offload harbor fix; linear descendant of round-2 793ff3fb).
    # 3rd incremental harbor bump in a row (round-1 55ae9e66 -> round-2 793ff3fb=318e18ce -> round-3 35fbdbcc). SKYRL
    # de40d31c (baked, unchanged vs 318e18ce). wheels + rl_env_constraints UNCHANGED (fast prebuilt-wheelhouse, NO nvcc;
    # KANIKO_CACHE=1 reused the base apt/uv/venv layers). Pull-verified: 48 layers, max 3.46 GB, 22.71 GB total. Build
    # asserts green (flash_attn_2_cuda / torch 2.11.0+cu128 / vllm 0.1.dev16611+g76259c63a / skyrl_train / torchtitan
    # ExpertParallel; baked harbor 0.8.1 commit 35fbdbcc; baked MarinSkyRL HEAD de40d31c). Floating :gpu-rl tag WAS moved (PUSH_FLOATING=1).
    # gpu-rl-d0e4a9b8 (built 2026-07-14, kaniko job gpurl-kaniko-d0e4a9b8, SINGLE_SNAPSHOT=0 pullable): HARBOR_COMMIT
    # bump 35fbdbcc -> d81b2f32 (round-4: async /tokenize probe OFF the coordinator loop; linear descendant of round-3 35fbdbcc).
    # 4th incremental harbor bump in a row (round-1 55ae9e66 -> round-2 793ff3fb=318e18ce -> round-3 35fbdbcc=f9110c79 ->
    # round-4 d81b2f32). SKYRL de40d31c (baked, unchanged vs f9110c79). wheels + rl_env_constraints UNCHANGED (fast
    # prebuilt-wheelhouse, NO nvcc; KANIKO_CACHE=1 reused the base apt/uv/venv + torch/vLLM/flash-attn layers). Pull-verified:
    # 48 layers, max 3.46 GB (top: 3.46/3.2/3.0/2.71/2.07). Build asserts green (flash_attn_2_cuda / torch 2.11.0+cu128 /
    # vllm 0.1.dev16611+g76259c63a / skyrl_train / torchtitan ExpertParallel; baked harbor commit d81b2f32; baked MarinSkyRL
    # HEAD de40d31c). Floating :gpu-rl tag WAS moved to this build (PUSH_FLOATING=1). Build wall-clock ~20m (15:48->16:09).
    # gpu-rl-b397b82a (built 2026-07-14, kaniko job gpurl-kaniko-b397b82a, SINGLE_SNAPSHOT=0 pullable): HARBOR_COMMIT
    # bump d81b2f32 -> 101b1400 (round-5: 645d1074 global-caches the vLLM context-limit /v1/models probe [kills the
    # per-instance sync GET on the coordinator loop — the dominant v0j saturation blocker] + 101b1400 replaces litellm
    # with openai.AsyncOpenAI on the completions hot path [executor-free; retires ~440 LOC of FD-leak monkeypatches];
    # linear descendant of round-4 d81b2f32). 5th incremental harbor bump in a row (round-1 55ae9e66 -> round-2
    # 793ff3fb=318e18ce -> round-3 35fbdbcc=f9110c79 -> round-4 d81b2f32=d0e4a9b8 -> round-5 101b1400). SKYRL de40d31c
    # (baked, unchanged vs d0e4a9b8). wheels + rl_env_constraints UNCHANGED (fast prebuilt-wheelhouse, NO nvcc;
    # KANIKO_CACHE=1 reused the base apt/uv/venv + torch/vLLM/flash-attn layers). litellm is RETAINED (==1.90.0) for
    # harbor's off-path utilities — it is no longer on the completions hot path, so the harbor install still succeeds.
    # Pull-verified: 48 layers, max 3.46 GB (top: 3.46/3.2/3.0/2.71/2.07). Build asserts green (flash_attn_2_cuda /
    # torch 2.11.0+cu128 / vllm 0.1.dev16611+g76259c63a / skyrl_train / torchtitan ExpertParallel; baked harbor commit
    # 101b1400; harbor==0.8.1; baked MarinSkyRL HEAD de40d31c). Floating :gpu-rl tag WAS moved to this build (PUSH_FLOATING=1). Build wall-clock ~20m.
    # gpu-rl-e03896b7 (built 2026-07-15, kaniko job gpurl-kaniko-e03896b7, SINGLE_SNAPSHOT=0 pullable): HARBOR_COMMIT
    # bump 101b1400 -> f4a6b1a0 (round-6: offload the AsyncOpenAI raw-JSON parse OFF the coordinator asyncio loop —
    # orjson + asyncio.to_thread in lite_llm._acreate_chat_raw). Live py-spy on the running 30B v0k coordinator pinned
    # inline json.loads of the raw vLLM body at 84% of GIL-held samples as the batch-of-8 rollout-supply SAWTOOTH root
    # cause (TIS forces logprobs+return_token_ids so vLLM echoes prompt_token_ids every turn -> O(context) parse/turn,
    # ~10-30ms GIL-hold that stalls every co-resident trial's dispatch -> engines drain). Fix: orjson (3-10x faster,
    # shrinks the hold) parsed via to_thread (interleaves per-trial parses). orjson is now an EXPLICIT rl-image dep
    # (added `uv pip install orjson`; harbor installs --no-deps so it must be added here) — harbor falls back to stdlib
    # json when orjson is absent (non-RL images) or when it rejects a body (non-finite -Infinity logprobs), so
    # byte-fidelity is preserved and non-RL harbor is a no-op. Scoped to the OpenAI-compat _acreate_chat_raw path only;
    # native litellm fallback untouched. SKYRL de40d31c (baked, unchanged). wheels + rl_env_constraints UNCHANGED (fast
    # prebuilt-wheelhouse, NO nvcc). Pull-verified: 48 layers, max 3.46 GB (top: 3.46/3.2/3.0/2.71/2.07). Build asserts
    # green (kaniko state=succeeded exit 0, ~25m; asserts run inside the build so success == flash_attn_2_cuda /
    # skyrl_train / vllm / torchtitan.ExpertParallel import OK). baked harbor commit f4a6b1a0.
    "@sha256:e8b48241b548da570a319ff421e72787692ee87dae2408c99a2d0c6794186177"  # noqa: E501  (gpu-rl-e03896b7, PULLABLE; harbor f4a6b1a0 round-6 orjson parse-offload)
    # (prev: gpu-rl-b397b82a @sha256:bac11e44 harbor 101b1400 round-5 ctx-limit global-cache + AsyncOpenAI; gpu-rl-d0e4a9b8 @sha256:0fbf41e5 harbor d81b2f32 round-4 async tokenize offload; gpu-rl-f9110c79 @sha256:5e211fbf harbor 35fbdbcc round-3; gpu-rl-318e18ce @sha256:35fbf815 harbor 793ff3fb round-2 + tilelang 0.1.9 base-build fix; gpu-rl-19bd8c5e @sha256:98adaa38 log-capture-safe tqdm)
)
_SUPERSEDED_RL_IMAGES = (
    # gpu-rl-69634c0b (built 2026-07-02, kaniko job gpurl-kaniko-69634c0b): a HARBOR_COMMIT-ONLY bump
    # of the prior gpu-rl-1af0ae2d (@sha256:d77b34dd…) — baked harbor f7f51f13 → 0729a3e9, which sits
    # on top of ef42e75e and carries BOTH Daytona throughput fixes: (1) ef42e75e "replace 1s exec poll
    # with 1ms+jitter" (per-exec runtime+RTT vs ceil(runtime)+1s → feeds the RL engines faster), and
    # (2) 0729a3e9 "bake tmux+asciinema into every snapshot at BUILD time" (terminus-2 per-episode
    # tmux install short-circuits → kills the ~401s agent_setup / 63% AgentSetupTimeout; takes effect
    # only on a FRESH snapshot mint). Everything else is unchanged: MarinSkyRL 39faff7d (baked),
    # vLLM-fork 76259c63 (compiled) + flash-attn 2.8.3 + torch 2.11.0+cu128, and the rl env stays
    # PINNED to the gpu-rl-81045a29 known-good freeze via docker/rl_env_constraints.txt (UV_CONSTRAINT)
    # — so the NCCL regression of the deleted gpu-rl-00220aac cannot recur. Harbor is a --no-deps
    # source-only swap, so the wheel cache-key is untouched → the prebuilt vLLM-fork + flash-attn
    # wheels (laion/gpu-rl-build-wheels) stayed ABI-correct → FAST no-nvcc prebuilt-wheelhouse path.
    # Build asserts ran green: flash_attn_2_cuda OK, torch 2.11.0+cu128 / vllm 0.1.dev16611+g76259c63a
    # / skyrl_train import OK, torchtitan ExpertParallel import OK (EP>1 MoE unblock),
    # baked harbor 0.8.0 @ commit 0729a3e9, baked MarinSkyRL HEAD == 39faff7d.
    "@sha256:d9c7e6046e8392f3bb50567fa46e8ef3d39e49bd7fdc34409bf40f380a8596a2"
)
DEFAULT_GPU_VARIANT = "H100"
DEFAULT_GPUS_PER_NODE = 8  # gd-8xh100ib-i128 = 8x H100-80GB + IB
# These H100 nodes are requested WHOLE-NODE-EXCLUSIVE (no co-tenants) — so request ALL the
# node's allocatable resources; don't under-request (wasted capacity + a too-low --memory
# caused a container-cgroup OOM at FSDP weight-load on the 30B run). Node allocatable ≈ 128
# CPU / ~2014 GiB mem / 8 GPU.
#   - CPU 48 (NOT 64): ~64-68 of the 128 cores are persistent daemonset reservation, so a
#     request >~60 FAILS the single-IB-leaf gang admission (observed: 64 unplaceable, 48 admits).
#   - MEMORY 1700GB (≈1583 GiB cgroup) — RAISED from 1400GB on 2026-07-11 because 1400 was
#     UNDER-allocating. It must clear TWO opposing footguns:
#     (a) too LOW → container-cgroup OOM at the training forward. The old 512GB OOM'd at FSDP
#         weight-load; and 1400GB (≈1303 GiB cgroup) sits RIGHT AT the ~1303 GiB forward peak
#         at 8 ranks/node (the 80B cp1 128-GPU EP8×FSDP8 geometry) — the r2 rankspread run
#         measured a 1028 GiB forward peak at only 4 ranks/node (2026-07-11 diagnosis), so at
#         8 ranks/node the peak clears 1400's cgroup → no headroom = OOM risk; and
#     (b) too HIGH (1800GB ≈ 1676 GiB) → sits so close to node-allocatable (~2014 GiB) that
#         after daemonset/persistent-reservation overhead a leafgroup gang (all-or-nothing,
#         one IB leaf) can't fit all pods → Kueue SchedulingGated stall (cost multiple
#         60-120min stalls overnight 2026-06-26, on a 1-GPU probe AND 8-node gangs).
#     1700GB fits with headroom (≈1583 GiB < 2014 GiB allocatable) AND clears the forward
#     peak. Lower toward the real need on an admission stall; do NOT raise toward 1800.
#     (1000-1200GB suffices for 2-node smokes.) See .claude/ops/iris/coreweave_gpu_ops.md.
#   - DISK defaults to "auto" = 80% of the node's live allocatable ephemeral-storage (~27.2 TiB
#     → ~21 TiB). WHY NOT the old 512GB: the long MoE training step's Ray object store spills to
#     /tmp (a metered emptyDir that counts against the --disk ephemeral-storage limit), growing
#     to >1.6 TB and EVICTING the pod (2026-06-28). Whole-node-exclusive gangs have NO co-tenants,
#     so reserving disk is pure waste — claim ~80%. (R2 object-spilling, the durable fix, is also
#     on; this headroom is belt-and-suspenders.) Pass --disk explicitly to override.
DEFAULT_CPU_PER_NODE = 48.0
DEFAULT_MEMORY_PER_NODE = "1700GB"
# --disk "auto" → DISK_FRACTION of the GPU node's live allocatable ephemeral-storage at launch
# (FALLBACK_DISK_GIB iff the node query fails). See _resolve_default_disk().
DEFAULT_DISK_PER_NODE = "auto"
DISK_FRACTION = 0.80
FALLBACK_DISK_GIB = 21800  # ~80% of the h100-8x ~27.2 TiB allocatable, used only if kubectl is unavailable
DEFAULT_PRIORITY = "interactive"


def _parse_quantity_to_gib(q: str) -> float:
    """Parse a k8s resource quantity (plain bytes, or Ki/Mi/Gi/Ti binary / k/M/G/T decimal suffix) to GiB."""
    q = q.strip()
    for suf, mult in (("Ki", 2**10), ("Mi", 2**20), ("Gi", 2**30), ("Ti", 2**40), ("Pi", 2**50)):
        if q.endswith(suf):
            return float(q[: -len(suf)]) * mult / 2**30
    for suf, mult in (("k", 1e3), ("M", 1e6), ("G", 1e9), ("T", 1e12), ("P", 1e15)):
        if q.endswith(suf):
            return float(q[: -len(suf)]) * mult / 2**30
    return float(q) / 2**30  # plain bytes


def _resolve_default_disk(fraction: float = DISK_FRACTION) -> str:
    """``fraction`` of the GPU node's LIVE allocatable ephemeral-storage, as a ``"<N>Gi"`` string.

    Whole-node-exclusive gangs have no co-tenants, so claim most of the node NVMe (the old fixed
    512GB default evicted long MoE steps once Ray's object store spilled to the metered /tmp).
    Queries kubectl for the MIN allocatable across 8-GPU nodes (never over-request a smaller node);
    falls back to FALLBACK_DISK_GIB if kubectl is unavailable (requires KUBECONFIG)."""
    import subprocess

    try:
        out = subprocess.run(
            [
                "kubectl",
                "get",
                "nodes",
                "-o",
                r'jsonpath={range .items[*]}{.status.capacity.nvidia\.com/gpu}{" "}'
                r'{.status.allocatable.ephemeral-storage}{"\n"}{end}',
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
        allocs = [
            _parse_quantity_to_gib(p[1])
            for p in (line.split() for line in out.splitlines())
            if len(p) == 2 and p[0] == "8"
        ]
        if allocs:
            gib = int(min(allocs) * fraction)
            print(
                f"[rl-iris] --disk auto: {fraction:.0%} of node allocatable "
                f"(min {min(allocs):.0f}GiB across {len(allocs)} GPU nodes) = {gib}Gi",
                flush=True,
            )
            return f"{gib}Gi"
        print(
            f"[rl-iris] --disk auto: no 8-GPU nodes returned by kubectl; using fallback {FALLBACK_DISK_GIB}Gi",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001 - best-effort; fall back rather than block a launch
        print(
            f"[rl-iris] --disk auto: kubectl node query failed ({type(exc).__name__}: {exc}); "
            f"using fallback {FALLBACK_DISK_GIB}Gi",
            flush=True,
        )
    return f"{FALLBACK_DISK_GIB}Gi"


# The gpu-rl image's RL venv (deps-only: torch 2.11 + vLLM fork + skyrl editable).
RL_PYTHON = "/opt/openthoughts/envs/rl/bin/python"
SKYRL_HOME = "/opt/skyrl"
# In-container source sync target. iris syncs the launcher's `workspace`
# (this MarinSkyRL repo, PROJECT_ROOT — see IrisClient.remote(..., workspace=PROJECT_ROOT))
# to /app and sets IRIS_WORKDIR=/app; putting /app first on PYTHONPATH makes the live
# synced cloud.iris + skyrl-train code win over the image's baked copies. The runtime is
# self-contained here (cloud.iris.*) — no OpenThoughts-Agent workspace is required in-pod.
APP_DIR = "/app"


def _resolve_cluster_config_default() -> str:
    """Find the marin repo's cw-us-east-02a cluster YAML."""
    rel = f"lib/iris/config/{DEFAULT_CLUSTER}.yaml"
    candidates = [
        Path.home() / "Documents/marin" / rel,
        Path("/Users/benjaminfeuer/Documents/marin") / rel,
        Path(os.environ.get("MARIN_ROOT", "")) / rel,
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return rel


def _default_secrets_env() -> Optional[str]:
    cand = os.environ.get("OT_AGENT_SECRETS_ENV") or os.path.expanduser("~/Documents/secrets.env")
    return cand if os.path.isfile(cand) else None


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch a MarinSkyRL RL training job on the Iris CoreWeave H100 cluster.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- RL job args (mirror launch_rl_cloud.py) ---
    parser.add_argument(
        "--rl_config",
        required=True,
        help="Path to SkyRL/MarinSkyRL config YAML (repo-relative or absolute).",
    )
    parser.add_argument("--rl-config", dest="rl_config", help=argparse.SUPPRESS)

    parser.add_argument(
        "--model_path",
        required=True,
        help="Model path or HuggingFace ID (e.g., Qwen/Qwen3-8B).",
    )
    parser.add_argument("--model-path", dest="model_path", help=argparse.SUPPRESS)

    parser.add_argument(
        "--model-warm-source",
        "--model_warm_source",
        dest="model_warm_source",
        default=None,
        help="In-region CW-object-store prefix seeded (once, via scripts/iris/"
        "mirror_hf_to_s3.py) with the model weights, so the controller SYNCS them "
        "from there into each node's HF cache instead of cold-pulling ~160 GB per "
        "node from HF Hub (the flaky path behind the 80B r4a/r4b bring-up failures). "
        "Default: AUTO-DERIVE s3://marin-us-east-02a/models/<org>--<name> from the "
        "repo id (a missing/empty source is a clean no-op -> HF prestage fallback, "
        "byte-identical to today). Pass 'none'/'off' to DISABLE the warm path (pure "
        "HF prestage). Only used when the config runs HF_HUB_OFFLINE=1 with a "
        "repo-id model_path (same gate as --prestage-model).",
    )

    parser.add_argument(
        "--train_data",
        default="[]",
        help="Training data paths as a JSON list (e.g., '[\"org/dataset\"]').",
    )
    parser.add_argument("--train-data", dest="train_data", help=argparse.SUPPRESS)

    parser.add_argument(
        "--val_data",
        default="[]",
        help="Validation data paths as a JSON list.",
    )
    parser.add_argument("--val-data", dest="val_data", help=argparse.SUPPRESS)

    parser.add_argument(
        "--skyrl_override",
        action="append",
        default=[],
        help="SkyRL Hydra override (repeatable).",
    )
    parser.add_argument("--skyrl-override", dest="skyrl_override", action="append", help=argparse.SUPPRESS)

    parser.add_argument(
        "--experiments_dir",
        default="/app/experiments",
        help="In-container experiments output dir (on the synced /app workspace).",
    )
    parser.add_argument("--experiments-dir", dest="experiments_dir", help=argparse.SUPPRESS)

    # --- Resource / topology args (GPU multi-node) ---
    parser.add_argument(
        "--num-nodes",
        "--num_nodes",
        dest="num_nodes",
        type=int,
        default=1,
        help="Number of WHOLE H100 nodes to request EXCLUSIVELY, gang/co-scheduled "
        "(one iris task per node, all 8 GPUs each, coscheduled by leafgroup/IB).",
    )
    parser.add_argument(
        "--gpus-per-node",
        "--gpus_per_node",
        dest="gpus_per_node",
        type=int,
        default=DEFAULT_GPUS_PER_NODE,
        help="GPUs per node (CoreWeave nodes are 8x H100).",
    )
    parser.add_argument(
        "--gpu-variant",
        "--gpu_variant",
        dest="gpu_variant",
        default=DEFAULT_GPU_VARIANT,
        help="GPU variant (default H100).",
    )
    parser.add_argument(
        "--cpu",
        type=float,
        default=DEFAULT_CPU_PER_NODE,
        help="CPU cores per node.",
    )
    parser.add_argument(
        "--memory",
        default=DEFAULT_MEMORY_PER_NODE,
        help="Memory per node.",
    )
    parser.add_argument(
        "--disk",
        default=DEFAULT_DISK_PER_NODE,
        help=f"Ephemeral disk per node. Default 'auto' = {int(DISK_FRACTION * 100)}%% of the GPU "
        "node's live allocatable ephemeral-storage (whole-node-exclusive gangs have no "
        "co-tenants, so claim most of the node NVMe — keeps Ray object-spill / checkpoints "
        "clear of the ephemeral-storage eviction). Pass an explicit value (e.g. 4000GB) to override.",
    )
    parser.add_argument(
        "--ray-port",
        "--ray_port",
        dest="ray_port",
        type=int,
        default=6379,
        help="Port the cross-node Ray head binds.",
    )
    parser.add_argument(
        "--rendezvous-dir",
        "--rendezvous_dir",
        dest="rendezvous_dir",
        default=None,
        help="Shared object-store/path (gs://, s3://, or shared dir) for the multi-node "
        "Ray head/worker rendezvous. Required for --num-nodes>1. On cw-us-east-02a "
        "use an s3:// URI under the cluster's default bucket, e.g. "
        "s3://marin-us-east-02a/iris/rl-rdv/<job>; the cluster injects working creds "
        "+ AWS_ENDPOINT_URL into every task pod (iris-task-env Secret), so no external "
        "creds are needed. NOTE: the default object store moved R2 (s3://marin-na) -> "
        "CW (s3://marin-us-east-02a) on 2026-07-05 (marin c7caecc95a); pods now inject "
        "CW creds+endpoint and can NO LONGER reach s3://marin-na (R2).",
    )
    parser.add_argument(
        "--rendezvous-timeout",
        "--rendezvous_timeout",
        dest="rendezvous_timeout",
        type=int,
        default=None,
        help="Seconds the worker ranks poll for rank-0's Ray-head rendezvous file "
        "(forwarded to start_rl_iris_controller.py --rendezvous-timeout). Unset = the "
        "controller default (1800s). RAISE it (e.g. 3600) for a big model whose rank-0 "
        "pre-stage/snapshot_download can legitimately take >30 min, so a SLOW-but-not-hung "
        "head prestage completes inside the window instead of the workers timing out and "
        "killing the gang (the 80B rank-spread bring-up flake, 2026-07-11).",
    )
    parser.add_argument(
        "--trials-dir",
        "--trials_dir",
        dest="trials_dir",
        default="auto",
        help="Where Harbor writes per-trial agentic-RL rollout artifacts "
        "(terminal_bench_config.trials_dir). 'auto' (default) = the durable shared store at "
        "s3://marin-us-east-02a/iris/<job_name>/trace_jobs (pods reach it via auto-injected "
        "creds; inspectable post-hoc). 'local'/'off' = keep the config default (node-local "
        "/app/experiments/<run>/trace_jobs). Or pass an explicit s3://, gs://, or path URI. "
        "Ignored if you already set terminal_bench_config.trials_dir via --skyrl_override.",
    )

    # --- Iris submission args (mirror launch_eval_iris.py / IrisLauncher) ---
    parser.add_argument(
        "--cluster",
        default=DEFAULT_CLUSTER,
        help="Iris cluster name (default cw-us-east-02a).",
    )
    parser.add_argument(
        "--cluster-config",
        "--cluster_config",
        dest="cluster_config",
        default=_resolve_cluster_config_default(),
        help="Path to the iris cluster YAML (default: cw-us-east-02a in the marin repo).",
    )
    parser.add_argument(
        "--task-image",
        "--task_image",
        "--docker_image",
        "--docker-image",
        dest="task_image",
        default=DEFAULT_RL_DOCKER_IMAGE,
        help=f"Container image (default {DEFAULT_RL_DOCKER_IMAGE}).",
    )
    parser.add_argument(
        "--job-name",
        "--job_name",
        dest="job_name",
        default=None,
        help="Job name (auto-derived if not set).",
    )
    parser.add_argument(
        "--priority",
        default=DEFAULT_PRIORITY,
        choices=["production", "interactive", "batch"],
        help="Iris priority band.",
    )
    parser.add_argument(
        "--max-retries",
        "--max_retries",
        dest="max_retries",
        type=int,
        default=0,
        help="Max retries on failure (iris auto-retries preemptions separately).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Job timeout in seconds (0 = no timeout).",
    )
    parser.add_argument(
        "--no-wait",
        dest="no_wait",
        action="store_true",
        default=False,
        help="Submit and detach instead of streaming logs.",
    )
    parser.add_argument(
        "--preemptible",
        dest="preemptible",
        action="store_true",
        default=None,
        help="Force scheduling on preemptible workers.",
    )
    parser.add_argument(
        "--no-preemptible",
        dest="preemptible",
        action="store_false",
        help="Force scheduling on non-preemptible workers.",
    )
    parser.add_argument(
        "--secrets-env",
        "--secrets_env",
        dest="secrets_env",
        default=_default_secrets_env(),
        help="KEY=VALUE env file injected into the task (HF_TOKEN, WANDB_API_KEY, etc.). "
        "Defaults to $OT_AGENT_SECRETS_ENV, else ~/Documents/secrets.env.",
    )
    parser.add_argument(
        "--daytona-api-key-env",
        "--daytona_api_key_env",
        dest="daytona_api_key_env",
        default=os.environ.get("DAYTONA_KEY_OVERRIDE"),
        help="Name of an env var whose VALUE is forwarded to the pod as DAYTONA_API_KEY "
        "(routes agentic RL onto a dedicated Daytona org, e.g. "
        "--daytona-api-key-env DAYTONA_RL_API_KEY). Applied AFTER --secrets-env is "
        "re-sourced (which does 'file overrides shell'), so the override actually STICKS "
        "where a plain shell `export DAYTONA_API_KEY=...` is silently clobbered. "
        "Referenced by NAME only; no key value on the command line. "
        "Defaults to $DAYTONA_KEY_OVERRIDE.",
    )
    parser.add_argument(
        "--skyrl-ref",
        "--skyrl_ref",
        dest="skyrl_ref",
        default=None,
        help="If set, `git fetch && git checkout <ref>` the baked MarinSkyRL clone at "
        "/opt/skyrl BEFORE running, so the live editable install picks up a newer "
        "(or pinned) commit than the one baked into the image. Use to apply a "
        "MarinSkyRL fix that landed AFTER the image was built without waiting for an "
        "image rebuild (deps are baked, but skyrl-train is an editable git clone, so "
        "a checkout is live). Default: unset = use whatever commit the image baked.",
    )
    # ----------------------------------------------------------------------- #
    # MarinSkyRL runtime-knob flags (deslop stage 3). Each promotes a live      #
    # SKYRL_* runtime env var to a first-class CLI flag. ALL default to None    #
    # ("unspecified") so an all-defaults launch injects NOTHING and the pod env #
    # is byte-identical to today (the SkyRL code's own default applies). A       #
    # config's `extra_env:` block is overlaid on TOP of these, so extra_env      #
    # still wins: precedence is  env/extra_env > flag > code-default.            #
    # ----------------------------------------------------------------------- #
    g = parser.add_argument_group("MarinSkyRL runtime knobs (SKYRL_* -> flags)")
    g.add_argument(
        "--r3-transport",
        "--r3_transport",
        dest="r3_transport",
        choices=["by_value", "resident", "decentral"],
        default=None,
        help="R3 (rollout routed-experts) transport for MoE async RL. 'decentral' "
        "(code default) routes the captured routed-experts generation-worker -> "
        "node-resident consumer (head holds ~0 R3); 'resident' de-dups to 1 "
        "copy/dp-group on the driver head plasma; 'by_value' is the old per-actor "
        "by-value dispatch. Folds SKYRL_R3_RESIDENT + SKYRL_R3_DECENTRAL. "
        "Default: unset = code default (decentral).",
    )
    g.add_argument(
        "--r3-put-timeout-s",
        "--r3_put_timeout_s",
        dest="r3_put_timeout_s",
        type=int,
        default=None,
        help="Bounded ray.put() timeout (s) for an R3 dp-chunk dispatch "
        "(SKYRL_DISPATCH_PUT_TIMEOUT_S). Default: unset = 600.",
    )
    g.add_argument(
        "--nccl-timeout-s",
        "--nccl_timeout_s",
        dest="nccl_timeout_s",
        type=int,
        default=None,
        help="Worker NCCL-collective timeout in seconds (SKYRL_WORKER_NCCL_TIMEOUT_IN_S). Default: unset = 1800.",
    )
    g.add_argument(
        "--host-ram-monitor",
        dest="host_ram_monitor",
        choices=["on", "off"],
        default=None,
        help="Policy-worker host-RAM/cgroup-mem monitor thread (SKYRL_POLICY_HOST_RAM_MONITOR). Default: unset = on.",
    )
    g.add_argument(
        "--host-ram-monitor-interval-s",
        dest="host_ram_monitor_interval_s",
        type=int,
        default=None,
        help="Host-RAM monitor sample interval, s (SKYRL_POLICY_HOST_RAM_MONITOR_INTERVAL). Default: unset = 60.",
    )
    g.add_argument(
        "--tis-splice",
        dest="tis_splice",
        choices=["on", "off"],
        default=None,
        help="TIS served-id splice policy (SKYRL_TIS_SPLICE) — use vLLM's raw served "
        "token ids as the generated region for exact-by-id TIS alignment. "
        "Default: unset = on (no-op on non-thinking turns).",
    )
    g.add_argument(
        "--gdn-mask-fla",
        dest="gdn_mask_fla",
        choices=["auto", "on", "off"],
        default=None,
        help="Force the pure-torch GatedDeltaNet path / mask the broken fla wheel "
        "(SKYRL_GDN_MASK_FLA). 'auto' (and unset) derive it from the model arch "
        "(on for Qwen3-Next/GDN, off for dense). Default: unset = auto.",
    )
    g.add_argument(
        "--gdn-flashqla",
        dest="gdn_flashqla",
        choices=["on", "off"],
        default=None,
        help="Opt-in FlashQLA fused GDN tilelang kernel (SKYRL_GDN_FLASHQLA); needs the "
        "fla_tilelang overlay. Default: unset = off.",
    )
    g.add_argument(
        "--forward-dispatch-fix",
        dest="forward_dispatch_fix",
        choices=["on", "off"],
        default=None,
        help="MoE async-dispatch forward fix (SKYRL_FORWARD_DISPATCH_FIX), a correctness "
        "knob. Default: unset = on. Pass off only for an A/B.",
    )
    g.add_argument(
        "--weightsync-drain-barrier",
        dest="weightsync_drain_barrier",
        choices=["on", "off"],
        default=None,
        help="Post-weight-sync async drain barrier (SKYRL_WEIGHTSYNC_DRAIN_BARRIER), a "
        "correctness knob. Default: unset = on.",
    )
    g.add_argument(
        "--cp-require-right-align",
        dest="cp_require_right_align",
        choices=["on", "off"],
        default=None,
        help="Require right-aligned attention mask under context-parallel "
        "(SKYRL_CP_REQUIRE_RIGHT_ALIGN), a correctness knob. Default: unset = on.",
    )
    g.add_argument(
        "--w13-reload-bracket",
        dest="w13_reload_bracket",
        choices=["on", "off"],
        default=None,
        help="Bracket the MoE weight-sync with layerwise-reload init/finalize so FusedMoE "
        "w13 is re-swapped exactly once (SKYRL_W13_RELOAD_BRACKET), a correctness "
        "knob. Default: unset = on.",
    )
    g.add_argument(
        "--ep-loader-chunk-rows",
        dest="ep_loader_chunk_rows",
        type=int,
        default=None,
        help="Per-broadcast dim-0 row budget for the streamed EP full-state-dict loader "
        "(SKYRL_EP_LOADER_CHUNK_ROWS). Default: unset = 8.",
    )
    g.add_argument(
        "--collective-count-diag",
        dest="collective_count_diag",
        choices=["on", "off"],
        default=None,
        help="GC-proof per-rank default-PG collective-count instrumentation "
        "(SKYRL_COLLECTIVE_COUNT_DIAG), a DIAGNOSTIC knob for the 80B gs1 NCCL "
        "desync. Logs each policy rank's default-PG collective count at forward "
        "phase boundaries (forward/_forward_impl enter+exit + the first MoE-EP "
        "all-to-all per forward) to the finelog, which survives pod GC — diffing "
        "the counts across ranks at the wedge localizes the divergent EP group. "
        "O(phases), reads torch's own PG seq counter (no perturbation). "
        "Default: unset = off.",
    )

    parser.add_argument(
        "--dry-run",
        "--dry_run",
        dest="dry_run",
        action="store_true",
        default=False,
        help="Print the resolved config + in-container command without submitting.",
    )

    return parser


def build_skyrl_flag_env(args: argparse.Namespace) -> dict[str, str]:
    """Translate the MarinSkyRL runtime-knob CLI flags into SKYRL_* env vars for the
    pod. Only flags that were explicitly set (non-None) emit an entry, so an
    all-defaults invocation returns {} and the pod env stays byte-identical to today.
    The caller overlays the config's ``extra_env:`` on top of this, so a config's
    explicit value still wins (precedence: env/extra_env > flag > code default)."""
    env: dict[str, str] = {}

    def _onoff(name: str, value) -> None:
        if value is not None:
            env[name] = "1" if value == "on" else "0"

    # R3 transport: fold the nested resident && decentral gating into one choice.
    if args.r3_transport == "by_value":
        env["SKYRL_R3_RESIDENT"] = "0"
    elif args.r3_transport == "resident":
        env["SKYRL_R3_RESIDENT"] = "1"
        env["SKYRL_R3_DECENTRAL"] = "0"
    elif args.r3_transport == "decentral":
        env["SKYRL_R3_RESIDENT"] = "1"
        env["SKYRL_R3_DECENTRAL"] = "1"
    if args.r3_put_timeout_s is not None:
        env["SKYRL_DISPATCH_PUT_TIMEOUT_S"] = str(args.r3_put_timeout_s)
    if args.nccl_timeout_s is not None:
        env["SKYRL_WORKER_NCCL_TIMEOUT_IN_S"] = str(args.nccl_timeout_s)
    _onoff("SKYRL_POLICY_HOST_RAM_MONITOR", args.host_ram_monitor)
    if args.host_ram_monitor_interval_s is not None:
        env["SKYRL_POLICY_HOST_RAM_MONITOR_INTERVAL"] = str(args.host_ram_monitor_interval_s)
    _onoff("SKYRL_TIS_SPLICE", args.tis_splice)
    # GDN mask: 'auto' (like unset) leaves the env unset so the code auto-derives.
    if args.gdn_mask_fla in ("on", "off"):
        env["SKYRL_GDN_MASK_FLA"] = "1" if args.gdn_mask_fla == "on" else "0"
    _onoff("SKYRL_GDN_FLASHQLA", args.gdn_flashqla)
    _onoff("SKYRL_FORWARD_DISPATCH_FIX", args.forward_dispatch_fix)
    _onoff("SKYRL_WEIGHTSYNC_DRAIN_BARRIER", args.weightsync_drain_barrier)
    _onoff("SKYRL_CP_REQUIRE_RIGHT_ALIGN", args.cp_require_right_align)
    _onoff("SKYRL_W13_RELOAD_BRACKET", args.w13_reload_bracket)
    if args.ep_loader_chunk_rows is not None:
        env["SKYRL_EP_LOADER_CHUNK_ROWS"] = str(args.ep_loader_chunk_rows)
    _onoff("SKYRL_COLLECTIVE_COUNT_DIAG", args.collective_count_diag)
    return env


def load_config_extra_env(rl_config_path: str) -> dict[str, str]:
    """Read a top-level ``extra_env:`` mapping from the RL config YAML.

    The Iris path has NO ``container:`` block (the gpu-rl Docker image is the
    runtime), so any SLURM-style ``container.extra_env`` shell-export plumbing never
    runs — without this, env declared in the YAML is silently
    dropped and only the launcher's hardcoded passthrough (HF/WANDB/DAYTONA) reaches
    the pod. This forwards a top-level ``extra_env:`` block (and, defensively,
    ``container.extra_env`` if a ported config still carries one) into the iris
    EnvironmentSpec so e.g. EPDIAG probe arms + R3/DCP guard env take effect.

    Values are coerced to str (YAML may parse "1"/true as int/bool). Returns {} if
    the file is unreadable or declares no extra_env (byte-identical behavior for the
    existing extra_env-less iris configs).
    """
    try:
        full = PROJECT_ROOT / rl_config_path
        path = full if full.exists() else Path(rl_config_path)
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    except Exception as exc:  # noqa: BLE001
        print(f"[rl-iris] WARNING: could not read extra_env from {rl_config_path}: {exc}", file=sys.stderr)
        return {}
    extra = dict(raw.get("extra_env") or {})
    container_env = (raw.get("container") or {}).get("extra_env") or {}
    for k, v in container_env.items():
        extra.setdefault(k, v)
    out: dict[str, str] = {}
    for k, v in extra.items():
        if v is None:
            continue
        if isinstance(v, bool):
            v = int(v)
        out[str(k)] = str(v)
    return out


def load_config_trainer_ckpt_path(rl_config_path: str) -> Optional[str]:
    """Return an EXPLICIT ``trainer.ckpt_path`` from the RL config YAML, else None.

    The iris configs set ``ckpt_path: null`` (auto-derived downstream in
    rl_config_translation). A config that sets it explicitly (non-null, non-empty) should
    WIN over the launcher's durable-s3 default, so build_task_command consults this
    before injecting its override. Returns None when the file is unreadable, has no
    ``trainer.ckpt_path``, or the value is null/empty (byte-identical to today for
    every existing iris config, which all leave it null)."""
    try:
        full = PROJECT_ROOT / rl_config_path
        path = full if full.exists() else Path(rl_config_path)
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    except Exception as exc:  # noqa: BLE001
        print(f"[rl-iris] WARNING: could not read ckpt_path from {rl_config_path}: {exc}", file=sys.stderr)
        return None
    val = (raw.get("trainer") or {}).get("ckpt_path")
    if val is None or (isinstance(val, str) and not val.strip()):
        return None
    return str(val)


def _job_scope_fr_dump_path(prefix: str, job_name: str) -> str:
    """Rewrite a JOB-SCOPED NCCL flight-recorder dump path so its slug segment is the
    ACTUAL job name, e.g. ``/tmp/fr_dumps/<slug>/nccl_fr_rank`` -> ``/tmp/fr_dumps/
    <job_name>/nccl_fr_rank``.

    WHY (2026-07-11 FR-slug bug): the 80B configs hardcode
    ``TORCH_NCCL_DEBUG_INFO_TEMP_FILE: /tmp/fr_dumps/80b-next-cp1/nccl_fr_rank`` in
    their ``extra_env:``, so a run launched under a DIFFERENT ``--job-name`` (e.g.
    ``80b-next-cp1-r3d2``) still wrote its FR dumps under the stale ``80b-next-cp1``
    slug (harmless there, but wrong — a future FR dump would land under the wrong
    slug). Deriving the slug from the live job name keeps the dump under the right
    per-job dir. The controller's ``ensure_fr_dump_dir`` mkdir -p's whatever dirname
    the cvar carries, so overriding the cvar here is sufficient.

    ONLY rewrites the job-scoped ``.../fr_dumps/<slug>/<file>`` pattern; a bare
    generic path (``/tmp/nccl_fr_rank``, which every non-80B iris config uses) has no
    slug segment and is returned UNCHANGED (byte-identical for those configs)."""
    parent = os.path.dirname(prefix)  # e.g. /tmp/fr_dumps/<slug>
    grandparent = os.path.dirname(parent)  # e.g. /tmp/fr_dumps
    if os.path.basename(grandparent) != "fr_dumps":
        return prefix  # not a job-scoped fr_dumps path; leave it
    return os.path.join(grandparent, job_name, os.path.basename(prefix))


def normalize(args: argparse.Namespace) -> None:
    """Validate + normalize. Keep rl_config repo-relative so it resolves on /app."""
    # Resolve rl_config to a repo-relative path (it must exist on the synced
    # /app workspace, NOT be an absolute host path).
    rl_cfg = Path(args.rl_config)
    if rl_cfg.is_absolute():
        try:
            args.rl_config = str(rl_cfg.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            raise SystemExit(
                f"--rl_config {args.rl_config!r} is absolute and not under the repo "
                f"({PROJECT_ROOT}); pass a repo-relative path so it resolves on /app."
            )
    # Verify it exists locally (so we fail fast before submitting).
    if not (PROJECT_ROOT / args.rl_config).exists():
        # Fall back to cloud/iris/configs/<name>[.yaml].
        yaml_dir = Path("cloud/iris/configs")
        for cand in (yaml_dir / args.rl_config, yaml_dir / f"{args.rl_config}.yaml"):
            if (PROJECT_ROOT / cand).exists():
                args.rl_config = str(cand)
                break
        else:
            print(
                f"[rl-iris] WARNING: --rl_config {args.rl_config!r} not found under "
                f"{PROJECT_ROOT}; the worker will error if it isn't on /app.",
                file=sys.stderr,
            )

    if args.num_nodes < 1:
        raise SystemExit("--num-nodes must be >= 1.")
    if args.num_nodes > 1 and not args.rendezvous_dir:
        raise SystemExit(
            "--num-nodes>1 requires --rendezvous-dir (a shared gs://, s3://, or path URI "
            "both head and worker nodes can reach) for the multi-node Ray rendezvous."
        )


def build_task_command(args: argparse.Namespace) -> List[str]:
    """Build the in-container command, multi-node-aware.

    The full pipeline that runs inside each task container:
      cd /app
      && export SKYRL_HOME + PYTHONPATH (live /app + skyrl-train win)
      && <RL_PYTHON> cloud/iris/start_rl_iris_controller.py
            --ray-port ... --rendezvous-dir ...
            -- <RL_PYTHON> -m cloud.iris.run_rl --rl_config ... --num_nodes N ...

    Rank 0 (IRIS_TASK_ID==0) starts the Ray head and runs run_rl.py (which, with
    RAY_ADDRESS set + --num_nodes>1, attaches to the cluster instead of starting a
    local one). Workers join Ray and park. We invoke the gpu-rl venv python by
    absolute path so it is used regardless of whichever venv iris's setup phase
    activates.
    """
    total_gpus = args.num_nodes * args.gpus_per_node

    # The MarinSkyRL training command rank 0 runs (run_rl.py owns config parse,
    # hydra-arg build, HF data resolution, and the SkyRL entrypoint launch).
    train_cmd: List[str] = [
        RL_PYTHON,
        "-m",
        "cloud.iris.run_rl",
        "--rl_config",
        args.rl_config,
        "--model_path",
        args.model_path,
        "--job_name",
        args.job_name,
        "--gpus",
        str(total_gpus),
        "--num_nodes",
        str(args.num_nodes),
        "--gpus_per_node",
        str(args.gpus_per_node),
        "--experiments_dir",
        args.experiments_dir,
        "--ray_port",
        str(args.ray_port),
    ]
    if args.train_data and args.train_data != "[]":
        train_cmd.extend(["--train_data", args.train_data])
    if args.val_data and args.val_data != "[]":
        train_cmd.extend(["--val_data", args.val_data])
    for override in args.skyrl_override or []:
        train_cmd.extend(["--skyrl_override", override])

    # Durable Harbor rollout artifacts. The config default (trials_dir: null) resolves to a
    # node-local path on the rank-0 pod (/app/experiments/<run>/trace_jobs); point
    # terminal_bench_config.trials_dir at the durable shared store (s3://, creds auto-injected)
    # so rollouts persist + are inspectable post-hoc. Skip if the user opted out
    # (--trials-dir local) or already set it explicitly.
    trials_dir = (args.trials_dir or "auto").strip()
    user_set_trials = any("terminal_bench_config.trials_dir=" in o for o in (args.skyrl_override or []))
    if trials_dir.lower() not in ("local", "off", "none", "") and not user_set_trials:
        if trials_dir.lower() == "auto":
            trials_dir = f"s3://marin-us-east-02a/iris/{args.job_name}/trace_jobs"
        train_cmd.extend(["--skyrl_override", f"++terminal_bench_config.trials_dir={trials_dir}"])

    # Durable RESUMABLE checkpoint (preempt-safe -> makes `--priority batch` safe for
    # long runs). Without this, trainer.ckpt_path auto-derives (rl_config_translation) to
    # {experiments_dir}/{job_name}/checkpoints, and on iris experiments_dir defaults to
    # the in-container /app/experiments — EPHEMERAL pod-local disk. A batch preempt +
    # re-admit wipes it, so the trainer can't find latest_ckpt_global_step.txt and
    # restarts from step 0 despite resume_mode: latest. Redirect ckpt_path to a STABLE
    # per-job path on the durable CW object store — SAME bucket + auto-injected creds
    # path as trials_dir above, so ckpt co-locates with rollouts and follows any store
    # migration identically. It MUST be keyed on job_name ONLY (NOT a fresh-per-attempt
    # sub-path) so a re-admitted SAME --job-name job finds latest_ckpt_global_step.txt
    # (read path == write path, MarinSkyRL skyrl-train utils/io/io.py is fsspec-s3) and
    # auto-resumes from the banked step. iris-ONLY: SLURM uses a different launcher where
    # experiments_dir is durable $WORK, so this path never runs there. Respect an
    # explicit ckpt_path from the YAML or a --skyrl_override (either wins).
    user_set_ckpt = any("trainer.ckpt_path=" in o for o in (args.skyrl_override or []))
    yaml_ckpt = load_config_trainer_ckpt_path(args.rl_config)
    if not user_set_ckpt and not yaml_ckpt:
        ckpt_path = f"s3://marin-us-east-02a/iris/{args.job_name}/checkpoints"
        train_cmd.extend(["--skyrl_override", f"++trainer.ckpt_path={ckpt_path}"])
        print(f"[rl-iris] Durable resumable ckpt_path: {ckpt_path}")

    # The controller wraps the training command for the multi-node Ray bootstrap.
    controller_cmd: List[str] = [
        RL_PYTHON,
        "cloud/iris/start_rl_iris_controller.py",
        "--ray-port",
        str(args.ray_port),
    ]
    if args.rendezvous_dir:
        controller_cmd.extend(["--rendezvous-dir", args.rendezvous_dir])
    # Worker rendezvous poll deadline. Unset = controller default (1800s). Raise it when
    # rank-0's per-node pre-stage of a large model can legitimately exceed 30 min, so a
    # slow-but-not-hung head prestage completes before the workers give up + kill the gang.
    if args.rendezvous_timeout is not None:
        controller_cmd.extend(["--rendezvous-timeout", str(args.rendezvous_timeout)])
    # Per-NODE task-dataset staging. run_rl.py's resolve_rl_train_data() extracts the
    # HF task dataset to the node-local $DCFT=/opt/openthoughts/tasks/ (gpu-rl image),
    # but it runs ONLY on rank 0 (the head), so the Ray-scheduled rollout workers on
    # ranks 1..N-1 find an empty tasks dir and every rollout dies with
    # FileNotFoundError: .../task.toml -> reward always 0 (data-starved, doomed run).
    # Fix: forward --train-data to the controller so it can run the SAME extraction
    # on EVERY node before Ray starts, populating the identical node-local path on
    # all pods. Idempotent (on_exist=skip) — rank-0's later run_rl re-resolve is a
    # cheap no-op.
    if args.train_data and args.train_data != "[]":
        controller_cmd.extend(["--train-data", args.train_data])
    # Per-NODE model pre-staging, coupled to HF_HUB_OFFLINE. A config that runs the
    # FSDP ranks offline (extra_env HF_HUB_OFFLINE=1) has NO warm cache unless the
    # weights are pulled first; without pre-staging each of the N*8 ranks would race
    # HF Hub online inside init_model and a slow straggler blows the 20-min c10d store
    # barrier (the 80B init-straggle kill, 2026-07-10). When the config is offline,
    # forward the model repo-id so the controller pre-downloads it ONCE PER NODE into
    # the node-local HF cache before Ray — off the collective critical path. Online
    # configs are byte-identical (no flag forwarded).
    _cfg_env = load_config_extra_env(args.rl_config)
    if str(_cfg_env.get("HF_HUB_OFFLINE", "")).strip().lower() in ("1", "true", "yes", "on"):
        if args.model_path and not args.model_path.startswith(("s3://", "gs://", "gcs://")):
            controller_cmd.extend(["--prestage-model", args.model_path])
            # In-region warm source. Default = auto-derive the CW-S3 convention path from
            # the repo id; a seed job (mirror_hf_to_s3.py) populates it once and every node
            # then S3-syncs from there instead of cold-pulling from HF Hub. When the source
            # is un-seeded the controller falls back to the HF prestage (byte-identical to
            # pre-warm-path). 'none'/'off' disables the warm path entirely (pure HF prestage).
            warm = args.model_warm_source
            if warm is None:
                warm = f"s3://marin-us-east-02a/models/{args.model_path.replace('/', '--')}"
            elif warm.strip().lower() in ("none", "off", ""):
                warm = None
            if warm:
                controller_cmd.extend(["--model-warm-source", warm])
    controller_cmd.append("--")
    controller_cmd.extend(train_cmd)

    # Wrap in a bash bootstrap: cd to the synced workspace and set PYTHONPATH so
    # live /app + skyrl-train win over the image's baked copies. Use the absolute
    # RL venv python (set above) — independent of iris's activated venv.
    pythonpath = f"{APP_DIR}:{SKYRL_HOME}/skyrl-train"
    # Optional: refresh the baked MarinSkyRL editable clone to a newer/pinned commit
    # before running (deps are baked, but skyrl-train is `pip install -e` over a git
    # clone, so a checkout is live without reinstall). Fetch is best-effort but the
    # checkout MUST succeed (the ref is the whole point), so it's under `set -e`.
    skyrl_refresh = ""
    if args.skyrl_ref:
        ref = shlex.quote(args.skyrl_ref)
        skyrl_refresh = (
            f"git -C {shlex.quote(SKYRL_HOME)} fetch --quiet --all || true; "
            f"git -C {shlex.quote(SKYRL_HOME)} checkout {ref}; "
            # Purge baked bytecode after the checkout. The gpu-rl image bakes
            # `.pyc` for the editable skyrl-train at its build-time commit; if those
            # were compiled with hash-based (UNCHECKED_HASH) invalidation, Python
            # does NOT recompile when `git checkout` swaps the `.py` underneath, so
            # a `--skyrl-ref` checkout SILENTLY runs the stale baked bytecode (proven
            # 2026-06-25: the norm_topk_prob fix at 518179d checked out, but the pod
            # raised at the pre-fix line numbers). Delete the cache so the live `.py`
            # is recompiled. Best-effort (|| true) — must not block on a read-only fs.
            f"find {shlex.quote(SKYRL_HOME)}/skyrl-train -name '*.pyc' -delete 2>/dev/null || true; "
            f"find {shlex.quote(SKYRL_HOME)}/skyrl-train -name __pycache__ -type d -prune -exec rm -rf {{}} + 2>/dev/null || true; "
            f'echo "[rl-iris] MarinSkyRL now at $(git -C {shlex.quote(SKYRL_HOME)} rev-parse HEAD)"; '
        )
    ctrl = shlex.join(controller_cmd)
    # TileLang JIT-cache warm-start shim (Fix A) — GDN/FlashQLA runs only.
    # SKYRL_GDN_FLASHQLA=1 lazily JIT-compiles the FlashQLA GatedDeltaNet TileLang
    # kernels on the first GPU forward into the node-local, ephemeral TileLang cache
    # (~71 min cold on the first r4f run, x every one of the N gang pods — kaniko is
    # CPU-only so they can't be baked into the image). This brackets the train command
    # with a per-pod, per-NODE cache sync (the bash runs once per task pod / node, and
    # TileLang's cache is node-local, so one --down warms all 8 local GPU workers):
    #   --down BEFORE the controller -> pulls the keyed warm cache (seed cache.tgz +
    #          incremental per-hash-dir objects) into TILELANG_CACHE_DIR so TileLang
    #          hash-matches and skips the cold compile. A miss is a warn+continue no-op.
    #   --up   at EXIT (bash EXIT trap; fires on normal completion AND a `set -e`/crash
    #          exit) -> uploads NEWLY-compiled hash-dirs as per-hash objects (race-free
    #          across the ~16 writers — content-addressed, no cache.tgz overwrite).
    # The shim self-gates on SKYRL_GDN_FLASHQLA and NEVER fails the job (best-effort;
    # exits 0 even on S3 error). We ALSO branch here on SKYRL_GDN_FLASHQLA so a non-GDN
    # run (e.g. 30B-coder) keeps the BYTE-IDENTICAL `exec <controller>` fast path.
    # TILELANG_CACHE_DIR is exported (defaulting to TileLang's own default) so the shim
    # and the trainer's TileLang agree on the location; a config-set value wins.
    # TILELANG_CACHE_MODEL_PATH lets the shim derive the model component of the key.
    sync_py = "cloud/iris/tilelang_cache_sync.py"
    tl_down = f"{RL_PYTHON} {sync_py} --down || true"
    tl_up = f"{RL_PYTHON} {sync_py} --up || true"
    # The controller is run as a BACKGROUND child + `wait` (not `exec`) so we can
    # (a) run --up at exit via the bash EXIT trap and (b) FORWARD SIGTERM/SIGINT to
    # the controller — preserving the old `exec` graceful-shutdown path (rank-0's Ray
    # teardown + done-marker on preemption) that a plain child would lose. `wait` is
    # interrupted by the trapped signal (rc>128); we re-`wait` to reap the child's
    # real exit code after its forwarded-TERM shutdown.
    gdn_branch = (
        f'if [ "${{SKYRL_GDN_FLASHQLA:-0}}" = "1" ] || '
        f'[ "${{SKYRL_GDN_FLASHQLA:-}}" = "true" ] || '
        f'[ "${{SKYRL_GDN_FLASHQLA:-}}" = "on" ]; then '
        f'export TILELANG_CACHE_DIR="${{TILELANG_CACHE_DIR:-/root/.tilelang/cache}}"; '
        f"export TILELANG_CACHE_MODEL_PATH={shlex.quote(args.model_path)}; "
        f"{tl_down}; "
        f"trap {shlex.quote(tl_up)} EXIT; "
        f'trap \'[ -n "$_child" ] && kill -TERM "$_child" 2>/dev/null\' TERM INT; '
        f"set +e; {ctrl} & _child=$!; "
        f'wait "$_child"; _rc=$?; '
        f'if [ $_rc -gt 128 ]; then wait "$_child" 2>/dev/null; _rc=$?; fi; '
        f"exit $_rc; "
        f"else exec {ctrl}; fi"
    )
    bash = (
        f"set -e; cd {APP_DIR}; "
        f"{skyrl_refresh}"
        f"export SKYRL_HOME={shlex.quote(SKYRL_HOME)}; "
        f"export PYTHONPATH={shlex.quote(pythonpath)}:${{PYTHONPATH:-}}; "
        f"export VLLM_USE_V1=1; "
        f"{gdn_branch}"
    )
    return ["bash", "-c", bash]


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()
    normalize(args)

    if not args.job_name:
        args.job_name = f"rl-iris-{time.strftime('%Y%m%d-%H%M%S')}"

    # Load --secrets-env into os.environ on the launch host (so launch-host
    # hooks see it) AND collect them for injection into the task. Reuse the
    # (file overrides shell; same semantics as the OT-Agent iris launchers).
    load_secrets_env_into_os_environ(args.secrets_env)

    # Daytona org re-route (robust). load_secrets_env_into_os_environ() above does
    # "file overrides shell" (hpc/iris/env.py) — so a pre-launch `export
    # DAYTONA_API_KEY="$DAYTONA_RL_API_KEY"` is CLOBBERED by secrets.env's main-org
    # value, which is then what the passthrough (below) forwards to the pod. To route
    # onto the dedicated RL Daytona org we must remap DAYTONA_API_KEY *after* the
    # re-source, referencing the source var by NAME only (no key value in code/CLI).
    override_src = getattr(args, "daytona_api_key_env", None)
    if override_src:
        override_val = os.environ.get(override_src)
        if not override_val:
            raise SystemExit(
                f"[rl-iris] --daytona-api-key-env={override_src} but that env var is "
                f"empty/unset. `source {args.secrets_env}` first; it must define {override_src}."
            )
        os.environ["DAYTONA_API_KEY"] = override_val
        _fp = hashlib.sha1(override_val.encode()).hexdigest()[:12]
        print(
            f"[rl-iris] Daytona re-route: DAYTONA_API_KEY <- ${override_src} (sha1={_fp})",
            flush=True,
        )

    command = build_task_command(args)

    # Per-task resources: a WHOLE node (8 H100 + IB), one task per node.
    gpu_spec = f"{args.gpu_variant}x{args.gpus_per_node}"

    # Resolve the "auto" disk default to ~80% of the node's live allocatable ephemeral-storage.
    # (Whole-node-exclusive gangs have no co-tenants → reserving disk is wasted; a too-low fixed
    # default evicted long MoE steps once Ray spilled to the metered /tmp. See _resolve_default_disk.)
    if str(args.disk).strip().lower() == "auto":
        args.disk = _resolve_default_disk()

    user = os.environ.get("USER") or os.environ.get("USERNAME") or "user"
    print(f"[rl-iris] Job:        /{user}/{args.job_name}", flush=True)
    print(f"[rl-iris] Cluster:    {args.cluster}  ({args.cluster_config})", flush=True)
    print(f"[rl-iris] Image:      {args.task_image}", flush=True)
    print(
        f"[rl-iris] Topology:   {args.num_nodes} node(s) x {gpu_spec}  "
        f"(= {args.num_nodes * args.gpus_per_node} GPUs, exclusive, gang/leafgroup)",
        flush=True,
    )
    print(f"[rl-iris] Per node:   cpu={args.cpu} memory={args.memory} disk={args.disk}", flush=True)
    print(f"[rl-iris] Priority:   {args.priority}", flush=True)
    print(f"[rl-iris] RL config:  {args.rl_config}  model={args.model_path}", flush=True)
    # Surface the resolved SKYRL_* runtime-knob flag env here (before the --dry-run
    # return) so a dry-run confirms e.g. --collective-count-diag actually resolves.
    # This is display-only; main() re-derives it (idempotent, pure fn of args) below.
    _flag_env_preview = build_skyrl_flag_env(args)
    if _flag_env_preview:
        print(
            f"[rl-iris] SKYRL flag env: {', '.join(f'{k}={v}' for k, v in sorted(_flag_env_preview.items()))}",
            flush=True,
        )
    if args.num_nodes > 1:
        print(f"[rl-iris] Rendezvous: {args.rendezvous_dir}", flush=True)
    print(f"[rl-iris] Command:    {shlex.join(command)}", flush=True)

    if args.dry_run:
        print("[rl-iris] --dry-run: not submitting", flush=True)
        return 0

    # Defer heavy iris imports so --dry-run / --help stay snappy.
    #
    # NOTE: post iris PR #6652 (pydantic config parsing) + #6730 (multi-backend
    # controller) the old submit API moved. The config is now a pydantic
    # ``IrisClusterConfig`` loaded via the MODULE-LEVEL ``load_config(path)``
    # (the ``IrisConfig`` class + its ``.load()`` / ``.provider_bundle()`` /
    # ``.proto`` are gone). The provider bundle is now built by the module-level
    # ``iris.cluster.composer.provider_bundle(config)``, and ``LocalCluster``
    # moved to ``iris.cluster.local_cluster``. The job-build helpers
    # (build_resources / build_job_constraints / resolve_multinode_defaults /
    # EnvironmentSpec / Entrypoint / job_pb2) and the ``IrisClient.remote(...)`` /
    # ``client.submit(...)`` surface are UNCHANGED — see how the marin CLI itself
    # now submits in iris/cli/job.py + iris/cli/connect.py, which this mirrors.
    from iris.client import IrisClient
    from iris.cluster.config import load_config
    from iris.cluster.composer import provider_bundle
    from iris.cluster.local_cluster import LocalCluster
    from iris.cluster.types import EnvironmentSpec, Entrypoint
    from iris.cli.job import build_resources, build_job_constraints, resolve_multinode_defaults
    from iris.rpc import job_pb2

    # Per-task resources: whole node, all GPUs (no co-tenant → exclusive).
    resources = build_resources(None, gpu_spec, cpu=args.cpu, memory=args.memory, disk=args.disk)

    # Multi-node gang: replicas=num_nodes; for GPUs with replicas>1 this returns
    # CoschedulingConfig(group_by="leafgroup") — co-schedule all nodes on one IB
    # leaf fabric, atomically (Kueue gang admission on cw-us-east-02a).
    replicas, coscheduling = resolve_multinode_defaults(None, args.gpu_variant, args.num_nodes)

    resources_proto = resources.to_proto()
    constraints = build_job_constraints(
        resources_proto=resources_proto,
        tpu_variants=[],
        replicas=replicas,
        regions=None,
        zone=None,
        preemptible=args.preemptible,
    )

    priority_band = {
        "production": job_pb2.PRIORITY_BAND_PRODUCTION,
        "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE,
        "batch": job_pb2.PRIORITY_BAND_BATCH,
    }.get(args.priority, job_pb2.PRIORITY_BAND_UNSPECIFIED)

    # Env: secrets file values + the standard RL/iris-serve signals. iris injects
    # IRIS_TASK_ID / IRIS_NUM_TASKS / IRIS_ADVERTISE_HOST per task automatically.
    env_vars: dict[str, str] = {}
    # MarinSkyRL runtime-knob flags (deslop stage 3) -> SKYRL_* env vars. Seeded
    # FIRST (below the config extra_env) so a config's explicit extra_env value still
    # OVERRIDES a flag; an all-defaults launch contributes {} (byte-identical).
    flag_env = build_skyrl_flag_env(args)
    if flag_env:
        env_vars.update(flag_env)
        print(f"[rl-iris] SKYRL flag env: {', '.join(f'{k}={v}' for k, v in sorted(flag_env.items()))}", flush=True)
    # Forward the RL config YAML's top-level `extra_env:` block (the Iris analog of
    # the SLURM container.extra_env exports — see load_config_extra_env). Overlaid
    # ON TOP of the flag env so an explicit config value wins; the launcher's own
    # signals (rendezvous/secrets, below) then win over both on any collision.
    config_extra_env = load_config_extra_env(args.rl_config)
    if config_extra_env:
        env_vars.update(config_extra_env)
        print(f"[rl-iris] Config extra_env: {', '.join(sorted(config_extra_env))}", flush=True)
    # FR-slug fix: a config may hardcode a JOB-SCOPED NCCL flight-recorder dump path
    # (/tmp/fr_dumps/<slug>/nccl_fr_rank) with a STALE slug from the config it was
    # copied from. Re-scope the slug to the live --job-name so a future FR dump lands
    # under the right per-job dir (the controller mkdir -p's the cvar's dirname). No-op
    # for the bare generic /tmp/nccl_fr_rank path every non-80B config uses.
    for _fr_cvar in ("TORCH_NCCL_DEBUG_INFO_TEMP_FILE", "TORCH_FR_DUMP_TEMP_FILE"):
        _old = env_vars.get(_fr_cvar)
        if _old:
            _new = _job_scope_fr_dump_path(_old, args.job_name)
            if _new != _old:
                env_vars[_fr_cvar] = _new
                print(f"[rl-iris] FR-slug re-scope: {_fr_cvar} {_old} -> {_new}", flush=True)
    if args.rendezvous_dir:
        env_vars["OT_AGENT_IRIS_RENDEZVOUS_DIR"] = args.rendezvous_dir
    env_vars["OT_AGENT_IRIS_RAY_PORT"] = str(args.ray_port)
    # Forward the launch host's secrets (mirrors launch_eval_iris.py passthrough).
    #
    # IMPORTANT — do NOT forward AWS_*/R2_* here. The cw-us-east-02a cluster
    # projects an `iris-task-env` k8s Secret into EVERY task pod via `envFrom`
    # (because storage.remote_state_dir is an s3:// URI), and that secret already
    # carries the correct in-cluster R2 credentials + endpoint
    # (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_ENDPOINT_URL / AWS_REGION /
    # FSSPEC_S3). In K8s, explicit container `env` entries take precedence over
    # `envFrom`, so forwarding the launch host's AWS_* (which point at a
    # DIFFERENT account and lack AWS_ENDPOINT_URL) would CLOBBER the pod's
    # injected creds and make the s3://marin-us-east-02a rendezvous (multi-node)
    # silently target real AWS S3 instead of the cluster store. NOTE: the default
    # object store moved R2 (s3://marin-na) -> CW (s3://marin-us-east-02a) on
    # 2026-07-05 (marin c7caecc95a) — pods now inject CW creds+AWS_ENDPOINT_URL and
    # can no longer reach R2. Let the cluster-injected creds win; the
    # fsspec rendezvous in start_rl_iris_controller.py uses default credential
    # discovery and picks them up.
    #
    # Daytona credentials MUST be forwarded: agentic RL (terminal_bench / Harbor)
    # builds a Daytona sandbox per trial, and iris injects only HF/WANDB into the
    # task pod — nothing else. Without DAYTONA_API_KEY the worker's harbor client
    # raises DaytonaAuthenticationError on every env build, so no sandbox comes
    # up, the verifier never runs, and EVERY trajectory finalizes as
    # VerificationNotCompletedError with reward 0 (observed zeroing an entire
    # reverify rollout). Mirror the base IrisLauncher passthrough set
    # so the same creds reach the RL worker.
    #
    # WANDB routing default: the iris RL configs log to wandb (trainer.logger: wandb;
    # CoreWeave has egress). SkyRL's wandb.init passes project= but NOT entity=
    # (MarinSkyRL tracking.py), so without WANDB_ENTITY the run silently lands in the
    # API key's DEFAULT entity (e.g. nyu-dice-lab), not the team org. Default both to
    # the OT-Agent team here so every run lands in
    # dogml/OpenThoughts-Agent; an explicitly-set launch-host WANDB_ENTITY/PROJECT wins.
    os.environ.setdefault("WANDB_ENTITY", "dogml")
    os.environ.setdefault("WANDB_PROJECT", "OpenThoughts-Agent")
    for k in (
        "HF_TOKEN",
        "WANDB_API_KEY",
        "WANDB_ENTITY",
        "WANDB_PROJECT",
        "DAYTONA_API_KEY",
        "DAYTONA_JWT_TOKEN",
        "DAYTONA_ORGANIZATION_ID",
        "DAYTONA_API_URL",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "TOGETHER_API_KEY",
    ):
        v = os.environ.get(k)
        if v:
            env_vars[k] = v

    # Load the cluster config (pydantic IrisClusterConfig) and build the provider
    # bundle, then discover + tunnel to the controller. This mirrors the marin
    # CLI's own path (iris/cli/connect.py::require_controller_url): for a local
    # controller start an in-process LocalCluster; otherwise use the config's
    # controller_address() (defaults.worker.controller_address) if set, else fall
    # back to the backend's discover_controller(). cw-us-east-02a's controller
    # kind is "coreweave" (non-local, no IAP auth) → the discover path.
    iris_config = load_config(args.cluster_config)
    bundle = provider_bundle(iris_config)
    if iris_config.controller.controller_kind() == "local":
        controller_address = LocalCluster(iris_config).start()
    else:
        controller_address = iris_config.controller_address() or bundle.controller.discover_controller(
            iris_config.controller
        )

    with bundle.controller.tunnel(address=controller_address) as controller_url:
        client = IrisClient.remote(controller_url, workspace=PROJECT_ROOT)
        entrypoint = Entrypoint.from_command(*command)
        job = client.submit(
            entrypoint=entrypoint,
            name=args.job_name,
            resources=resources,
            environment=EnvironmentSpec(env_vars=env_vars, extras=[]),
            constraints=constraints or None,
            coscheduling=coscheduling,
            replicas=replicas,
            max_retries_failure=args.max_retries,
            task_image=args.task_image,
            priority_band=priority_band,
            timeout=None if args.timeout == 0 else _seconds_to_duration(args.timeout),
        )
        full_job_id = str(job.job_id)
        print(
            f"[rl-iris] Submitted: {full_job_id}  (replicas={replicas}, "
            f"coscheduling={getattr(coscheduling, 'group_by', None)})",
            flush=True,
        )

        if args.no_wait:
            return 0
        try:
            status = job.wait(stream_logs=True, timeout=float("inf"))
            exit_code = 0 if status.state == job_pb2.JOB_STATE_SUCCEEDED else 1
        except KeyboardInterrupt:
            print(f"[rl-iris] Terminating job {full_job_id}...", file=sys.stderr, flush=True)
            client.terminate_job(job.job_id)
            exit_code = 130
        print(f"[rl-iris] Job exit: {exit_code}", flush=True)
        return exit_code


def _seconds_to_duration(secs: int):
    from rigging.timing import Duration

    return Duration.from_seconds(secs)


if __name__ == "__main__":
    sys.exit(main())
