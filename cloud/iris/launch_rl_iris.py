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
import json
import os
import re
import secrets
import shlex
import sys
import time
from pathlib import Path
from typing import Any, List, Optional
from urllib.parse import urlparse

import yaml

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
    # HARBOR_CHEAP_REAPER=0) + skyrl 3caeb79f (TIS served-id splice, now baked-default — no runtime source
    # override needed for 35B). Same PULLABLE recipe; vLLM-fork 76259c63, flash-attn 2.8.3, torch 2.11.0+cu128 unchanged.
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
    # (already-validated; previously runtime-only via a source override, now baked-default). Parent is exactly
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
    # (penfever/working HEAD, direct child: CP>1 _C::rms_norm Meta-kernel fix) so a runtime 272bf011 pin is a
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
    # gpu-rl-f4f25bae (built 2026-07-17, kaniko job gpurl-kaniko-f4f25bae, SINGLE_SNAPSHOT=0 pullable):
    # HARBOR_COMMIT bump to c872216e (harbor main HEAD = opencode RL literal bridge: HARBOR_MODEL_ENDPOINT
    # baseURL fix + correlated rollout_details). Baking it lets future RL launches drop --harbor-ref main.
    # ALSO adds boto3 + smart_open to skyrl-train deps (the Dockerfile build assert required them but they
    # were only transitives of litellm via harbor, installed after the assert). SKYRL 2861eaef (cu128 lock,
    # PR #19). wheels UNCHANGED (fast prebuilt-wheelhouse, NO nvcc). Pull-verified: 32 layers, max 6.56 GB.
    # Build asserts green (flash_attn_2_cuda / torch 2.11.0+cu128 / vllm / skyrl_train / torchtitan
    # ExpertParallel / boto3 import OK). baked harbor c872216e.
    # gpu-rl-megatron-a1e7a363 (built 2026-07-20, kaniko job gpurl-kaniko-a1e7a363, INSTALL_MEGATRON=1,
    # SINGLE_SNAPSHOT=0 pullable): HARBOR_COMMIT bump to 5efac6fa (harbor main HEAD = the Daytona keepalive,
    # PR #19 — background refresh_activity() so opencode trials don't idle-reap at auto_stop=30). Baking it
    # lets ALL RL launches drop --harbor-ref entirely (no-refs policy: a merged branch auto-deletes, and the
    # runtime uv reinstall then dies state-5 at bring-up — this happened to the keep6 -e1 pair). Megatron
    # variant = the weight-sync het-bootstrap fix (PR #71, 79432f4a) + all d0016149 contents. HARBOR_COMMIT
    # plumbed through build_gpu_rl_kaniko.sh via MarinSkyRL PR #74. Pull-verified: 35 layers, max 7.48 GB.
    # gpu-rl-megatron-9c17f8a4 (built 2026-07-22, kaniko job gpurl-kaniko-9c17f8a4,
    # INSTALL_MEGATRON=1): baked MarinSkyRL main 9c17f8a4, including the
    # ChunkedDistributedLogprob preallocated-backward fix for packed-token OOMs, plus
    # canonical Harbor 394c58fe. Registry-verified: 37 layers, max 2.80 GiB, 18.43 GiB
    # total; it retains the proven cache-free split-layer pullability profile for RNO.
    "@sha256:3ed8480f725579d6ce88086ec56838f6f39b169bab19245ccc42a09d5b61d93e"  # noqa: E501
    # (prev: gpu-rl-megatron-b063514b @sha256:6c2c0041, same Harbor)
    # (prev: gpu-rl-megatron-a1e7a363 @sha256:570e9cc1, Harbor 5efac6fa)
    # (prev: gpu-rl-f4f25bae @sha256:7bbc17b6 harbor c872216e literal bridge; gpu-rl-e03896b7 @sha256:e8b48241b harbor f4a6b1a0 round-6 orjson parse-offload; gpu-rl-b397b82a @sha256:bac11e44 harbor 101b1400 round-5; gpu-rl-d0e4a9b8 @sha256:0fbf41e5 harbor d81b2f32 round-4; gpu-rl-f9110c79 @sha256:5e211fbf harbor 35fbdbcc round-3; gpu-rl-318e18ce @sha256:35fbf815 harbor 793ff3fb round-2)
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
# A whole H100 node exposes 128 CPUs in the Iris cluster configuration, but the
# CoreWeave scheduler leaves daemonset headroom. Requests above 48 have proven
# unplaceable, so use the cluster's advertised capacity only to reduce this
# safe ceiling for smaller future node types.
MAX_DEFAULT_CPU_PER_NODE = 48.0
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

# marin-iris wheel installed into the RL venv at pod bootstrap for the controller-ingress
# registration path (GAP D). The gpu-rl image bakes ONLY MarinSkyRL + harbor, never iris (a
# marin-monorepo pkg), so cloud.iris.ingress_utils' `import iris.cluster.client.* / iris.rpc.*`
# would ModuleNotFoundError in driver init. This dev build is validated against the live
# marin controller's registration/mint RPC protocol and parses the current parent marin.yaml
# `platform.gcp.registry_mirrors` field; override via --iris-ref.
DEFAULT_IRIS_VERSION = "marin-iris==0.2.54.dev202607210800"

MARIN_LOGIN_RECORD_PATH = Path.home() / ".config" / "marin" / "credentials" / "marin.json"
_JOB_NAME_MAX_LENGTH = 63


def _resolve_cluster_config_default(cluster: str = DEFAULT_CLUSTER) -> str:
    """Find the marin repo's ``<cluster>.yaml`` iris cluster config.

    ``cluster`` selects the CoreWeave cluster (e.g. ``cw-us-east-02a`` = 256×H100 default,
    ``cw-rno2a`` = the 512×H100 RNO2A cluster for the delphi pilot). Falls back to the
    bare relative path if no marin checkout is found (the caller can still pass an explicit
    ``--cluster-config``)."""
    rel = f"lib/iris/config/{cluster}.yaml"
    candidates = [
        Path.home() / "Documents/marin" / rel,
        Path("/Users/benjaminfeuer/Documents/marin") / rel,
        Path(os.environ.get("MARIN_ROOT", "")) / rel,
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return rel


def _resolve_parent_cluster_config(cluster_config: Optional[str]) -> Optional[str]:
    """Path to the PARENT (marin) cluster YAML for federated submission.

    The marin meta-scheduler config (marin.yaml) that owns iris.oa.dev and lists the
    CoreWeave clusters as delegation peers. Defaults to the ``marin.yaml`` sibling of
    ``--cluster-config`` (they live in the same ``lib/iris/config/`` dir); falls back
    to the same search roots as :func:`_resolve_cluster_config_default`.
    """
    if cluster_config:
        sib = Path(cluster_config).with_name("marin.yaml")
        if sib.exists():
            return str(sib)
    rel = "lib/iris/config/marin.yaml"
    for c in (
        Path.home() / "Documents/marin" / rel,
        Path("/Users/benjaminfeuer/Documents/marin") / rel,
        Path(os.environ.get("MARIN_ROOT", "")) / rel,
    ):
        if c.exists():
            return str(c)
    return None


def _load_cluster_config(cluster_config: str) -> dict[str, Any]:
    """Load the selected Iris cluster configuration for launch-time defaults."""
    try:
        with open(cluster_config) as f:
            loaded = yaml.safe_load(f)
    except OSError as exc:
        raise SystemExit(
            f"Could not load --cluster-config {cluster_config!r} to resolve RL launch defaults: {exc}. "
            "Pass an existing cluster config or explicit --cpu and --rendezvous-dir values."
        ) from exc
    if not isinstance(loaded, dict):
        raise SystemExit(f"--cluster-config {cluster_config!r} must contain a YAML mapping.")
    return loaded


def _cluster_storage_root(cluster_config: dict[str, Any]) -> str:
    """Return the durable object-store root containing the Iris controller state."""
    storage = cluster_config.get("storage")
    remote_state_dir = storage.get("remote_state_dir") if isinstance(storage, dict) else None
    if not isinstance(remote_state_dir, str) or not remote_state_dir.startswith(("s3://", "gs://")):
        raise SystemExit(
            "The selected Iris cluster config needs storage.remote_state_dir set to an s3:// or gs:// URI "
            "to derive --rendezvous-dir; pass --rendezvous-dir explicitly."
        )
    return remote_state_dir.rstrip("/").rsplit("/", 1)[0]


def _cluster_gpu_cpu_capacity(cluster_config: dict[str, Any], *, gpu_variant: str, gpus_per_node: int) -> float:
    """Return CPU capacity for the matching GPU scale group in an Iris config."""
    scale_groups = cluster_config.get("scale_groups")
    if not isinstance(scale_groups, dict):
        raise SystemExit("The selected Iris cluster config has no scale_groups mapping; pass --cpu explicitly.")
    for scale_group in scale_groups.values():
        resources = scale_group.get("resources") if isinstance(scale_group, dict) else None
        if not isinstance(resources, dict):
            continue
        if (
            resources.get("device_type") == "gpu"
            and str(resources.get("device_variant", "")).lower() == gpu_variant.lower()
            and resources.get("device_count") == gpus_per_node
        ):
            cpu = resources.get("cpu")
            if isinstance(cpu, (int, float)) and cpu > 0:
                return float(cpu)
    raise SystemExit(
        f"The selected Iris cluster config has no {gpus_per_node}x{gpu_variant} GPU scale group; pass --cpu explicitly."
    )


def _rl_config_harness_name(rl_config: str) -> Optional[str]:
    """Read the configured Harbor harness name without constructing trainer state."""
    try:
        with open(rl_config) as f:
            config = yaml.safe_load(f) or {}
    except OSError:
        return None
    if not isinstance(config, dict):
        return None

    candidate_paths = (
        ("terminal_bench_config", "harbor", "name"),
        ("terminal_bench", "harbor", "name"),
        ("generator", "harbor", "harness", "name"),
        ("generator", "harbor", "name"),
    )
    for path in candidate_paths:
        current: Any = config
        for key in path:
            if not isinstance(current, dict):
                break
            current = current.get(key)
        else:
            if isinstance(current, str) and current.strip():
                return current.strip().lower()
    return None


def _sanitize_job_name_component(value: str) -> str:
    """Make one human-readable Kubernetes job-name component."""
    value = value.strip().rstrip("/").split("/")[-1]
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return value or "run"


def derive_default_job_name(
    args: argparse.Namespace,
    *,
    timestamp: Optional[str] = None,
    nonce: Optional[str] = None,
) -> str:
    """Build a unique, valid Iris job name from the selected RL config and model."""
    config_name = _sanitize_job_name_component(Path(args.rl_config).stem)
    model_name = _sanitize_job_name_component(args.model_path)
    timestamp = timestamp or time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    nonce = nonce or secrets.token_hex(3)
    suffix = f"-{timestamp}-{nonce}"
    prefix = f"rl-{config_name}-{model_name}"
    return f"{prefix[: _JOB_NAME_MAX_LENGTH - len(suffix)].rstrip('-')}{suffix}"


def resolve_launch_defaults(args: argparse.Namespace) -> None:
    """Resolve cluster-dependent and harness-dependent defaults before validation."""
    if not args.job_name:
        args.job_name = derive_default_job_name(args)

    needs_cluster_config = args.cpu is None or (args.num_nodes > 1 and not args.rendezvous_dir)
    cluster_config = _load_cluster_config(args.cluster_config) if needs_cluster_config else None

    if args.cpu is None:
        assert cluster_config is not None
        capacity = _cluster_gpu_cpu_capacity(
            cluster_config,
            gpu_variant=args.gpu_variant,
            gpus_per_node=args.gpus_per_node,
        )
        args.cpu = min(capacity, MAX_DEFAULT_CPU_PER_NODE)

    if args.num_nodes > 1 and not args.rendezvous_dir:
        assert cluster_config is not None
        storage_root = _cluster_storage_root(cluster_config)
        args.rendezvous_dir = f"{storage_root}/rendezvous/{args.job_name}"

    if args.record_literal is None:
        harness = _rl_config_harness_name(args.rl_config)
        args.record_literal = harness is None or harness.replace("_", "-") != "terminus-2"


def _cluster_dashboard_host(cluster_config_path: Optional[str]) -> Optional[str]:
    """Bare host of a cluster config's ``dashboard_url`` — the public host of the
    controller that OWNS endpoints registered on that cluster. None if unreadable."""
    if not cluster_config_path:
        return None
    try:
        import yaml
        from urllib.parse import urlparse

        with open(cluster_config_path) as f:
            raw = yaml.safe_load(f) or {}
        url = raw.get("dashboard_url")
        return urlparse(url).hostname if url else None
    except Exception:  # noqa: BLE001
        return None


def _rl_config_is_agentic(rl_config: Optional[str]) -> bool:
    """True when the rl_config drives an in-sandbox agent (opencode/harbor/terminal_bench)
    that must call BACK to the served model. Best-effort text scan."""
    try:
        if not rl_config or not os.path.isfile(rl_config):
            return False
        with open(rl_config, "r") as f:
            text = f.read().lower()
        return any(k in text for k in ("terminal_bench", "harbor", "opencode"))
    except OSError:
        return False


def _rl_config_needs_controller_ingress(rl_config: Optional[str]) -> bool:
    """True ONLY for the OPENCODE harness, which needs the co-located RecordProxy literal
    bridge (token-id/logprob capture for TIS) + the cross-cluster ``/proxy/t`` capability
    URL. Other agentic harnesses (terminus-2) do NOT: they call the served model over the
    DIRECT marinskyrl HTTP endpoint (``ingress_mode=direct``), the historical path, and
    must NOT be force-routed through controller-ingress (the federated ``/proxy/t`` path
    breaks non-streaming terminus-2 -> upstream/stream timeout). Detect the ACTIVE harbor
    harness (``name: opencode``), not mere presence of harbor/terminal_bench blocks."""
    try:
        if not rl_config or not os.path.isfile(rl_config):
            return False
        with open(rl_config, "r") as f:
            text = f.read().lower()
        # Active harness declared as `name: opencode` in the harbor block (ignore comments).
        return any(line.strip().startswith("name:") and "opencode" in line for line in text.splitlines())
    except OSError:
        return False


def autoconfigure_ingress(args: argparse.Namespace) -> None:
    """Derive the controller-ingress config from the target cluster so an agentic CoreWeave
    launch JUST WORKS from ``--target-cluster`` alone — no manual ``--ingress-mode`` /
    ``--ingress-host``.

    Rationale: controller-ingress is required ONLY for the OPENCODE harness (the co-located
    RecordProxy literal bridge + the cross-cluster ``/proxy/t`` capability URL). It is NOT
    the only reachable topology on CoreWeave — the terminus-2 harness reaches the served
    model over the DIRECT marinskyrl HTTP endpoint (``ingress_mode=direct``, the historical
    path) and MUST NOT be force-routed through controller-ingress (federated ``/proxy/t``
    breaks non-streaming terminus-2). So we auto-enable controller ONLY for opencode; for
    that case the ingress host is cluster-determined (``iris.oa.dev``), removing the
    ``--ingress-host`` mismatch error class. Prefer default > flag > env var."""
    target = str(getattr(args, "target_cluster", "") or "")
    cluster = str(getattr(args, "cluster", "") or "")
    is_cw = target.startswith("cw-") or cluster.startswith("cw-")
    # Resolve the "auto" sentinel (the default). An EXPLICIT --ingress-mode direct|controller
    # is ALWAYS honored (explicit flag beats derivation); only "auto" is derived here.
    mode = getattr(args, "ingress_mode", "auto")
    if mode == "auto":
        # Auto-enable controller ONLY for an opencode rl_config on CoreWeave (needs the
        # literal bridge + /proxy/t). terminus-2 & everything else -> direct (marinskyrl HTTP).
        if is_cw and _rl_config_needs_controller_ingress(getattr(args, "rl_config", None)):
            args.ingress_mode = "controller"
            print(
                "[rl-iris] auto: --ingress-mode=controller (opencode rl_config on a CoreWeave target)",
                flush=True,
            )
        else:
            args.ingress_mode = "direct"
    if not is_cw:
        return  # non-CoreWeave: host derivation below is controller/CoreWeave-only
    # (2) The ingress host is cluster-determined on CoreWeave: the marin parent iris.oa.dev.
    if getattr(args, "ingress_mode", "direct") == "controller":
        prev = getattr(args, "ingress_host", None)
        if prev not in (None, "", "iris.oa.dev"):
            print(
                f"[rl-iris] auto: overriding --ingress-host {prev} -> iris.oa.dev "
                "(CoreWeave federated parent; the host is cluster-determined, not a free choice)",
                flush=True,
            )
        elif prev is None:
            print(
                "[rl-iris] auto: --ingress-host=iris.oa.dev (derived from --target-cluster; "
                "CoreWeave federated parent)",
                flush=True,
            )
        args.ingress_host = "iris.oa.dev"


def validate_controller_ingress_reachability(args: argparse.Namespace) -> None:
    """Fail loud BEFORE submit when ``--ingress-mode controller`` would produce a
    capability URL a Daytona sandbox CANNOT reach — the Exp2 opencode-RL blocker
    (ported from OT-Agent 8fdabb12, extended for the federated remediation).

    opencode runs in a Daytona sandbox and reaches the co-located vLLM over the public
    internet at ``https://<ingress_host>/proxy/t/<token>/<endpoint>/v1``. The endpoint
    is REGISTERED on the controller of the cluster the job runs on and the token is
    minted with that controller's key, so the capability URL only resolves when
    ``<ingress_host>`` is a controller that can BOTH route to the endpoint AND be
    reached from Daytona:

      * A **directly-submitted CoreWeave** job cannot: the peer controller's own host
        (``dashboard_url``, e.g. ``iris-cw-us-east-02a.oa.dev``) is IP-locked to the
        marin egress; and iris.oa.dev (marin) only FEDERATES ``/proxy`` to a CoreWeave
        endpoint for a job it DELEGATED. A direct submit → iris.oa.dev has no route →
        404 → opencode never reaches vLLM → RecordProxy captures 0 traffic, the job
        burns an H100 node making 0 trials.
      * The **federated** path (``--target-cluster <peer>``) fixes it: marin delegates
        the job to the peer child, so ``has_received_job_from_peer`` passes and marin
        federation-proxies ``/proxy``. The endpoint is registered on the peer AND
        MIRRORED onto marin by FederationSync; the capability token is minted at the
        PARENT (iris.oa.dev) for the mirrored endpoint. So controller-ingress on
        CoreWeave is ALLOWED iff ``--target-cluster`` is set and ``--ingress-host`` is
        the marin host.

    Escape hatch (once a further remediation is wired): ``OTAGENT_ALLOW_INGRESS_HOST_MISMATCH=1``.
    """
    if getattr(args, "ingress_mode", "direct") != "controller":
        return
    if os.environ.get("OTAGENT_ALLOW_INGRESS_HOST_MISMATCH") == "1":
        print(
            "[rl-iris] WARNING: OTAGENT_ALLOW_INGRESS_HOST_MISMATCH=1 — skipping the "
            "controller-ingress reachability guard.",
            flush=True,
        )
        return

    cluster = str(getattr(args, "cluster", "") or "")
    ingress_host = str(getattr(args, "ingress_host", "") or "")
    target_cluster = str(getattr(args, "target_cluster", "") or "")
    dash_host = _cluster_dashboard_host(getattr(args, "cluster_config", None))
    is_coreweave = cluster.startswith("cw-") or (dash_host or "") not in ("", "iris.oa.dev")

    if is_coreweave:
        # The ONLY reachable CoreWeave topology: federated submission through marin.
        if not target_cluster:
            raise SystemExit(
                "[rl-iris] BLOCKED: --ingress-mode controller on a directly-submitted "
                f"CoreWeave job (--cluster={cluster or '?'}, controller host="
                f"{dash_host or '?'}) is NOT reachable from a Daytona sandbox.\n"
                "  The capability URL would 404: iris.oa.dev only federates /proxy for a "
                "job it DELEGATED, and the CoreWeave controller's own host is IP-locked. "
                "opencode would never reach vLLM (0 trials, RecordProxy captures nothing) "
                "— the 2026-07-16 Exp2 blocker.\n"
                "  Fix: pass --target-cluster " + (cluster or "<peer>") + " to federate "
                "the job through the marin meta-scheduler (keep --ingress-host iris.oa.dev), "
                "so marin delegates it to the peer and federation-proxies /proxy.\n"
                "  Override (only once another remediation is wired): "
                "OTAGENT_ALLOW_INGRESS_HOST_MISMATCH=1."
            )
        if ingress_host and ingress_host != "iris.oa.dev":
            raise SystemExit(
                f"[rl-iris] BLOCKED: federated CoreWeave controller-ingress needs "
                f"--ingress-host iris.oa.dev (the marin parent that owns the mirrored "
                f"endpoint + signs the token), got --ingress-host {ingress_host}. A "
                "peer-signed token 401s at iris.oa.dev (federation trust is "
                "unidirectional: cw trusts marin, not the reverse)."
            )
        return
    # Non-CoreWeave (e.g. a marin-local submission): the host must match the controller
    # that owns the endpoint.
    if ingress_host and dash_host and ingress_host != dash_host and not target_cluster:
        raise SystemExit(
            f"[rl-iris] BLOCKED: --ingress-host {ingress_host} does not match this "
            f"cluster's controller host {dash_host} (--cluster={cluster}). Override with "
            "OTAGENT_ALLOW_INGRESS_HOST_MISMATCH=1."
        )


def prepare_federated_parent_credentials(args: argparse.Namespace) -> str | None:
    """Validate and return the cached Marin IAP login needed by a federated pod.

    A CoreWeave task has neither a cached human login nor a Marin-allowlisted service
    account. Controller ingress therefore cannot mint a parent capability token unless
    the launcher forwards the operator's cached Marin IAP login record. Mint an IAP
    token here, before submitting or allocating GPUs, so a stale or absent record fails
    locally instead of after the endpoint-registration wait in the task.
    """
    if not getattr(args, "target_cluster", None) or getattr(args, "ingress_mode", "direct") != "controller":
        return None
    if not MARIN_LOGIN_RECORD_PATH.is_file():
        raise SystemExit(
            "[rl-iris] BLOCKED: federated CoreWeave controller ingress requires the cached "
            f"Marin IAP login record at {MARIN_LOGIN_RECORD_PATH}. "
            "Run `iris --cluster=marin login` and relaunch."
        )
    try:
        record = json.loads(MARIN_LOGIN_RECORD_PATH.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"[rl-iris] BLOCKED: {MARIN_LOGIN_RECORD_PATH} is not valid JSON. "
            "Run `iris --cluster=marin login` and relaunch."
        ) from exc

    if record.get("cluster") != "marin" or urlparse(str(record.get("endpoint", ""))).hostname != "iris.oa.dev":
        raise SystemExit(
            f"[rl-iris] BLOCKED: {MARIN_LOGIN_RECORD_PATH} is not a Marin iris.oa.dev login record. "
            "Run `iris --cluster=marin login` and relaunch."
        )
    refresh_token = record.get("edge_refresh_token")
    if not isinstance(refresh_token, str) or not refresh_token:
        raise SystemExit(
            f"[rl-iris] BLOCKED: {MARIN_LOGIN_RECORD_PATH} has no edge_refresh_token. "
            "Run `iris --cluster=marin login` and relaunch."
        )

    from rigging.auth import IapRefreshTokenProvider, MARIN_DESKTOP_OAUTH_CLIENT

    provider = IapRefreshTokenProvider(
        MARIN_DESKTOP_OAUTH_CLIENT.client_id,
        MARIN_DESKTOP_OAUTH_CLIENT.client_secret,
        refresh_token,
        login_hint="log in to cluster 'marin' to authenticate",
    )
    try:
        token = provider.get_token()
    except Exception as exc:
        raise SystemExit(
            "[rl-iris] BLOCKED: unable to mint an IAP token from the cached Marin login record. "
            "Run `iris --cluster=marin login` and relaunch."
        ) from exc
    if not token:
        raise SystemExit(
            "[rl-iris] BLOCKED: cached Marin login did not mint an IAP token. "
            "Run `iris --cluster=marin login` and relaunch."
        )
    print(
        "[rl-iris] Federated parent-IAP preflight passed; forwarding the cached Marin login record to the peer task.",
        flush=True,
    )
    return json.dumps(record)


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
        default=None,
        help="CPU cores per node. Default: derive from the selected GPU cluster's scale group "
        f"with a scheduling-safe cap of {MAX_DEFAULT_CPU_PER_NODE:g}.",
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
        default=None,
        help="Path to the iris cluster YAML. Default: auto-resolve lib/iris/config/"
        "<--cluster>.yaml in the marin repo (so --cluster cw-rno2a targets the 512xH100 "
        "RNO2A cluster without a manual --cluster-config).",
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
    # ----------------------------------------------------------------------- #
    # Cross-cluster ingress / federated submission (Exp2 opencode-RL fix #1).   #
    #                                                                           #
    # The default (direct) path is UNCHANGED: submit straight to --cluster's    #
    # own controller (byte-identical to before). The federated path is opt-in   #
    # via --target-cluster: submit through the marin meta-scheduler             #
    # (iris.oa.dev) with a `cluster EQ <peer>` constraint so marin DELEGATES    #
    # the whole job to the peer child and can then federation-proxy /proxy      #
    # requests to the peer's endpoint. This is the ONLY topology in which a     #
    # Daytona sandbox can reach a co-located CoreWeave vLLM through a single    #
    # public host (iris.oa.dev): the peer controller's own host is IP-locked    #
    # with no off-cluster surface, and marin only federates /proxy for a job it #
    # delegated (controller has_received_job_from_peer). See                    #
    # validate_controller_ingress_reachability() + .claude/ops/iris/iris_ingress.md. #
    # ----------------------------------------------------------------------- #
    parser.add_argument(
        "--ingress-mode",
        "--ingress_mode",
        dest="ingress_mode",
        default="auto",
        choices=["auto", "direct", "controller"],
        help="How the co-located served model (RecordProxy/vLLM) is exposed to a "
        "Daytona sandbox. 'auto' (default) = derive per harness (opencode->controller on "
        "CoreWeave, everything else->direct); an EXPLICIT 'direct'/'controller' always wins. "
        "'direct' = legacy path, no controller-ingress "
        "wiring (byte-identical). 'controller' = register the endpoint with the iris "
        "controller and serve it through the /proxy/t/<token>/... capability URL; on a "
        "CoreWeave cluster this REQUIRES --target-cluster (federated submission) so the "
        "capability URL is reachable — see validate_controller_ingress_reachability().",
    )
    parser.add_argument(
        "--ingress-host",
        "--ingress_host",
        dest="ingress_host",
        default=None,
        help="Public controller-ingress host the sandbox-facing capability URL is built "
        "against (only used with --ingress-mode controller). For the federated CoreWeave "
        "path this MUST be the marin meta-scheduler host 'iris.oa.dev' (the parent that "
        "owns the mirrored endpoint + signs the token), NOT the peer's own host.",
    )
    parser.add_argument(
        "--target-cluster",
        "--target_cluster",
        dest="target_cluster",
        default=None,
        help="Federate the whole job to this peer cluster via the marin meta-scheduler "
        "instead of submitting directly to --cluster's controller. Appends a "
        "`cluster EQ <peer>` constraint and submits through the marin controller "
        "(iris.oa.dev, IAP-gated — needs `iris login`), so marin delegates the job to "
        "the peer child and can federation-proxy /proxy to the peer's endpoint. Required "
        "to make --ingress-mode controller reachable from Daytona on CoreWeave. Leave "
        "unset for the default direct submission.",
    )
    parser.add_argument(
        "--parent-cluster-config",
        "--parent_cluster_config",
        dest="parent_cluster_config",
        default=None,
        help="Path to the PARENT (marin) iris cluster YAML used for federated submission "
        "when --target-cluster is set. Defaults to the marin.yaml sibling of "
        "--cluster-config. The direct path never reads this.",
    )
    parser.add_argument(
        "--record-literal",
        "--record_literal",
        dest="record_literal",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Co-locate Harbor's RecordProxy in front of vLLM to capture literal.jsonl. "
        "Default: enabled for every harness except terminus-2. Pass --record-literal to force "
        "it on or --no-record-literal to opt out. It is forwarded when controller ingress is used.",
    )
    parser.add_argument(
        "--parent-controller-config-in-pod",
        "--parent_controller_config_in_pod",
        dest="parent_controller_config_in_pod",
        default=None,
        help="In-pod path to the parent (marin) cluster YAML the in-pod worker mints "
        "against, if it differs from the launch-host --parent-cluster-config path. "
        "Defaults to the launch-host resolved marin.yaml path (must be materialized "
        "in-pod — see the OTAGENT_PARENT_CONTROLLER_CONFIG forwarding NOTE).",
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
        "--iris-ref",
        "--iris_ref",
        dest="iris_ref",
        default=DEFAULT_IRIS_VERSION,
        help="marin-iris pip spec installed into the RL venv at pod bootstrap for the "
        "controller-ingress registration/mint path (GAP D: iris is NOT baked into the "
        "gpu-rl image). Only installed under --ingress-mode controller (direct mode is "
        f"byte-identical, no install). Default: {DEFAULT_IRIS_VERSION}.",
    )
    parser.add_argument(
        "--harbor-ref",
        "--harbor_ref",
        dest="harbor_ref",
        default=None,
        help="If set, `uv pip install --no-deps --force-reinstall` harbor at this git ref "
        "into the RL venv at pod bootstrap (pure-python, ~1 min, no image rebuild) BEFORE "
        "running — the harbor analog of the /app source sync for the NON-editable baked "
        "harbor. Use to "
        "apply the opencode literal BRIDGE (harbor branch feuer/opencode-literal-rollout-"
        "details: the x-ot-trial-id header + rollout_build correlator + hosted_vllm gate) "
        "without waiting for it to land in the baked image. Under set -e + a hard "
        "`import harbor.literal.rollout_build` check so a stale/missing bridge fails loud. "
        "Default: unset = use the baked harbor.",
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


def _load_rl_config_yaml(rl_config_path: str) -> dict:
    """Resolve an RL config path (repo-relative, else as given) and parse its YAML to a dict.

    Raises on an unreadable/invalid file; callers that want a soft default wrap this."""
    full = PROJECT_ROOT / rl_config_path
    path = full if full.exists() else Path(rl_config_path)
    with open(path) as f:
        return yaml.safe_load(f) or {}


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
        raw = _load_rl_config_yaml(rl_config_path)
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


def load_config_policy_chat_template(rl_config_path: str) -> Optional[str]:
    """The config's top-level ``policy_chat_template`` (repo-relative jinja path), or None.

    None when the key is unset (existing configs have no such key). Set only by single-turn
    RLVR configs that must force a chat template onto the policy tokenizer cache because the
    SFT repo ships none. Read errors propagate: this drives fail-loud template machinery, so
    an unreadable config must abort rather than silently skip the override."""
    value = _load_rl_config_yaml(rl_config_path).get("policy_chat_template")
    return str(value) if value else None


def load_config_trainer_ckpt_path(rl_config_path: str) -> Optional[str]:
    """Return an EXPLICIT ``trainer.ckpt_path`` from the RL config YAML, else None.

    The iris configs set ``ckpt_path: null`` (auto-derived downstream in
    rl_config_translation). A config that sets it explicitly (non-null, non-empty) should
    WIN over the launcher's durable-s3 default, so build_task_command consults this
    before injecting its override. Returns None when the file is unreadable, has no
    ``trainer.ckpt_path``, or the value is null/empty (byte-identical to today for
    every existing iris config, which all leave it null)."""
    try:
        raw = _load_rl_config_yaml(rl_config_path)
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

    # Cross-cluster ingress (opencode-RL literal capture): forward the ingress flags to
    # the in-pod runner (cloud.iris.run_rl), which stands up the RecordProxy + registers
    # + mints the capability URL. Only emitted under --ingress-mode controller; the
    # default (direct) path adds nothing (byte-identical run_rl invocation).
    if getattr(args, "ingress_mode", "direct") == "controller":
        train_cmd.extend(["--ingress_mode", "controller"])
        if getattr(args, "ingress_host", None):
            train_cmd.extend(["--ingress_host", args.ingress_host])
        if getattr(args, "target_cluster", None):
            train_cmd.extend(["--target_cluster", args.target_cluster])
            # Parent (marin) config the in-pod worker mints against. Prefer an explicit
            # in-pod path; else pass the resolved marin.yaml path (must be materialized
            # in-pod — see the OTAGENT_PARENT_CONTROLLER_CONFIG env forwarding + NOTE).
            parent_cfg_in_pod = getattr(args, "parent_controller_config_in_pod", None) or (
                args.parent_cluster_config or _resolve_parent_cluster_config(args.cluster_config)
            )
            if parent_cfg_in_pod:
                train_cmd.extend(["--parent_controller_config", parent_cfg_in_pod])
    if getattr(args, "record_literal", False):
        train_cmd.append("--record_literal")

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
    _policy_chat_template = load_config_policy_chat_template(args.rl_config)
    _offline = str(_cfg_env.get("HF_HUB_OFFLINE", "")).strip().lower() in ("1", "true", "yes", "on")
    # A policy_chat_template override rewrites the node-local tokenizer cache, so it REQUIRES
    # a prestage even when the config is not offline (nothing to rewrite otherwise).
    if _offline or _policy_chat_template:
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
    # Force the delphi chat template onto every node's tokenizer cache (single-turn RLVR).
    # Repo-relative path (resolved in-pod against /app by the controller). No-op for configs
    # without policy_chat_template.
    if _policy_chat_template:
        controller_cmd.extend(["--policy-chat-template", _policy_chat_template])
    controller_cmd.append("--")
    controller_cmd.extend(train_cmd)

    # Wrap in a bash bootstrap: cd to the synced workspace and set PYTHONPATH so
    # live /app + skyrl-train win over the image's baked copies. Use the absolute
    # RL venv python (set above) — independent of iris's activated venv.
    #
    # ALL MarinSkyRL code resolves from the SYNCED /app (the launcher syncs
    # PROJECT_ROOT -> /app), retiring the /app-vs-/opt/skyrl shadow split:
    #   - /app            : cloud.* (cloud/iris/*)  -> synced source
    #   - /app/skyrl-train: skyrl_train.* + examples.terminal_bench.* -> synced source
    #   - /opt/skyrl/skyrl-train: baked fallback (kept LAST so a partial /app can't
    #     ModuleNotFoundError, but /app wins for anything it carries)
    # The venv/deps still come from the /opt/skyrl editable install; only the SOURCE
    # that `import skyrl_train`/`examples.*` resolves is moved to /app. This makes the
    # /app sync the single deploy vector — everything now rides the /app sync off `main`;
    # /opt/skyrl stays at its baked HEAD purely as a last-resort import fallback.
    pythonpath = f"{APP_DIR}:{APP_DIR}/skyrl-train:{SKYRL_HOME}/skyrl-train"
    # Optional: reinstall harbor at a newer/pinned commit BEFORE running. Harbor is baked
    # NON-editable into the RL venv, so (unlike skyrl-train, whose source rides the /app
    # sync) there is no editable clone to
    # `git checkout`; the analog is a `--force-reinstall --no-deps` from git (pure-python,
    # ~1 min, no image rebuild). --no-deps so only the harbor source swaps (its resolved
    # deps stay the baked known-good set). Under `set -e` + a hard
    # `import harbor.literal.rollout_build` check so a failed reinstall or a stale baked
    # harbor (missing the opencode literal-bridge module) KILLS the job loud rather than
    # silently no-ops (the false-negative the generator's lazy-import would otherwise hide).
    harbor_refresh = ""
    if getattr(args, "harbor_ref", None):
        hspec = "harbor[daytona] @ git+https://github.com/marin-community/harbor.git@" + args.harbor_ref
        harbor_refresh = (
            f"uv pip install --python {shlex.quote(RL_PYTHON)} --no-deps "
            f"--force-reinstall {shlex.quote(hspec)}; "
            f'{RL_PYTHON} -c "import harbor, importlib.metadata as m; '
            f"print('[rl-iris] harbor now', m.version('harbor'))\"; "
            f'{RL_PYTHON} -c "import harbor.literal.rollout_build; '
            f"print('[rl-iris] harbor.literal.rollout_build import OK')\"; "
        )
    # GAP D fix: install marin-iris into the RL venv at bootstrap for the controller-
    # ingress registration/mint path. cloud.iris.ingress_utils hard-imports
    # iris.cluster.client.* / iris.rpc.*, but the gpu-rl image bakes ONLY MarinSkyRL +
    # harbor (never iris, a marin-monorepo pkg) -> `ModuleNotFoundError: No module named
    # 'iris'` in driver init. This is a lightweight live install (no ~40-min
    # kaniko rebuild): marin-iris is a pure-python wheel, installed live. Only needed in
    # controller mode (direct mode never imports iris), so gate on ingress_mode ==
    # controller -> the default direct path is byte-identical (no install, no env change).
    #   - NO [controller] extra: the CLIENT registration path (EndpointClient + rpc stubs)
    #     needs neither kubernetes<36 nor Secret-Manager (it loads grpc + connectrpc +
    #     rigging + finelog only), so skipping it avoids the biggest dep-conflict source.
    #   - No --constraint file: the lock IS the constraint (Dockerfile.gpu-rl builds the
    #     RL venv via `uv sync --frozen` off skyrl-train/uv.lock; there is no baked
    #     rl_env_constraints.txt anymore — pointing at it File-not-found'd on newer images).
    #     `uv pip install` does not upgrade already-satisfying installed deps, and iris's
    #     deps are all present with satisfied >= bounds, so uv only ADDS pure-python leaves;
    #     torch/vllm/flash_attn aren't in iris's tree. The one real downgrade vector (boto,
    #     an upper-bound pin) is separately snapshot+restored below.
    #   - GAP E#2 boto guard: the marin-iris solve DOWNGRADES the (deliberately-unpinned)
    #     botocore cluster (1.43.46 -> 1.43.0), breaking `from botocore.docs.utils import
    #     DocumentModifiedShape` (accelerate imports it transitively -> a MASKED
    #     "accelerate circular import" that killed every prior controller-ingress smoke).
    #     Snapshot the baked boto pins with `uv pip freeze` (the venv is uv-managed, NO
    #     pip module) BEFORE the install and force-restore them (--no-deps) AFTER.
    #   - Under `set -e` + a hard import of the exact registration path AND torch/vllm/
    #     flash_attn + DocumentModifiedShape asserts: a clobbered pin KILLS the job loud
    #     at bootstrap rather than dying deep in driver init.
    iris_refresh = ""
    if getattr(args, "ingress_mode", "direct") == "controller":
        ispec = getattr(args, "iris_ref", None) or DEFAULT_IRIS_VERSION
        iris_refresh = (
            f"_BOTO_BAKED=$(uv pip freeze --python {shlex.quote(RL_PYTHON)} 2>/dev/null | "
            f"grep -iE '^(botocore|boto3|s3transfer|awscli)==' | tr '\\n' ' ' || true); "
            f'echo "[rl-iris] boto baked pins: $_BOTO_BAKED"; '
            f"uv pip install --python {shlex.quote(RL_PYTHON)} {shlex.quote(ispec)}; "
            f'if [ -n "$_BOTO_BAKED" ]; then uv pip install --python '
            f"{shlex.quote(RL_PYTHON)} --no-deps $_BOTO_BAKED; fi; "
            f'{RL_PYTHON} -c "import importlib.metadata as m; '
            f"import iris.cluster.client.endpoint_client, iris.cluster.client.job_info, "
            f"iris.rpc.controller_connect, iris.cluster.types; "
            f"print('[rl-iris] marin-iris now', m.version('marin-iris'), "
            f"'(controller-ingress import OK)')\"; "
            f'{RL_PYTHON} -c "import botocore; from botocore.docs.utils import '
            f"DocumentModifiedShape; print('[rl-iris] boto cluster intact: botocore', "
            f'botocore.__version__)"; '
            f'{RL_PYTHON} -c "import torch, vllm, flash_attn, flash_attn_2_cuda; '
            f"print('[rl-iris] post-iris pins intact: torch', torch.__version__, "
            f"'vllm', vllm.__version__)\"; "
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
        f"{harbor_refresh}"
        f"{iris_refresh}"
        f"export SKYRL_HOME={shlex.quote(SKYRL_HOME)}; "
        f"export PYTHONPATH={shlex.quote(pythonpath)}:${{PYTHONPATH:-}}; "
        f"export VLLM_USE_V1=1; "
        f"{gdn_branch}"
    )
    return ["bash", "-c", bash]


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    # Resolve the cluster YAML from --cluster when not explicitly given, so
    # `--cluster cw-rno2a` targets the 512xH100 RNO2A cluster (the delphi pilot's
    # 512-node target) without a manual --cluster-config.
    if not args.cluster_config:
        args.cluster_config = _resolve_cluster_config_default(args.cluster)
    normalize(args)
    resolve_launch_defaults(args)

    # Derive the controller-ingress config from the target cluster so an agentic
    # CoreWeave launch works from --target-cluster alone (no manual --ingress-mode/
    # --ingress-host). Runs BEFORE the reachability guard, which then only sees the
    # single correct cluster-determined config.
    autoconfigure_ingress(args)

    # Fail loud (before any submit / GPU allocation) when controller-ingress would
    # produce a capability URL the Daytona sandbox cannot reach — the Exp2 blocker
    # (opencode never reaches vLLM on CoreWeave via a directly-submitted job). The
    # default direct path returns immediately (byte-identical).
    validate_controller_ingress_reachability(args)
    parent_credentials_json = prepare_federated_parent_credentials(args)

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
    # --target-cluster (federated submission) appends a `cluster EQ <peer>` constraint
    # so the marin meta-scheduler DELEGATES the whole job to the peer child (see the
    # submission block below). None on the default direct path (byte-identical).
    constraints = build_job_constraints(
        resources_proto=resources_proto,
        tpu_variants=[],
        replicas=replicas,
        regions=None,
        zone=None,
        preemptible=args.preemptible,
        target_cluster=args.target_cluster,
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
    # ── Per-cluster infra-env DEFAULTS (fill-gap belt for cluster-specific footguns) ──────────
    # Some clusters need a specific network/NCCL interface that a cluster-AGNOSTIC RL config
    # won't (and shouldn't) carry. Fill it in here, keyed on --target-cluster, ONLY if neither
    # the flag env nor the config's extra_env already set it (lowest precedence — an explicit
    # value always wins). cw-rno2a: its host_network:true nodes expose IB/IPoIB (ibs*/ibp*) +
    # virtual ifaces, so NCCL AND Ray's raylet/GCS mis-detect the bootstrap interface and the
    # multi-node gang SILENTLY never forms (keep-6 2026-07-19: 6 arms idle-heartbeated 3.4h at
    # zero progress before this was diagnosed). Pin the bootstrap socket to the host ethernet PF
    # via the exclude pattern (value = cw-rno2a.yaml:128). No-op on cw-us-east-02a (auto-detect
    # already lands on the PF). Add a cluster row here rather than editing every RL config.
    _target_cluster = str(getattr(args, "target_cluster", "") or "")
    _CLUSTER_ENV_DEFAULTS: dict[str, dict[str, str]] = {
        "cw-rno2a": {"NCCL_SOCKET_IFNAME": "^ibs,ibp,lo,docker,veth,cilium,lxc"},
    }
    for _k, _v in _CLUSTER_ENV_DEFAULTS.get(_target_cluster, {}).items():
        if _k not in env_vars:
            env_vars[_k] = _v
            print(f"[rl-iris] Cluster infra-env default for {_target_cluster}: {_k}={_v}", flush=True)
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

    # Federated controller-ingress pod plumbing (opencode-RL literal capture): the in-pod
    # worker mints the capability token at the PARENT (marin/iris.oa.dev) for the mirrored
    # endpoint, which needs (a) the parent cluster config path and (b) IAP credentials to
    # authenticate to iris.oa.dev. We forward the config path + any launch-host IAP cred
    # env so the in-pod _ParentControllerClient can re-mint the IAP OIDC token.
    #
    # The parent config file is not baked into the gpu-rl image, so forward its contents
    # for in-pod materialization. Federated controller ingress always forwards the cached
    # Marin login record after prepare_federated_parent_credentials() has minted a token
    # from it locally. Direct submission (no --target-cluster) forwards none of this.
    if getattr(args, "target_cluster", None) and getattr(args, "ingress_mode", "direct") == "controller":
        from cloud.iris.ingress_utils import (
            PARENT_CONTROLLER_CONFIG_ENV,
            PARENT_CONTROLLER_CONFIG_YAML_ENV,
            PARENT_CREDENTIALS_JSON_ENV,
        )

        parent_cfg = (
            getattr(args, "parent_controller_config_in_pod", None)
            or args.parent_cluster_config
            or _resolve_parent_cluster_config(args.cluster_config)
        )
        if parent_cfg:
            env_vars[PARENT_CONTROLLER_CONFIG_ENV] = parent_cfg
            # marin.yaml is not baked into the gpu-rl image and is not part of the
            # synced workspace, so the path above won't resolve in-pod. Forward the
            # file CONTENT (write-from-env, mirroring the cached login record) so the
            # in-pod worker (materialize_parent_controller_config) writes it to a real
            # path and repoints the env. marin.yaml carries no secrets (signing_key is
            # a gcp-secret:// ref resolved server-side). When parent_cfg is an explicit
            # in-pod path (baked/synced), os.path.isfile is False on the launch host →
            # no content forwarded (operator owns materialization).
            if os.path.isfile(parent_cfg):
                with open(parent_cfg) as _pf:
                    env_vars[PARENT_CONTROLLER_CONFIG_YAML_ENV] = _pf.read()
        for k in (
            "IRIS_IAP_REFRESH_TOKEN",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "IRIS_EDGE_REFRESH_TOKEN",
        ):
            v = os.environ.get(k)
            if v:
                env_vars[k] = v
        # A CoreWeave pod has no cached `iris login` and no Marin-allowlisted ambient
        # service account. Forwarding this validated record is therefore mandatory for
        # the in-pod parent mint; it remains a secret in the submitted job environment.
        if parent_credentials_json is None:
            raise AssertionError("federated controller ingress must have validated parent credentials")
        env_vars[PARENT_CREDENTIALS_JSON_ENV] = parent_credentials_json

    # Load the cluster config (pydantic IrisClusterConfig) and build the provider
    # bundle, then discover + tunnel to the controller. This mirrors the marin
    # CLI's own path (iris/cli/connect.py::require_controller_url): for a local
    # controller start an in-process LocalCluster; otherwise use the config's
    # controller_address() (defaults.worker.controller_address) if set, else fall
    # back to the backend's discover_controller(). cw-us-east-02a's controller
    # kind is "coreweave" (non-local, no IAP auth) → the discover path.
    #
    # FEDERATED submission (--target-cluster set): submit through the PARENT (marin)
    # meta-scheduler instead of the peer's own controller. We load marin.yaml (whose
    # dashboard_url is the IAP-gated iris.oa.dev) and tunnel THERE; the `cluster EQ
    # <peer>` constraint appended above makes marin delegate the whole job to the peer
    # child. This is what lets marin later federation-proxy /proxy to the peer's
    # (mirrored) endpoint — the only Daytona-reachable CoreWeave ingress topology.
    # Reaching iris.oa.dev requires IAP creds (`iris login` with an @openathena.ai
    # account, or an allowlisted service account); tunnel()/IrisClient handle the auth.
    submit_cluster_config = args.cluster_config
    if args.target_cluster:
        parent_cfg = args.parent_cluster_config or _resolve_parent_cluster_config(args.cluster_config)
        if not parent_cfg:
            raise SystemExit(
                "[rl-iris] --target-cluster set but no parent (marin) cluster config "
                "could be resolved. Pass --parent-cluster-config <path to marin.yaml>."
            )
        submit_cluster_config = parent_cfg
        print(
            f"[rl-iris] Federated submission: delegating to peer '{args.target_cluster}' "
            f"via the marin meta-scheduler ({parent_cfg}).",
            flush=True,
        )
    from contextlib import contextmanager as _contextmanager

    @_contextmanager
    def _direct_client():
        # Direct submission to --cluster's own controller. On CoreWeave the loopback SSH
        # tunnel presents as the trusted local_admin identity (no IAP login needed) —
        # byte-identical to before.
        iris_config = load_config(submit_cluster_config)
        bundle = provider_bundle(iris_config)
        if iris_config.controller.controller_kind() == "local":
            controller_address = LocalCluster(iris_config).start()
        else:
            controller_address = iris_config.controller_address() or bundle.controller.discover_controller(
                iris_config.controller
            )
        with bundle.controller.tunnel(address=controller_address) as controller_url:
            yield IrisClient.remote(controller_url, workspace=PROJECT_ROOT)

    if args.target_cluster:
        # Federated submission MUST carry the IAP *user* identity: the controller rejects
        # a loopback/local_admin tunnel identity for a federated job ("a local_admin
        # (CIDR/loopback) identity cannot submit a federated job"), because delegation
        # forwards the submitter's identity to the peer for its owner check. Connect to
        # the marin parent exactly as the `iris job run` CLI does — open_iris_client
        # threads the IAP ClientCredentials (iris JWT + IAP OIDC token) from the cached
        # `iris login`, so the submission carries the user identity rather than loopback.
        # Requires a completed `iris --cluster=marin login` (@openathena.ai).
        from iris.cli.connect import open_iris_client

        client_cm = open_iris_client(config_file=Path(submit_cluster_config), workspace=PROJECT_ROOT)
    else:
        client_cm = _direct_client()

    with client_cm as client:
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
