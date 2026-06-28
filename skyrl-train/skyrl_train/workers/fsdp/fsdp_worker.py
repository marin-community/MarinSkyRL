import asyncio

from skyrl_train.utils.trainer_utils import get_rope_scaling_config, get_rope_theta_config
import ray
import torch
import torch.distributed
from transformers import AutoConfig
from torch.distributed.fsdp.api import ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
import io

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from skyrl_train.model_wrapper import HFModelWrapper, get_llm_for_sequence_regression
from skyrl_train.distributed.fsdp_strategy import FSDPStrategy
from skyrl_train.utils import get_physical_gpu_id, str_to_torch_dtype
from skyrl_train.training_batch import TrainingInputBatch, TrainingOutputBatch
from skyrl_train.distributed.fsdp_utils import fsdp_version, get_init_weight_context_manager
from skyrl_train.workers.worker import (
    PolicyWorkerBase,
    CriticWorkerBase,
    RefWorkerBase,
)
from skyrl_train.weight_sync import WeightExtractor, WeightChunk
from skyrl_train.weight_sync.weight_extractor_utils import yield_module_grouped_chunks


class FSDPWeightExtractor(WeightExtractor):
    """Extracts weights from FSDP-sharded models.

    Args:
        model: FSDP model to extract weights from
        group_by_module: If True, group parameters by module (e.g., for FlashRL QKV fusion)
        batch_size_threshold_gb: If > 0, batch complete modules together until threshold is reached
        moe_grouped_gemm: If True, the model was grouped-swapped (Stage 3b) so its MoE
            blocks are ``GroupedMoEShim`` instances holding grouped ``experts.w1/w2/w3``
            tensors. The extracted state dict is then name/shape-remapped back to the
            per-expert HF layout the inference engine expects (Stage 4b). Default False
            keeps the path byte-identical to the non-grouped (a3-production) extractor.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        group_by_module: bool = False,
        batch_size_threshold_gb: float = 0.0,
        moe_grouped_gemm: bool = False,
    ):
        self.model = model
        self.group_by_module = group_by_module
        self.batch_size_threshold_gb = batch_size_threshold_gb
        self.moe_grouped_gemm = moe_grouped_gemm
        # Per-arch inference-engine (vLLM) weight-NAME translation. Most grouped-MoE
        # arches (qwen3_moe/qwen3_next/olmoe) emit broadcast names that already match
        # vLLM's stock params_dict, so this stays the identity for them. Mixtral is the
        # exception (transformers-5.x ``mlp.*`` vs vLLM's stock ``block_sparse_moe.*``);
        # ``translate_moe_name_to_vllm`` renames ONLY Mixtral keys (see moe_weight_remap).
        _cfg = getattr(model, "config", None)
        self._model_type = getattr(_cfg, "model_type", "") or "" if _cfg is not None else ""
        # Qwen3.5/3.6 VLM-shell weight-sync (tmax Stage 2): the RL policy is the
        # unwrapped TEXT tower (``Qwen3_5MoeForCausalLM``, names ``model.*``) but the
        # vLLM rollout engine instantiates the multimodal SHELL
        # (``Qwen3_5MoeForConditionalGeneration``), whose ``load_weights`` expects the
        # text decoder under the HF namespace ``model.language_model.*``. When the
        # policy config is the hybrid text tower, the broadcast names must be
        # remapped ``model.X`` -> ``model.language_model.X`` (see
        # ``map_text_name_to_vlm_engine``). Identity for every other arch.
        from skyrl_train.models.qwen3_5_vlm import is_qwen3_5_text_tower

        self._is_qwen3_5_text_tower = is_qwen3_5_text_tower(_cfg)

    def _translate_name(self, name: str) -> str:
        """Apply the per-arch inference-engine name translation (identity for all
        arches except Mixtral, and the Qwen3.5/3.6 VLM-shell namespace). Scoped via
        ``self._model_type`` / ``self._is_qwen3_5_text_tower``."""
        from skyrl_train.models.layers.moe_weight_remap import translate_moe_name_to_vllm

        name = translate_moe_name_to_vllm(name, self._model_type)
        if self._is_qwen3_5_text_tower:
            from skyrl_train.models.qwen3_5_vlm import map_text_name_to_vlm_engine

            name = map_text_name_to_vlm_engine(name)
        return name

    def extract_weights(self, dtype: torch.dtype):
        """Extract weights from FSDP model.

        Args:
            dtype: Target dtype for inference

        Yields:
            WeightChunk objects (one per parameter, or grouped by module)
        """
        # Configure state_dict type for FSDP v1
        if fsdp_version(self.model) == 1:
            FSDP.set_state_dict_type(
                self.model,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        # Get state dict (handles FSDP sharding)
        params = self.model.state_dict()

        # Stage 7 (80B) — STREAMED grouped gather. For grouped-swapped models the old
        # path eagerly `full_tensor()`-gathered EVERY layer's grouped `experts.w1/w2/w3`
        # (the 512-expert stacks) into a single remapped dict before the broadcast loop,
        # materializing the whole unsharded MoE on ONE GPU → OOM at 80B (job 602650,
        # 93.78/95 GiB). When grouped + the (disaggregated/NCCL-broadcast) per-tensor
        # transport, stream instead: gather → remap → yield → FREE one MoE layer (and one
        # non-MoE param) at a time, so peak GPU memory is a single layer's expert stack,
        # not all 48. Byte-identical to the eager remap (same converter, same tensors);
        # only the materialization order/lifetime changes.
        # Gated: non-grouped models (a3: moe_grouped_gemm=False) skip this entirely and
        # take the unchanged simple/grouped-by-module paths below — code-path identical.
        if self.moe_grouped_gemm and not self.group_by_module:
            yield from self._extract_weights_streamed(params, dtype)
            return

        # Stage 4b: if the trainer was grouped-swapped (Stage 3b) AND on the CUDA-IPC /
        # FlashRL module-grouping path (colocated NCCL IPC — not the 80B disaggregated
        # broadcast), fall back to the eager whole-model remap. This combination is not
        # on the 80B path; left unchanged.
        if self.moe_grouped_gemm:
            params = self._remap_grouped_state_dict(params)

        if not self.group_by_module:
            # Simple path: yield one chunk per parameter
            for name, param in params.items():
                tensor = self._gather_tensor(param).to(dtype).detach().contiguous()
                name = self._translate_name(name)
                yield WeightChunk(
                    names=[name],
                    dtypes=[str(dtype)],
                    shapes=[list(tensor.shape)],
                    tensors=[tensor],
                )
        else:
            for chunk in yield_module_grouped_chunks(
                params=params,
                dtype=dtype,
                gather_tensor_fn=self._gather_tensor,
                get_shape_fn=lambda name, param, tensor: list(tensor.shape),
                batch_size_threshold_gb=self.batch_size_threshold_gb,
            ):
                yield chunk

    def _gather_tensor(self, param: torch.Tensor) -> torch.Tensor:
        """Gather sharded tensor into full tensor.

        For EP+FSDP-composed grouped-expert params (placement
        ``(_StridedShard(fsdp), Shard(ep))``) ``full_tensor()`` reassembles the
        expert ROWS in the WRONG global order on torch 2.11 (the
        ``_StridedShard.is_shard()==False`` / non-ascending-all_gather quirk that
        silently corrupted the r2–r7 MoE weight sync). ``gather_dtensor_strided_safe``
        gathers via each placement's own ``_split_tensor`` instead, so the global
        expert order is correct and version-independent. It is byte-identical to
        ``full_tensor()`` for every non-strided (a3 / non-EP / plain-Shard) param.
        """
        from skyrl_train.distributed.fsdp_utils import gather_dtensor_strided_safe

        device = torch.cuda.current_device()
        if not isinstance(param, DTensor):
            return param
        out = gather_dtensor_strided_safe(param.to(device, non_blocking=True))

        # MIXED-PG WEIGHT-SYNC SERIALIZATION (default OFF -> byte-identical).
        #
        # The weight-sync gather sequence interleaves collectives on DIFFERENT
        # process groups WITHOUT a barrier between them: strided grouped-expert
        # params (experts.w1/w2/w3) gather via ``dist.all_gather`` on the GLOBAL
        # default PG (gather_dtensor_strided_safe), while plain-Shard non-expert
        # params (attn / norm / router.gate / shared_expert) gather via
        # ``full_tensor()`` on a CP/FSDP SUB-MESH sub-communicator. In the streamed
        # MoE path these are issued back-to-back inside one layer's gather batch
        # (_extract_weights_streamed), and on rank 0 the long nranks=N+1 weight
        # Broadcast to the inference engines is additionally interleaved between
        # yields. The only barrier in the broadcast loop is per-YIELDED-CHUNK on the
        # default PG -- it does NOT sit between the intra-layer sub-mesh gathers and
        # does NOT order NCCL ops across communicators. That mixed-PG overlap is the
        # CoreWeave MoE weight-sync NCCL deadlock (2026-06-28: both arms wedged on a
        # mesh_fsdp sub-comm full_tensor() AllGather while default_pg was drained).
        #
        # When SKYRL_WEIGHT_SYNC_SERIALIZE=1, fully drain + barrier after EVERY
        # gather so no two different-group collectives can overlap: stream-complete
        # this gather on all ranks, then meet at a default-PG barrier before the next
        # gather (or the rank-0 broadcast) is issued. Pure ordering hardening -- it
        # changes NO tensor values, so it is correctness/w13-neutral and cannot
        # re-introduce the 80B materialize-all OOM (still one layer at a time). OFF by
        # default => the existing a3 / dense / 80B paths are byte-identical.
        import os

        if os.environ.get("SKYRL_WEIGHT_SYNC_SERIALIZE", "0") == "1":
            torch.cuda.synchronize()
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        return out

    def _extract_weights_streamed(self, params, dtype: torch.dtype):
        """Streamed grouped-MoE weight extraction (Stage 7 / 80B OOM fix).

        Yields one ``WeightChunk`` per HF parameter, gathering + remapping LAZILY so
        peak GPU memory is bounded by a single MoE layer's grouped expert stack (3 ×
        ``[num_experts, moe_dim, dim]``) rather than the whole unsharded 80B model. The
        emitted tensors are byte-identical to the eager ``_remap_grouped_state_dict``
        path — same ``full_tensor()`` gather, same ``convert_tt_layer_to_hf`` per-expert
        split, same dtype/contiguity — only their lifetime is per-layer.

        IMPORTANT (collective correctness): ``full_tensor()`` is a collective over the
        FSDP/EP mesh, so EVERY rank must drive this generator and reach each gather in
        the SAME order. Iteration order is the deterministic ``state_dict()`` ordering on
        all ranks, so the gather sequence is identical across ranks (matches the eager
        path, which also gathered in dict order).
        """
        from skyrl_train.models.layers.moe_weight_remap import convert_tt_layer_to_hf

        # Post-prefix-strip suffixes of the grouped-block tensors the converter consumes.
        grouped_suffixes = (
            ".mlp.experts.w1",
            ".mlp.experts.w2",
            ".mlp.experts.w3",
            ".mlp.router.gate.weight",
            ".mlp.shared_expert.w1.weight",
            ".mlp.shared_expert.w2.weight",
            ".mlp.shared_expert.w3.weight",
        )

        def _layer_of(name: str):
            # ``model.layers.{i}.mlp.experts.w1`` -> i ; None for non-layer keys.
            parts = name.split(".")
            if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
                try:
                    return int(parts[2])
                except ValueError:
                    return None
            return None

        # First pass: strip prefixes (cheap, no gather) and partition into per-layer
        # grouped-MoE tensors vs. everything else, preserving state_dict() order.
        # ``layer_groups[i]`` = list of (stripped_name, dtensor_param) for that layer's
        # grouped MoE keys; ``passthrough`` = ordered (stripped_name, param) for the rest.
        from collections import OrderedDict

        layer_groups: "OrderedDict[int, list]" = OrderedDict()
        # Ordered plan of work items: ("moe", layer_idx) flushes that layer once, "param"
        # yields a single non-MoE tensor. Emitted in first-encounter order so the gather
        # sequence is deterministic and identical on every rank.
        plan = []
        seen_moe_layer = set()
        passthrough = OrderedDict()

        for name, param in params.items():
            new_name = self._strip_grouped_prefix(name)
            if new_name.endswith(grouped_suffixes):
                li = _layer_of(new_name)
                layer_groups.setdefault(li, []).append((new_name, param))
                if li not in seen_moe_layer:
                    seen_moe_layer.add(li)
                    plan.append(("moe", li))
            else:
                passthrough[new_name] = param
                plan.append(("param", new_name))

        for kind, key in plan:
            if kind == "param":
                param = passthrough[key]
                tensor = self._gather_tensor(param).to(dtype).detach().contiguous()
                key = self._translate_name(key)
                yield WeightChunk(
                    names=[key],
                    dtypes=[str(dtype)],
                    shapes=[list(tensor.shape)],
                    tensors=[tensor],
                )
                del tensor
            else:
                # Gather ONLY this layer's grouped MoE tensors, remap per-expert, yield
                # each, then free the layer's grouped stack before moving on.
                layer_sd = {}
                for sname, sparam in layer_groups[key]:
                    layer_sd[sname] = self._gather_tensor(sparam).detach().contiguous()
                # In-place grouped -> per-expert HF split for THIS layer only. The
                # per-expert entries are views into w1/w2/w3 (no extra alloc); we
                # .contiguous() each on yield so the parent stack can free after the loop.
                convert_tt_layer_to_hf(layer_sd, key)
                for ename, etensor in layer_sd.items():
                    out = etensor.to(dtype).detach().contiguous()
                    ename = self._translate_name(ename)
                    yield WeightChunk(
                        names=[ename],
                        dtypes=[str(dtype)],
                        shapes=[list(out.shape)],
                        tensors=[out],
                    )
                    del out
                # Drop all references to this layer's gathered tensors + per-expert views
                # so the (large) grouped expert stack is freed before the next layer.
                del layer_sd
                torch.cuda.empty_cache()

    # MoE grouped-block (GroupedMoEShim.moe) segment that sits between the HF
    # `...mlp.` prefix and the grouped `experts.w1/...`/`router.gate` keys the
    # `convert_tt_to_hf_moe` converter matches on. FSDP2 `fully_shard` does not add a
    # `_fsdp_wrapped_module` segment to state_dict keys, but FSDP1 (and nested wraps)
    # can — strip it defensively so the remap is layout-agnostic.
    _SHIM_SEG = ".mlp.moe."
    _FSDP_SEG = "._fsdp_wrapped_module."

    @staticmethod
    def _strip_grouped_prefix(name: str) -> str:
        """Normalize a live grouped-swapped key to the converter's expected form.

        ``...layers.{i}.mlp.moe.experts.w1`` -> ``...layers.{i}.mlp.experts.w1``
        ``...layers.{i}.mlp.moe.router.gate.weight`` -> ``...mlp.router.gate.weight``
        Also drops any FSDP ``_fsdp_wrapped_module`` segments.
        """
        name = name.replace(FSDPWeightExtractor._FSDP_SEG, ".")
        name = name.replace(FSDPWeightExtractor._SHIM_SEG, ".mlp.")
        return name

    def _remap_grouped_state_dict(self, params):
        """Strip the GroupedMoEShim/FSDP prefix + run ``convert_tt_to_hf_moe`` in place.

        Only the grouped MoE tensors (``experts.w1/w2/w3``, ``router.gate``, the shared
        expert ``w1/w2/w3``) need to be materialized to full tensors before the converter
        slices them per-expert (``w1[j]``) — a DTensor ``Shard(0)`` on the expert dim would
        otherwise give a partial slice. Non-MoE params are left as-is (gathered lazily in the
        existing broadcast loop). After the converter runs, expert keys become the per-expert
        HF names the inference engine already loads.
        """
        from skyrl_train.models.layers.moe_weight_remap import convert_tt_to_hf_moe

        # Grouped-block tensors the converter consumes (post-prefix-strip suffixes).
        grouped_suffixes = (
            ".mlp.experts.w1",
            ".mlp.experts.w2",
            ".mlp.experts.w3",
            ".mlp.router.gate.weight",
            ".mlp.shared_expert.w1.weight",
            ".mlp.shared_expert.w2.weight",
            ".mlp.shared_expert.w3.weight",
        )

        remapped = {}
        for name, param in params.items():
            new_name = self._strip_grouped_prefix(name)
            if new_name.endswith(grouped_suffixes):
                # Materialize before the converter slices per-expert.
                remapped[new_name] = self._gather_tensor(param).detach().contiguous()
            else:
                remapped[new_name] = param

        # In-place grouped -> per-expert HF remap (splits w1/w2/w3 into experts.{j}.*).
        convert_tt_to_hf_moe(remapped)
        return remapped


class FSDPPolicyWorkerBase(PolicyWorkerBase):
    def offload_to_cpu(self, pin_memory=True, non_blocking=True, offload_optimizer=True, offload_model=True):
        self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
        self.strategy.offload_to_cpu(
            self.model, self.optimizer, pin_memory, non_blocking, offload_optimizer, offload_model
        )

    def backload_to_gpu(self, non_blocking=True, backload_optimizer=True, backload_model=True):
        self.strategy.backload_to_gpu(self.model, self.optimizer, non_blocking, backload_optimizer, backload_model)

    def read_post_step_weights(self, names):
        """TEST-ONLY (Stage 6 weight-equality gate): return the post-step HF-named
        weight tensors the broadcast would send, for a representative ``names`` set.

        Runs the SAME ``extract_weights`` path used by ``broadcast_to_inference_engines``
        (grouped->HF remap + FSDP ``full_tensor()`` gather), so the returned tensors
        are byte-identical to what the engine receives. ``extract_weights`` /
        ``full_tensor()`` are collective over the full mesh, so EVERY rank must run
        the generator; only rank 0 returns the (full) tensors as CPU fp32 — other
        ranks return an empty dict to keep the payload small.
        """
        wanted = set(names)
        collected = {}
        generator_dtype = str_to_torch_dtype(self.cfg.generator.model_dtype)
        is_rank0 = torch.distributed.get_rank() == 0
        # Must drive the full generator on every rank (the per-tensor full_tensor()
        # gather is a collective); harvest only the requested names on rank 0.
        for chunk in self.weight_extractor.extract_weights(generator_dtype):
            for name, tensor in zip(chunk.names, chunk.tensors):
                if is_rank0 and name in wanted:
                    collected[name] = tensor.detach().to("cpu", dtype=torch.float32).contiguous()
        return collected

    def diag_ep8_geometry(self):
        """TEST-ONLY (EP=8 cross-node diag): return this rank's mesh geometry +
        physical-node identity so the driver can PROVE an EP group straddles >=2
        nodes. No collectives, no gather — pure introspection.

        Returns a dict with global rank, hostname, mesh shape/dim-names, this rank's
        per-mesh-dim coordinate, and the EP submesh coordinate (the index of this rank
        within its 8-rank EP group).
        """
        import socket

        mesh = self.strategy.device_mesh
        dim_names = list(mesh.mesh_dim_names)
        shape = tuple(mesh.shape)
        coord = list(mesh.get_coordinate())
        ep_dim = dim_names.index("ep") if "ep" in dim_names else None
        # The EP-group identity = the coord with the ep dim removed (all ranks sharing
        # this tuple form one 8-way EP group). The ep coord = position within the group.
        group_key = tuple(c for i, c in enumerate(coord) if i != ep_dim)
        return {
            "rank": int(torch.distributed.get_rank()),
            "host": socket.gethostname(),
            "mesh_dim_names": dim_names,
            "mesh_shape": shape,
            "coord": coord,
            "ep_dim": ep_dim,
            "ep_coord": (coord[ep_dim] if ep_dim is not None else None),
            "ep_group_key": group_key,
        }

    def diag_ep8_disk_ref_compare(self, model_path, layer_idx=0, n_rep_gather=2):
        """TEST-ONLY (EP=8 cross-node, NON-CIRCULAR weight-equality assert).

        Captures the value-corruption signature of the FSDP->vLLM MoE weight gather
        WITHOUT any inference engine, rollout, or engine-readback. For ``layer_idx``'s
        grouped expert stacks (w1/w2/w3) this rank:

          1. ON-GPU: runs the REAL ``self._gather_tensor`` (= ``gather_dtensor_strided_safe``
             over the ``(_StridedShard(fsdp), Shard(ep))`` composite) ``n_rep_gather``
             times, keeping every result on the CUDA device (NO ``.cpu().float()``
             round-trip, which would hide a W3 stream race).
          2. REFERENCE (non-circular): rank 0 loads the BASE model's per-expert
             weights independently from the on-disk HF checkpoint shards via
             ``safetensors.safe_open`` — a path that NEVER touches the EP gather.
          3. DIFFS each gathered expert row j vs the disk reference row j on GPU:
             max_abs, a cross-expert nearest-match (find disk m with gathered[j]==ref[m],
             m!=j => W1 swap), a prefix-block Δ test (rows off by a fixed shift => W2),
             gather-repeat determinism (gather1 vs gather2 => W3), and a dtype/byte check (W4).

        Collective contract: ``_gather_tensor`` is an all_gather over the (fsdp,ep)
        submesh, so EVERY rank must call it in the SAME order. All ranks gather; only
        rank 0 loads the disk reference + emits the signature (returns {} elsewhere).
        """
        import hashlib
        import json
        import os
        import socket

        rank = int(torch.distributed.get_rank())
        host = socket.gethostname()

        # --- locate THIS layer's grouped expert DTensor params (shim layout) ---
        # Keys look like ``...layers.{i}.mlp.moe.experts.w1`` (GroupedMoEShim) possibly
        # with an ``_fsdp_wrapped_module`` segment. Match by suffix on the stripped name.
        named = dict(self.model.model.named_parameters())

        def _strip(n):
            return n.replace("._fsdp_wrapped_module.", ".").replace(".mlp.moe.", ".mlp.")

        want_suffix = {
            "w1": f".layers.{layer_idx}.mlp.experts.w1",
            "w3": f".layers.{layer_idx}.mlp.experts.w3",
            "w2": f".layers.{layer_idx}.mlp.experts.w2",
        }
        found = {}
        for n, p in named.items():
            sn = _strip(n)
            for tag, suf in want_suffix.items():
                if sn.endswith(suf):
                    found[tag] = p
        # ---- ON-GPU gather, repeated, kept on device ----
        # ``_gather_tensor`` (= gather_dtensor_strided_safe) lives on the weight
        # EXTRACTOR, not the worker — it is the EXACT gather the broadcast path uses.
        gather_fn = self.weight_extractor._gather_tensor
        gathered = {tag: [] for tag in found}  # tag -> list of n_rep CUDA tensors
        placements_info = {}
        for rep in range(n_rep_gather):
            # Deterministic, all-ranks-identical iteration order over w1,w2,w3.
            for tag in ("w1", "w2", "w3"):
                if tag not in found:
                    continue
                p = found[tag]
                if rep == 0 and isinstance(p, DTensor):
                    placements_info[tag] = (str(p.placements), tuple(p.shape), str(p.dtype))
                g = gather_fn(p)  # ON-GPU gather (gather_dtensor_strided_safe)
                gathered[tag].append(g)  # keep on CUDA

        if rank != 0:
            # free + return (non-rank-0 still had to drive the collectives above)
            return {}

        # ---------------- rank 0: disk reference + signature ----------------
        out = {"rank": rank, "host": host, "layer": layer_idx,
               "placements": placements_info, "n_rep_gather": n_rep_gather,
               "lines": [], "verdict": None, "wrong_expert_map": {}}

        # Resolve the on-disk HF checkpoint shards (local cache or download).
        from huggingface_hub import snapshot_download
        from safetensors import safe_open

        local_dir = model_path
        if not (os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "config.json"))):
            local_dir = snapshot_download(
                model_path,
                allow_patterns=["*.safetensors", "*.json"],
            )

        # Build name -> shard-file index.
        idx_path = os.path.join(local_dir, "model.safetensors.index.json")
        if os.path.exists(idx_path):
            with open(idx_path) as f:
                weight_map = json.load(f)["weight_map"]
        else:
            # single-shard model
            single = os.path.join(local_dir, "model.safetensors")
            weight_map = None
            single_file = single

        proj_for = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}

        def _load_disk_expert(tag, j):
            """Load base disk weight for expert j of this layer/tag as a CUDA tensor."""
            key = f"model.layers.{layer_idx}.mlp.experts.{j}.{proj_for[tag]}.weight"
            if weight_map is not None:
                shard = os.path.join(local_dir, weight_map[key])
            else:
                shard = single_file
            with safe_open(shard, framework="pt", device="cpu") as fp:
                t = fp.get_tensor(key)
            return t.to(gathered[tag][0].device)

        def _row_hash(t):
            return hashlib.md5(t.detach().to(torch.float32).cpu().contiguous().numpy().tobytes()).hexdigest()[:12]

        n_experts = gathered["w1"][0].shape[0] if "w1" in gathered else 0
        out["num_experts"] = n_experts

        # Pre-load ALL disk gate_proj rows once (for the cross-expert nearest-match on w1).
        disk_w1_rows = None
        EPS = 1e-6  # bf16 round-trip epsilon; gathered base == disk base should be exact-ish

        for tag in ("w1", "w2", "w3"):
            if tag not in gathered:
                continue
            g0 = gathered[tag][0]
            g1 = gathered[tag][min(1, len(gathered[tag]) - 1)]
            # --- W3: gather determinism (gather0 vs gather1) ---
            det_max = float((g0.float() - g1.float()).abs().max().item())
            out["lines"].append(
                f"[L{layer_idx}.{tag}] shape={tuple(g0.shape)} dtype={g0.dtype} "
                f"gather-repeat max_abs(g0-g1)={det_max:.3e} "
                f"({'NON-DETERMINISTIC=>W3' if det_max > EPS else 'deterministic'})"
            )
            if tag == "w1":
                disk_w1_rows = [_load_disk_expert("w1", j).float() for j in range(n_experts)]

            n_corrupt = 0
            worst = (None, -1.0)
            for j in range(n_experts):
                gj = g0[j].float()
                ref = _load_disk_expert(tag, j).float()
                if tuple(gj.shape) != tuple(ref.shape):
                    out["lines"].append(f"    {tag}[{j}] SHAPE_MISMATCH g={tuple(gj.shape)} ref={tuple(ref.shape)}")
                    n_corrupt += 1
                    continue
                ma = float((gj - ref).abs().max().item())
                if ma > worst[1]:
                    worst = (j, ma)
                if ma <= EPS:
                    continue
                n_corrupt += 1
                extra = ""
                # --- W1: cross-expert nearest-match (does gathered[j] == disk[m], m!=j?) ---
                if tag == "w1" and disk_w1_rows is not None:
                    best_m, best_e = None, float("inf")
                    for m in range(n_experts):
                        e = float((disk_w1_rows[m] - gj).abs().max().item())
                        if e < best_e:
                            best_e, best_m = e, m
                    if best_m is not None and best_m != j and best_e <= EPS:
                        extra += f"  WRONG_EXPERT(carries disk expert {best_m})"
                        out["wrong_expert_map"][j] = best_m
                    elif best_m is not None:
                        extra += f"  closest_disk_expert={best_m}@{best_e:.2e}"
                # --- W2: prefix-block Δ (is gathered row j == disk row j+Δ for a fixed Δ?) ---
                gh, rh = _row_hash(gj), _row_hash(ref)
                extra += f"  ghash={gh} refhash={rh}"
                out["lines"].append(f"    {tag}[{j}] max_abs={ma:.3e}{extra}")
            out["lines"].append(
                f"[L{layer_idx}.{tag}] CORRUPT {n_corrupt}/{n_experts}  worst=expert{worst[0]}@{worst[1]:.3e}"
            )

        # --- W2 contiguous-shift detector: if wrong_expert_map is a constant offset Δ ---
        if out["wrong_expert_map"]:
            deltas = {(m - j) % n_experts for j, m in out["wrong_expert_map"].items()}
            if len(deltas) == 1:
                d = next(iter(deltas))
                out["lines"].append(f"[L{layer_idx}] CONSTANT row shift Δ={d} across ALL wrong experts => W2-style block shift")
            else:
                out["lines"].append(f"[L{layer_idx}] wrong-expert offsets are NON-uniform (deltas={sorted(deltas)}) => W1 strided permutation")

        # Verdict
        any_corrupt = any("CORRUPT" in l and not l.endswith("0/" + str(n_experts)) for l in out["lines"])
        total_corrupt = sum(1 for l in out["lines"] if l.strip().startswith(("w1[", "w2[", "w3[")))
        out["total_corrupt_rows"] = total_corrupt
        out["verdict"] = ("CLEAN (gathered==disk at EP=8 on-GPU => corruption is DOWNSTREAM: "
                          "NCCL broadcast or vLLM load_weights)") if total_corrupt == 0 else (
                          f"CORRUPT ({total_corrupt} expert rows differ from disk reference at EP=8 cross-node)")
        return out

    def init_model(self, model_path, num_training_steps: int = None):
        assert self.cfg.trainer.strategy in ("fsdp", "fsdp2")
        strategy = FSDPStrategy(
            fsdp_config=self.cfg.trainer.policy.fsdp_config,
            optimizer_config=self.cfg.trainer.policy.optimizer_config,
            model_config=self.cfg.trainer.policy.model,
            fsdp_strategy=self.cfg.trainer.strategy,
            seed=self.cfg.trainer.seed,
            micro_train_batch_size_per_gpu=self.cfg.trainer.micro_train_batch_size_per_gpu,
            num_training_steps=num_training_steps,
        )
        strategy.setup_distributed()
        self.strategy = strategy

        # Stage 3: surface the CP submesh/group on the worker so the Stage-4 forward wrap
        # can read it. cp_size==1 leaves both None (flag-off path untouched).
        self.cp_mesh = getattr(strategy, "cp_mesh", None)
        self.cp_group = getattr(strategy, "cp_group", None)

        self._is_lora = self.cfg.trainer.policy.model.lora.rank > 0

        # Update per-gpu mini batch size based on device mesh
        self._normalize_mini_batch_size()

        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.strategy.device_mesh
        )
        with init_context():

            wrapped_model = HFModelWrapper(
                model_path,
                use_flash_attention_2=self.cfg.trainer.flash_attn,
                # NOTE (sumanthrh): Model initialization should always be in fp32
                # during training
                bf16=True,
                lora_rank=self.cfg.trainer.policy.model.lora.rank,
                lora_alpha=self.cfg.trainer.policy.model.lora.alpha,
                lora_dropout=self.cfg.trainer.policy.model.lora.dropout,
                target_modules=self.cfg.trainer.policy.model.lora.target_modules,
                exclude_modules=self.cfg.trainer.policy.model.lora.exclude_modules,
                sequence_parallel_size=self.cfg.trainer.policy.sequence_parallel_size,
                use_sample_packing=self.cfg.trainer.use_sample_packing,
                use_torch_compile=self.cfg.trainer.policy.use_torch_compile,
                rope_scaling=get_rope_scaling_config(self.cfg.trainer),
                rope_theta=get_rope_theta_config(self.cfg.trainer),
                moe_router_replay=bool(self.cfg.trainer.policy.fsdp_config.get("moe_router_replay", False)),
                moe_grouped_gemm=bool(self.cfg.trainer.policy.fsdp_config.get("moe_grouped_gemm", False)),
                attn_backend=self.cfg.trainer.get("attn_backend", "auto"),
                context_parallel_size=int(self.cfg.trainer.policy.fsdp_config.get("context_parallel_size", 1)),
                # Stage 4: surface the CP submesh + rotate method so the forward
                # enters torch-native context_parallel (ring SDPA). None at cp=1.
                cp_mesh=self.cp_mesh,
                cp_rotate_method=str(self.cfg.trainer.policy.fsdp_config.get("cp_rotate_method", "allgather")),
            )
            # in-place patch
            self._seq_parallel_monkey_patch(model=wrapped_model.model)

            if self.cfg.trainer.gradient_checkpointing:
                wrapped_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={
                        "use_reentrant": self.cfg.trainer.gradient_checkpointing_use_reentrant
                    }
                )

        self.model, self.optimizer, self.scheduler = strategy.prepare(
            (wrapped_model, None, None),
        )
        assert (
            self.optimizer is not None and self.scheduler is not None
        ), "FSDP preparation should create optimizer and scheduler"

        # Initialize weight extractor
        self.use_cuda_ipc = self.cfg.generator.weight_sync_backend == "nccl" and self.cfg.trainer.placement.colocate_all
        # TODO(haochen): Now module grouping (in order to support FlashRL) is only enabled for the CUDA IPC
        # transfer strategy, we can enable it for other strategies as well.
        self.weight_extractor = FSDPWeightExtractor(
            self.model.model,
            group_by_module=self.use_cuda_ipc,
            batch_size_threshold_gb=(
                self.cfg.generator.weight_transfer_threshold_cuda_ipc_GB if self.use_cuda_ipc else 0.0
            ),
            moe_grouped_gemm=bool(self.cfg.trainer.policy.fsdp_config.get("moe_grouped_gemm", False)),
        )

    async def _save_lora_adapters_and_sync(self, peft_model, lora_sync_path, inference_engine_client):
        """Collect LoRA parameters, save and call inference engine to load."""
        import os
        import json
        from dataclasses import asdict
        from safetensors.torch import save_file
        from skyrl_train.distributed.fsdp_utils import collect_lora_params

        lora_params = collect_lora_params(module=self.model.model)

        if torch.distributed.get_rank() == 0:
            os.makedirs(lora_sync_path, exist_ok=True)

            peft_config = asdict(peft_model.peft_config.get("default", {}))
            peft_config["task_type"] = peft_config["task_type"].value
            peft_config["peft_type"] = peft_config["peft_type"].value
            peft_config["target_modules"] = list(peft_config["target_modules"])

            # Save LoRA parameters and config
            save_file(lora_params, os.path.join(lora_sync_path, "adapter_model.safetensors"))
            with io.open(os.path.join(lora_sync_path, "adapter_config.json"), "w", encoding="utf-8") as f:
                json.dump(peft_config, f, ensure_ascii=False, indent=4)

            # Send LoRA disk loading request to inference engine. `lora_disk_load` is a specific identifier
            # to tell the inference engine to extract the `lora_disk_path`.
            lora_request = {
                "names": ["lora_disk_load"],
                "extras": [{"lora_disk_path": lora_sync_path}],
            }
            await inference_engine_client.update_named_weights(lora_request)

        torch.distributed.barrier()

    async def broadcast_to_inference_engines(self, inference_engine_client):
        use_prefix_cache = self.cfg.generator.enable_prefix_caching
        generator_dtype = str_to_torch_dtype(self.cfg.generator.model_dtype)
        cache_reset_task = None
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            # clear prefix cache
            cache_reset_task = inference_engine_client.reset_prefix_cache()

        torch.cuda.empty_cache()

        # Check if this is a LoRA model
        peft_model = getattr(self.model.model, "_fsdp_wrapped_module", self.model.model)

        if self._is_lora:
            assert hasattr(peft_model, "peft_config"), "LoRA model should have peft_config"

            # assume base model is already synced, sync LoRA adapters
            lora_sync_path = self.cfg.trainer.policy.model.lora.lora_sync_path
            await self._save_lora_adapters_and_sync(peft_model, lora_sync_path, inference_engine_client)
            return

        # Extract weights using the initialized extractor
        import os

        _fuse_weights = os.environ.get("SKYRL_FUSE_WEIGHTS", "0") == "1"

        # #1685 fix (FlashInfer-CUTLASS w13 swap skipped on RL update -> MoE token-salad):
        # bracket the WHOLE multi-chunk sync with vLLM's layerwise reload so per-chunk
        # model.load_weights DEFER processing and a single finalize re-runs
        # process_weights_after_loading (re-applying swap_w13_to_w31) EXACTLY once. PROVEN
        # by the disagg kernel-format diag: without this the engine holds checkpoint
        # [gate;up] while the FlashInfer CUTLASS kernel reads [up;gate]. Inert (swap-wise)
        # on triton/dense backends, so byte-identical there. Gated by env for safety.
        _w13_bracket = (
            not self.use_cuda_ipc
            and not _fuse_weights
            and os.environ.get("SKYRL_W13_RELOAD_BRACKET", "1") == "1"
        )

        if not self.use_cuda_ipc:
            # Signal engines to start accumulating weights (for FP8 batched quantization)
            if _fuse_weights and torch.distributed.get_rank() == 0:
                await inference_engine_client.begin_weight_update()

            # Open the layerwise-reload bracket (rank 0 drives the engine RPC).
            if _w13_bracket and torch.distributed.get_rank() == 0:
                await inference_engine_client.begin_weight_reload()
            if _w13_bracket:
                torch.distributed.barrier()

            # Broadcast path: one chunk per parameter
            for chunk in self.weight_extractor.extract_weights(generator_dtype):
                # Each chunk contains one parameter
                assert len(chunk) == 1
                name = chunk.names[0]
                tensor = chunk.tensors[0]

                if torch.distributed.get_rank() == 0:
                    # Create legacy update request
                    update_weight_task = asyncio.create_task(
                        inference_engine_client.update_named_weights(
                            {
                                "names": [name],
                                "dtypes": [self.cfg.generator.model_dtype],
                                "shapes": [list(tensor.shape)],
                            }
                        )
                    )

                # Broadcast tensor
                def broadcast_tensor(tensor):
                    if torch.distributed.get_rank() == 0:
                        torch.distributed.broadcast(tensor.data, 0, group=self._model_update_group)

                await asyncio.to_thread(broadcast_tensor, tensor)
                if torch.distributed.get_rank() == 0:
                    await update_weight_task
                torch.distributed.barrier()

            # Flush accumulated weights (triggers FP8 quantization on receiver)
            if _fuse_weights and torch.distributed.get_rank() == 0:
                await inference_engine_client.end_weight_update()

            # Close the layerwise-reload bracket: finalize_layerwise_reload re-runs
            # process_weights_after_loading over every layer ONCE -> re-applies the
            # FlashInfer-CUTLASS w13 [gate;up]->[up;gate] swap the per-chunk loads skipped.
            if _w13_bracket:
                torch.distributed.barrier()
                if torch.distributed.get_rank() == 0:
                    await inference_engine_client.finish_weight_reload()
        else:
            # CUDA IPC path: batched chunks (batching handled by extractor)
            from torch.multiprocessing.reductions import reduce_tensor

            # Iterate over batched chunks
            for chunk in self.weight_extractor.extract_weights(generator_dtype):
                weights_update_request = {"names": [], "dtypes": [], "shapes": [], "extras": [], "packed": False}

                # Process all parameters in this batch
                # TODO(haochen): Pack tensors into contiguous buffer before creating IPC handle
                # (like Megatron does) to reduce number of IPC handles and file descriptors
                for name, tensor, shape in zip(chunk.names, chunk.tensors, chunk.shapes):
                    # Create IPC handle for tensor
                    ipc_handle = reduce_tensor(tensor)
                    ipc_handle = {get_physical_gpu_id(): ipc_handle}
                    ipc_handle_list = [None] * torch.distributed.get_world_size()
                    torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

                    if torch.distributed.get_rank() == 0:
                        ipc_handles = {}
                        for d in ipc_handle_list:
                            ipc_handles.update(d)

                        weights_update_request["names"].append(name)
                        weights_update_request["dtypes"].append(self.cfg.generator.model_dtype)
                        weights_update_request["shapes"].append(shape)
                        weights_update_request["extras"].append({"ipc_handles": ipc_handles})

                    torch.distributed.barrier()
                    torch.cuda.synchronize()

                # Send batch
                if torch.distributed.get_rank() == 0:
                    await inference_engine_client.update_named_weights(weights_update_request)
                    torch.cuda.ipc_collect()
                torch.distributed.barrier()
                torch.cuda.synchronize()

        if cache_reset_task is not None:
            await cache_reset_task
        torch.cuda.empty_cache()
        torch.distributed.barrier()

    def get_weight_statistics(self):
        """Compute lightweight statistics for model weights"""
        raise NotImplementedError()

    def _set_pad_token_id(self, pad_token_id):
        # NOTE (sumanthrh): self.model -> HFModelWrapper; self.model -> DeepSpeedEngine, self.model.module -> AutoModelForCausalLM
        self.model.model.config.pad_token_id = pad_token_id

    def forward(
        self,
        data: TrainingInputBatch,
    ) -> TrainingOutputBatch:
        """Run forward pass on data in inference mode.

        Reshard the model after forward pass to redistribute memory and allow for offloading to cpu.
        """
        output = super().forward(data)
        # unshard the root FSDP module (https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes)
        if self._world_size > 1 and fsdp_version(self.model.model) == 1:
            self.model.model._handle.reshard(True)
        return output


class FSDPCriticWorkerBase(CriticWorkerBase):
    def offload_to_cpu(self, pin_memory=True, non_blocking=True, offload_optimizer=True, offload_model=True):
        self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
        self.strategy.offload_to_cpu(
            self.model, self.optimizer, pin_memory, non_blocking, offload_optimizer, offload_model
        )

    def backload_to_gpu(self, non_blocking=True, backload_optimizer=True, backload_model=True):
        self.strategy.backload_to_gpu(self.model, self.optimizer, non_blocking, backload_optimizer, backload_model)

    def init_model(self, model_path, num_training_steps: int = None):
        assert self.cfg.trainer.strategy in ("fsdp", "fsdp2")
        strategy = FSDPStrategy(
            fsdp_config=self.cfg.trainer.critic.fsdp_config,
            optimizer_config=self.cfg.trainer.critic.optimizer_config,
            fsdp_strategy=self.cfg.trainer.strategy,
            seed=self.cfg.trainer.seed,
            micro_train_batch_size_per_gpu=self.cfg.trainer.micro_train_batch_size_per_gpu,
            num_training_steps=num_training_steps,
        )
        strategy.setup_distributed()
        self.strategy = strategy

        # Stage 3: surface the CP submesh/group on the worker (None when cp_size==1).
        self.cp_mesh = getattr(strategy, "cp_mesh", None)
        self.cp_group = getattr(strategy, "cp_group", None)

        # Update per-gpu mini batch size based on device mesh
        self._normalize_mini_batch_size()

        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.strategy.device_mesh
        )
        with init_context():
            critic = get_llm_for_sequence_regression(
                model_path,
                "critic",
                use_flash_attention_2=self.cfg.trainer.flash_attn,
                # NOTE (sumanthrh): Model initialization should always be in fp32
                # during training
                bf16=True,
                lora_rank=self.cfg.trainer.critic.model.lora.rank,
                lora_alpha=self.cfg.trainer.critic.model.lora.alpha,
                lora_dropout=self.cfg.trainer.critic.model.lora.dropout,
                target_modules=self.cfg.trainer.critic.model.lora.target_modules,
                exclude_modules=self.cfg.trainer.critic.model.lora.exclude_modules,
                value_head_prefix=self.cfg.trainer.algorithm.value_head_prefix,
                init_value_head=self.cfg.trainer.policy.model.path == self.cfg.trainer.critic.model.path,
                sequence_parallel_size=self.cfg.trainer.critic.sequence_parallel_size,
                use_sample_packing=self.cfg.trainer.use_sample_packing,
                attn_backend=self.cfg.trainer.get("attn_backend", "auto"),
                context_parallel_size=int(self.cfg.trainer.critic.fsdp_config.get("context_parallel_size", 1)),
                # Stage 4: value forward must CP-shard identically (G3). None at cp=1.
                cp_mesh=self.cp_mesh,
                cp_rotate_method=str(self.cfg.trainer.critic.fsdp_config.get("cp_rotate_method", "allgather")),
            )
            self._seq_parallel_monkey_patch(model=critic, use_parent_class=True)

            if self.cfg.trainer.gradient_checkpointing:
                critic.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={
                        "use_reentrant": self.cfg.trainer.gradient_checkpointing_use_reentrant
                    }
                )

        # prepare models/optimizers...
        self.model, self.optimizer, self.scheduler = strategy.prepare(
            (critic, None, None),
        )
        assert self.optimizer is not None

    def forward(
        self,
        data: TrainingInputBatch,
    ) -> TrainingOutputBatch:
        """Run forward pass on data in inference mode.

        Reshard the model after forward pass to redistribute memory and allow for offloading to cpu.
        """
        output = super().forward(data)
        # unshard the root FSDP module (https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes)
        if self._world_size > 1 and fsdp_version(self.model.model) == 1:
            self.model.model._handle.reshard(True)
        return output


class FSDPRefWorkerBase(RefWorkerBase):
    def offload_to_cpu(self, pin_memory=True, non_blocking=True, **kwargs):
        self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
        self.strategy.offload_to_cpu(self.model, None, pin_memory, non_blocking)

    def backload_to_gpu(self, non_blocking=True, **kwargs):
        self.strategy.backload_to_gpu(self.model, None, non_blocking)

    def init_model(self, model_path):
        assert self.cfg.trainer.strategy in ("fsdp", "fsdp2")
        strategy = FSDPStrategy(
            fsdp_config=self.cfg.trainer.ref.fsdp_config,
            fsdp_strategy=self.cfg.trainer.strategy,
            seed=self.cfg.trainer.seed,
            micro_train_batch_size_per_gpu=self.cfg.trainer.micro_train_batch_size_per_gpu,
        )
        strategy.setup_distributed()
        self.strategy = strategy

        # Stage 3: surface the CP submesh/group on the worker (None when cp_size==1).
        self.cp_mesh = getattr(strategy, "cp_mesh", None)
        self.cp_group = getattr(strategy, "cp_group", None)

        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.strategy.device_mesh
        )

        with init_context():
            wrapped_model = HFModelWrapper(
                model_path,
                use_flash_attention_2=self.cfg.trainer.flash_attn,
                bf16=self.cfg.trainer.bf16,
                sequence_parallel_size=self.cfg.trainer.ref.sequence_parallel_size,
                use_sample_packing=self.cfg.trainer.use_sample_packing,
                rope_scaling=get_rope_scaling_config(self.cfg.trainer),
                rope_theta=get_rope_theta_config(self.cfg.trainer),
                attn_backend=self.cfg.trainer.get("attn_backend", "auto"),
                context_parallel_size=int(self.cfg.trainer.ref.fsdp_config.get("context_parallel_size", 1)),
                # Stage 4: ref-logprob forward must CP-shard identically to the
                # policy so KL aligns post-unshard (G3). None at cp=1.
                cp_mesh=self.cp_mesh,
                cp_rotate_method=str(self.cfg.trainer.ref.fsdp_config.get("cp_rotate_method", "allgather")),
            )
            self._seq_parallel_monkey_patch(model=wrapped_model.model)

        self.model = strategy.prepare(wrapped_model)
        self.model.eval()

    def forward(
        self,
        data: TrainingInputBatch,
    ) -> TrainingOutputBatch:
        """Run forward pass on data in inference mode.

        Reshard the model after forward pass to redistribute memory and allow for offloading to cpu.
        """
        output = super().forward(data)
        # unshard the root FSDP module (https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes)
        if self._world_size > 1 and fsdp_version(self.model.model) == 1:
            self.model.model._handle.reshard(True)
        return output


# Ray remote actors
PolicyWorker = ray.remote(num_gpus=1)(FSDPPolicyWorkerBase)
CriticWorker = ray.remote(num_gpus=1)(FSDPCriticWorkerBase)
RefWorker = ray.remote(num_gpus=1)(FSDPRefWorkerBase)
