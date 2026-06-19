import ray
from loguru import logger
from packaging import version
from ray.actor import ActorHandle
from typing import Any, List, Dict
from ray.util.placement_group import PlacementGroupSchedulingStrategy, placement_group

from skyrl_train.inference_engines.base import (
    InferenceEngineInterface,
    InferenceEngineInput,
    InferenceEngineOutput,
    NamedWeightsUpdateRequest,
)
from skyrl_train.inference_engines.utils import get_rendezvous_addr_port


# ---------------------------------------------------------------------------
# #232 FIX B — NCCL flight-recorder observability env -> vLLM engine workers.
#
# The vLLM inference-engine actors (and, under the ray executor backend, the
# per-rank TP worker actors they spawn) do NOT reliably inherit the NCCL
# flight-recorder / watchdog env vars that are set on the host launch shell.
# On Jupiter those vars are exported as APPTAINERENV_TORCH_NCCL_* (so apptainer
# imports them into the `ray start` raylet process), but the Ray-actor-spawned
# vLLM EngineCore / TP workers run with a runtime_env that does NOT carry the
# raylet's TORCH_NCCL_* through to the actual collective-running process
# (job 919724: the execute_model wedge fired but NO nccl_fr_rank* dump was
# written -> NCCL's 600s watchdog never armed inside the worker, so vLLM's 900s
# RPC watchdog always preempted it). Without these vars IN the worker process
# env, a residual TP-rank desync can never write a flight-recorder trace.
#
# Fix: explicitly forward an allowlist of NCCL FR / debug vars from the
# engine-creation process env (which DOES have them, set by the launch shell)
# into the actor's Ray runtime_env env_vars. With
# placement_group_capture_child_tasks=True (already set on the scheduling
# strategy), the ray-backend TP worker actors inherit this runtime_env too, so
# the vars land IN the process that actually runs the NCCL collective. When NONE
# of these vars are set in the launching env (every non-#232 run), the dict is
# empty and runtime_env is None -> byte-identical actor creation as before.
_NCCL_FR_ENV_PASSTHROUGH = (
    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC",
    "TORCH_NCCL_ENABLE_MONITORING",
    "TORCH_NCCL_DUMP_ON_TIMEOUT",
    "TORCH_NCCL_TRACE_CPP_STACK",
    "TORCH_NCCL_DEBUG_INFO_TEMP_FILE",
    "TORCH_NCCL_DEBUG_INFO_PIPE_FILE",
    "TORCH_FR_BUFFER_SIZE",
    "TORCH_NCCL_TRACE_BUFFER_SIZE",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "TORCH_NCCL_BLOCKING_WAIT_TIMEOUT_MS",
    "NCCL_BLOCKING_WAIT",
)


def _build_inference_engine_runtime_env() -> Dict[str, Any] | None:
    """Forward NCCL flight-recorder/watchdog vars (#232 FIX B) from the launch
    process env into the vLLM engine actor's Ray runtime_env, so they reach the
    actual collective-running worker process. Returns None when none are set
    (byte-identical actor creation for every run that does not set them)."""
    import os

    env_vars = {k: os.environ[k] for k in _NCCL_FR_ENV_PASSTHROUGH if k in os.environ}
    if not env_vars:
        return None
    logger.info(f"#232 FIX B: forwarding NCCL FR env to vLLM engine actors via runtime_env: {sorted(env_vars)}")
    return {"env_vars": env_vars}


class RayWrappedInferenceEngine(InferenceEngineInterface):
    """
    A thin wrapper around a Ray ActorHandle to another InferenceEngineInterface.
    This class implements the InferenceEngineInterface by delegating calls to the remote actor.
    """

    def __init__(self, inference_engine_actor: ActorHandle):
        self.inference_engine_actor = inference_engine_actor

    def tp_size(self):
        # Diagnostic: unwrap un-pickleable Ray exceptions into a plain
        # RuntimeError. When a raylet dies (e.g. GPFS SIGBUS/ESTALE during
        # weight-sync-state init) Ray raises a dynamically-generated
        # RayTaskError(ActorDiedError) whose re-serialization across the dying
        # boundary surfaces as a PicklingError / pydantic_compat
        # ModuleNotFoundError red herring. Re-raising as a picklable plain
        # exception preserves the TRUE cause in logs. Happy path unchanged.
        try:
            return ray.get(self.inference_engine_actor.tp_size.remote())
        except ray.exceptions.RayError as e:
            raise RuntimeError(f"tp_size() failed at Ray boundary: {e!r}") from None

    def pp_size(self):
        return ray.get(self.inference_engine_actor.pp_size.remote())

    def dp_size(self):
        return ray.get(self.inference_engine_actor.dp_size.remote())

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        return await self.inference_engine_actor.generate.remote(input_batch=input_batch)

    async def wake_up(self, *args: Any, **kwargs: Any):
        return await self.inference_engine_actor.wake_up.remote(*args, **kwargs)

    async def sleep(self, *args: Any, **kwargs: Any):
        return await self.inference_engine_actor.sleep.remote(*args, **kwargs)

    async def init_weight_update_communicator(
        self, master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing: bool = False
    ):
        return await self.inference_engine_actor.init_weight_update_communicator.remote(
            master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing
        )

    async def update_named_weights(self, request: NamedWeightsUpdateRequest):
        return await self.inference_engine_actor.update_named_weights.remote(request)

    async def teardown(self):
        return await self.inference_engine_actor.teardown.remote()

    async def reset_prefix_cache(self):
        return await self.inference_engine_actor.reset_prefix_cache.remote()

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self.inference_engine_actor.chat_completion.remote(request_payload)

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self.inference_engine_actor.completion.remote(request_payload)

    async def abort_generation(self) -> None:
        return await self.inference_engine_actor.abort_generation.remote()

    async def get_stats(self) -> Dict[str, Any]:
        """Get vLLM engine statistics from the remote actor.

        Returns statistics about the inference engine including throughput,
        KV cache usage, and request counts. Used by VLLMStatsCallback.
        """
        return await self.inference_engine_actor.get_stats.remote()


def create_ray_wrapped_inference_engines(
    num_inference_engines: int,
    tensor_parallel_size: int,
    model_dtype: str,
    pretrain: str,
    seed: int,
    vllm_v1_disable_multiproc: bool,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    expert_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    data_parallel_size: int = 1,
    decode_context_parallel_size: int = 1,
    shared_pg=None,
    gpu_memory_utilization=None,
    inference_engine_enable_sleep=False,
    async_engine=False,
    max_num_batched_tokens=8192,
    max_num_seqs=1024,
    tokenizer=None,
    backend="vllm",
    sleep_level=2,  # we only set to 1 for unit tests that do not explicitly sync weights or for LoRA
    enable_lora=False,
    max_lora_rank=64,
    max_loras=1,
    fully_sharded_loras=False,
    engine_init_kwargs: Dict[str, Any] = {},
    rope_scaling: Dict[str, Any] = {},
    rope_theta: float | None = None,
    enable_ray_prometheus_stats: bool = False,
    max_logprobs: int = 1,
    mp_backend: bool = False,
) -> List[InferenceEngineInterface]:
    """
    Create a list of RayWrappedInferenceEngine instances wrapping Ray actor handles to InferenceEngineInterface instances.

    mp_backend: opt-in. When True (and TP>1 / PP>1 and NOT colocated), run each vLLM
        inference engine with the `mp` (multiprocessing) executor backend instead of `ray`.
        This is required for the Qwen3-Next-80B-A3B R3 router-capture path
        (`enable_return_routed_experts=true`): the vLLM Ray Compiled-DAG deadlocks on the
        hybrid (GatedDeltaNet + full-attn) arch when capture is on (rank-0 stuck in the DAG
        channel read at 0% GPU; reproduced + root-caused 2026-06-08). The `mp` executor has
        no Ray Compiled-DAG and runs the same config cleanly at full (cudagraph) speed.
        Default False => byte-identical behaviour for every other run. Only valid for
        non-colocated engines (each engine owns its own GPUs); colocated/hybrid engines
        still require the ray backend for shared-GPU resource management.
    """
    from skyrl_train.utils import ray_noset_visible_devices, get_all_env_variables, get_ray_pg_ready_with_timeout
    from skyrl_train.utils.utils import use_per_engine_strict_pack_pg
    from skyrl_train.utils.constants import SKYRL_RAY_PG_TIMEOUT_IN_S

    if backend == "vllm":
        import vllm
        from skyrl_train.inference_engines.vllm.vllm_engine import VLLMRayActor, AsyncVLLMRayActor

        # if a dev version is being used, skip the version check
        if "dev" not in vllm.__version__:
            assert version.parse(vllm.__version__) >= version.parse("0.8.3"), "SkyRL-Train only supports vLLM >= 0.8.3"
    elif backend == "sglang":
        # We import SGLang later to avoid importing vllm. See `get_sglang_engine` for more.
        pass
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    inference_engine_actors = []
    # #232 FIX B: NCCL flight-recorder env to forward into the engine actor (and,
    # via placement_group_capture_child_tasks, its ray-backend TP worker actors).
    # None for every run that does not set the TORCH_NCCL_* FR vars -> no change.
    inference_engine_runtime_env = _build_inference_engine_runtime_env()
    noset_visible_devices = ray_noset_visible_devices(ray.get(get_all_env_variables.remote()))
    use_hybrid_engine = shared_pg is not None
    tp_pp_size = tensor_parallel_size * pipeline_parallel_size
    # NOTE: we use the ray backend for tensor parallel size > 1 or pipeline parallel size > 1 to explicitly manage resource allocation
    # mp_backend (opt-in) lets a NON-colocated multi-GPU engine use vLLM's `mp` executor
    # instead, which avoids the Ray Compiled-DAG deadlock on the Qwen3-Next R3 capture path.
    # In that mode the single SkyRL actor owns the whole TP*PP GPU slice and vLLM spawns its
    # workers as local subprocesses (no per-worker Ray actors).
    use_mp_backend = bool(mp_backend) and tp_pp_size > 1 and not use_hybrid_engine
    if bool(mp_backend) and tp_pp_size > 1 and use_hybrid_engine:
        raise ValueError(
            "generator.inference_engine_mp_backend=true is only supported for NON-colocated "
            "inference engines (trainer.placement.colocate_all=false). Colocated engines need "
            "the ray backend for shared-GPU resource management."
        )
    if tensor_parallel_size == 1 and pipeline_parallel_size == 1:
        distributed_executor_backend = "uni"
    elif use_mp_backend:
        distributed_executor_backend = "mp"
    else:
        distributed_executor_backend = "ray"
    data_parallel_backend = "mp"
    # The vLLM `mp` executor REQUIRES v1 multiprocessing to spawn its TP worker
    # subprocesses. The default vllm_v1_disable_multiproc=true sets
    # VLLM_ENABLE_V1_MULTIPROCESSING=0, which kills the mp executor's shm message
    # queue at warm-up ("RuntimeError: cancelled" in shm_broadcast.dequeue ->
    # EngineCore init fail). Force it off for the mp backend so the workers run.
    if use_mp_backend and vllm_v1_disable_multiproc:
        logger.info(
            "mp_backend: overriding vllm_v1_disable_multiproc=True -> False "
            "(the mp executor needs VLLM_ENABLE_V1_MULTIPROCESSING=1 to spawn TP workers)."
        )
        vllm_v1_disable_multiproc = False
    # With the mp executor the single actor must hold ALL tp_pp_size GPUs itself (vLLM forks
    # its workers locally). With ray/uni the actor holds the GPUs per the original logic.
    if use_mp_backend:
        num_gpus_per_actor = tp_pp_size
    else:
        num_gpus_per_actor = int(tensor_parallel_size == 1 and pipeline_parallel_size == 1)

    if use_hybrid_engine and tensor_parallel_size == 1 and pipeline_parallel_size == 1:
        # Every worker will use 0.2 GPU, so that we can schedule
        # inference and training workers on the same GPUs.
        num_gpus_per_actor = 0.2

    per_engine_gpu_count = tensor_parallel_size * pipeline_parallel_size * data_parallel_size
    # #232 ROOT-CAUSE FIX (cross-node TP all-reduce decode deadlock): when an engine
    # spans MORE THAN ONE GPU (TP*PP > 1), create one PG PER ENGINE with STRICT_PACK,
    # NOT a single flat PACK PG over all engines.
    #
    # The flat `placement_group(<all bundles>, strategy="PACK")` is SOFT — Ray packs
    # bundles greedily to minimize node count but gives NO per-engine node-affinity:
    # a single engine's `tp_pp_size` contiguous {GPU:1} bundles can land split across
    # TWO nodes (observed: job 923995, a TP=4 engine straddling jpbo-091-30 + -38).
    # When that happens, vLLM detects the TP process group spans nodes, DISABLES
    # custom_all_reduce ("Custom allreduce is disabled because this process group
    # spans across nodes"), and every per-decode-step TP all-reduce goes over the IB
    # RDMA fabric instead of on-node NVLink. Under sustained 131k-context decode that
    # cross-node NCCL all-reduce deadlocks (rank spins count-32768 AllReduce while its
    # cross-node peers block on the RDMA transport) — exactly the Option-B wedge.
    #
    # Per-engine STRICT_PACK forces every engine's bundles onto ONE node (a STRICT_PACK
    # PG is atomic-per-node), restoring the intended "TP=4 = one 4-GPU node, on-node
    # NVLink all-reduce" guarantee. Bundle indices become engine-local (0..n-1).
    #
    # *** PLACEMENT-PG-STARVATION FIX (gate STRICT_PACK on tp_pp_size > 1) ***
    # The per-engine STRICT_PACK above is only NEEDED when an engine owns >1 GPU
    # (TP>1 or PP>1) — that's the only case with an on-node TP/PP all-reduce to
    # protect. For TP==PP==1 (single-GPU engines, e.g. lever1's 16 TP=1 engines and
    # swesmith's 48), each engine is ONE {GPU:1} bundle, so there is no intra-engine
    # all-reduce to keep on-node, and STRICT_PACK is actively HARMFUL: N independent
    # 1-bundle STRICT_PACK PGs scatter round-robin across nodes, leaving every node
    # PARTIALLY used. The downstream policy/ref worker PG (worker.py:_initiate_actors,
    # `placement_group([{GPU:4,CPU:4}]*policy_num_nodes, strategy="PACK")`) then can't
    # find its required whole 4-GPU nodes and dies with
    # `RuntimeError: Failed to create placement group (2 bundles, 8 GPUs) in 180s`
    # (confirmed: lever1 924882 / swesmith 924888, both multi-node TP=1, post-e5f0ff5).
    # A flat PACK over all single-GPU bundles packs them DENSELY (fills whole nodes,
    # leaves whole nodes free), so the policy PACK PG gets its nodes. So: TP==PP==1 ->
    # restore the original flat PACK; TP*PP>1 -> per-engine STRICT_PACK.
    # NOTE: the gate is `tp_pp_size > 1`, NOT `per_engine_gpu_count > gpus_per_node` —
    # #232 is TP=4 on 4-GPU nodes (4 is NOT > 4), which the latter would wrongly send
    # down the flat-PACK path and re-break the cross-node-TP-split bug.
    # For the multi-GPU-engine ray/uni case that could still scatter densely-packed
    # engines onto partially-used nodes (e.g. TP=2 on 4-GPU nodes), the policy PG is
    # protected independently by the `placement.policy_strict_spread_pg` reserve-first
    # mechanism (main_base.get_policy_pg), which claims the policy's whole nodes BEFORE
    # these engine PGs are created.
    per_engine_pgs: list = []
    use_per_engine_strict_pack = use_per_engine_strict_pack_pg(
        use_hybrid_engine=use_hybrid_engine,
        use_mp_backend=use_mp_backend,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
    )
    if not use_hybrid_engine:
        if use_mp_backend:
            # mp executor: each (engine, DP-rank) is ONE Ray actor that itself
            # reserves the whole TP*PP GPU slice and forks its workers locally.
            # The actor's resource request is {GPU: tp_pp_size}, so it must land in
            # a single bundle that big — one {GPU: tp_pp_size} bundle per DP rank
            # (NOT tp_pp_size separate {GPU:1} bundles, which an actor requesting
            # tp_pp_size GPUs cannot fit into; Ray's _validate_resource_shape
            # requires a single actor to fit one bundle). One bundle per DP rank
            # keeps the bundle count == num actors so each gets a distinct index.
            # The {GPU: tp_pp_size} bundle is already atomic-per-node, so the mp path
            # was never affected by the cross-node split; keep its single PACK PG.
            bundles = [
                {"GPU": tp_pp_size, "CPU": tp_pp_size}
                for _ in range(num_inference_engines * data_parallel_size)
            ]
            shared_pg = placement_group(bundles, strategy="PACK")
            get_ray_pg_ready_with_timeout(shared_pg, timeout=SKYRL_RAY_PG_TIMEOUT_IN_S)
        elif use_per_engine_strict_pack:
            # ray/uni backend, multi-GPU engines (TP*PP > 1): one STRICT_PACK PG per
            # engine so each engine's per_engine_gpu_count {GPU:1} bundles are
            # guaranteed co-located on a single node (no cross-node TP all-reduce in
            # decode). #232 fix.
            for _ in range(num_inference_engines):
                pg = placement_group(
                    [{"GPU": 1, "CPU": 1} for _ in range(per_engine_gpu_count)],
                    strategy="STRICT_PACK",
                )
                per_engine_pgs.append(pg)
            for pg in per_engine_pgs:
                get_ray_pg_ready_with_timeout(pg, timeout=SKYRL_RAY_PG_TIMEOUT_IN_S)
            # Keep `shared_pg` defined for downstream indexing; per-engine path
            # re-selects the engine's own PG in the loop.
            shared_pg = per_engine_pgs[0]
        else:
            # ray/uni backend, single-GPU engines (TP==PP==1): ONE flat PACK PG over
            # all engine {GPU:1} bundles (the original pre-#232 behavior). PACK packs
            # densely -> fills whole nodes -> leaves whole nodes free for the
            # downstream policy/ref PACK PG. Restores the multi-node disaggregated
            # behavior that the per-engine STRICT_PACK broke (lever1/swesmith).
            bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_inference_engines * per_engine_gpu_count)]
            shared_pg = placement_group(bundles, strategy="PACK")
            get_ray_pg_ready_with_timeout(shared_pg, timeout=SKYRL_RAY_PG_TIMEOUT_IN_S)

    for i in range(num_inference_engines):
        # Per-engine STRICT_PACK PGs (ray/uni, multi-GPU engines) are engine-LOCAL: each
        # has its own bundle index space 0..per_engine_gpu_count-1, so base_pg_index
        # resets to 0. The mp PACK PG, the TP==PP==1 flat PACK PG, and the hybrid/sglang
        # flat PG remain GLOBAL, so they keep the i*per_engine_gpu_count offset.
        use_per_engine_pg = bool(per_engine_pgs)
        if use_per_engine_pg:
            engine_pg = per_engine_pgs[i]
            base_pg_index = 0
        else:
            engine_pg = shared_pg
            base_pg_index = i * per_engine_gpu_count

        # Get DP group rendezvous (addr, port) on the same node as DP rank 0 for this engine.
        # The mp PACK PG has one {GPU: tp_pp_size} bundle per (engine, DP-rank), so the
        # rendezvous bundle for engine i's DP-rank-0 is at i*data_parallel_size (not the
        # per-GPU base_pg_index, which would index past the smaller mp bundle list).
        rendezvous_pg_index = (i * data_parallel_size) if use_mp_backend else base_pg_index
        data_parallel_address, data_parallel_rpc_port = get_rendezvous_addr_port(engine_pg, rendezvous_pg_index)

        if backend == "vllm":
            if async_engine:
                actor_class = AsyncVLLMRayActor
            else:
                actor_class = VLLMRayActor

            lora_kwargs = {
                "enable_lora": enable_lora,
                "max_lora_rank": max_lora_rank,
                "max_loras": max_loras,
                "fully_sharded_loras": fully_sharded_loras,
            }

            rope_engine_kwargs = {}
            if rope_scaling:
                rope_engine_kwargs["rope_scaling"] = rope_scaling
                if "max_model_len" not in engine_init_kwargs:
                    rope_factor = rope_scaling.get("factor", None)
                    rope_max_pos = rope_scaling.get("original_max_position_embeddings", None)
                    assert rope_factor is not None, "Please provide rope scaling `factor` to compute model max length"
                    assert (
                        rope_max_pos is not None
                    ), "Please provide rope `original_max_position_embeddings` to compute model max length"
                    rope_engine_kwargs["max_model_len"] = int(rope_factor * rope_max_pos)
            if rope_theta is not None:
                rope_engine_kwargs["rope_theta"] = rope_theta

            # Launch one actor per DP rank
            for dp_rank in range(data_parallel_size):

                # Contiguous TP*PP slice reserved for a single DP rank.
                base_dp_pg_index = base_pg_index + dp_rank * tp_pp_size
                dp_rank_bundles = (
                    list(range(base_dp_pg_index, base_dp_pg_index + tp_pp_size)) if tp_pp_size > 1 else None
                )
                if use_mp_backend:
                    # The mp executor's single actor reserves the whole TP*PP GPU slice itself
                    # (vLLM forks its workers locally, no per-worker Ray actors). It must land
                    # in ONE bundle holding tp_pp_size GPUs, so the mp PACK PG (built above) is
                    # one {GPU: tp_pp_size} bundle per (engine, DP-rank) and this actor is pinned
                    # to its own dedicated bundle (index = i*data_parallel_size + dp_rank). The
                    # whole-slice bundle keeps all TP workers co-located on one node. bundle_indices
                    # stays None so vLLM does not attempt ray per-worker placement.
                    dp_rank_bundles = None
                    dp_rank_sched = PlacementGroupSchedulingStrategy(
                        placement_group=engine_pg,
                        placement_group_capture_child_tasks=True,
                        placement_group_bundle_index=i * data_parallel_size + dp_rank,
                    )
                else:
                    dp_rank_sched = PlacementGroupSchedulingStrategy(
                        placement_group=engine_pg,
                        placement_group_capture_child_tasks=True,
                        placement_group_bundle_index=base_dp_pg_index,
                    )

                dp_kwargs = (
                    {
                        "data_parallel_backend": data_parallel_backend,
                        "data_parallel_size": data_parallel_size,
                        "data_parallel_rank": dp_rank,
                        "data_parallel_address": data_parallel_address,
                        "data_parallel_rpc_port": data_parallel_rpc_port,
                    }
                    if data_parallel_size > 1
                    else {}
                )

                # The mp executor's TP workers exchange custom-all-reduce IPC handles
                # under the Ray-actor placement + remapped CUDA_VISIBLE_DEVICES; vLLM's
                # custom all-reduce fails there with a CUDA "invalid argument" at
                # custom_all_reduce.cuh (worker dies at warm-up). Disable it for mp so
                # NCCL handles the TP all-reduce (correctness-equal, slightly slower).
                # Do NOT clobber an explicit engine_init_kwargs override: the de-risk /
                # long-ctx configs already pass `disable_custom_all_reduce=true` through
                # `++generator.engine_init_kwargs.disable_custom_all_reduce`, and emitting
                # it here too makes both expand into .remote() -> "got multiple values for
                # keyword argument 'disable_custom_all_reduce'" (TypeError, crashes engine
                # creation before any PG/training; confirmed de-risk 925650-925654).
                mp_extra_kwargs = (
                    {"disable_custom_all_reduce": True}
                    if (use_mp_backend and "disable_custom_all_reduce" not in engine_init_kwargs)
                    else {}
                )

                # vLLM Decode Context Parallel (DCP): only forward the kwarg when enabled
                # (> 1). decode_context_parallel_size is a native vLLM EngineArgs field, so
                # it flows straight through **kwargs into vllm.LLM / AsyncEngineArgs with no
                # new vLLM call site. When == 1 the kwarg is ABSENT, making the engine init
                # byte-identical to today (G1). DCP rides the TP GPUs and does NOT touch
                # per_engine_gpu_count or the PACK PG bundle math above (G4).
                dcp_kwargs = (
                    {"decode_context_parallel_size": decode_context_parallel_size}
                    if decode_context_parallel_size > 1
                    else {}
                )

                # #232 FIX B: attach runtime_env only when FR vars are present, so
                # actor creation is byte-identical for every non-#232 run.
                engine_options = dict(
                    num_cpus=num_gpus_per_actor,
                    num_gpus=num_gpus_per_actor,
                    scheduling_strategy=dp_rank_sched,
                )
                if inference_engine_runtime_env is not None:
                    engine_options["runtime_env"] = inference_engine_runtime_env
                engine = actor_class.options(**engine_options).remote(
                    model=pretrain,
                    enforce_eager=enforce_eager,
                    worker_extension_cls="skyrl_train.inference_engines.vllm.vllm_engine.WorkerWrap",
                    tensor_parallel_size=tensor_parallel_size,
                    pipeline_parallel_size=pipeline_parallel_size,
                    enable_expert_parallel=expert_parallel_size > 1,
                    distributed_executor_backend=distributed_executor_backend,
                    seed=seed + i * data_parallel_size + dp_rank,
                    enable_prefix_caching=enable_prefix_caching,
                    dtype=model_dtype,
                    trust_remote_code=True,
                    vllm_v1_disable_multiproc=vllm_v1_disable_multiproc,
                    gpu_memory_utilization=gpu_memory_utilization,
                    bundle_indices=dp_rank_bundles,
                    num_gpus=0.2 if use_hybrid_engine else 1,
                    enable_sleep_mode=inference_engine_enable_sleep,
                    noset_visible_devices=noset_visible_devices,
                    max_num_batched_tokens=max_num_batched_tokens,
                    max_num_seqs=max_num_seqs,
                    max_logprobs=max_logprobs,
                    enable_ray_prometheus_stats=enable_ray_prometheus_stats,
                    **dp_kwargs,
                    **mp_extra_kwargs,
                    **dcp_kwargs,
                    **engine_init_kwargs,
                    **lora_kwargs,
                    **rope_engine_kwargs,
                )
                inference_engine_actors.append(engine)
        elif backend == "sglang":
            # NOTE: there is no async / sync engine distinction in SGLang

            # Per-engine STRICT_PACK PG uses engine-local bundle indices (0-based);
            # the legacy flat PG keeps the global i*per_engine_gpu_count offset.
            sglang_base_index = 0 if use_per_engine_pg else i * per_engine_gpu_count
            bundle_indices = None
            if per_engine_gpu_count > 1:
                bundle_indices = list(range(sglang_base_index, sglang_base_index + per_engine_gpu_count))

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=engine_pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=sglang_base_index,
            )

            # NOTE(Charlie): We need `torch.cuda.is_available()` to be True to import SGLang. Otherwise, it requires
            # importing vllm. See https://github.com/sgl-project/sglang/blob/v0.4.8.post1/python/sglang/srt/layers/quantization/utils.py#L11-L17
            # Similar comment: https://github.com/volcengine/verl/blob/9cc307767b0c787e8f5ef581dac929f7bde044ef/verl/workers/fsdp_workers.py#L520-L527
            @ray.remote
            def get_sglang_engine():
                # A workaround to avoid importing vllm is to give this task a GPU.
                import os

                before_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                from skyrl_train.inference_engines.sglang.sglang_engine import SGLangRayActor

                os.environ["CUDA_VISIBLE_DEVICES"] = before_cuda_visible_devices

                actor_class = SGLangRayActor
                engine = actor_class.options(
                    num_cpus=num_gpus_per_actor,
                    num_gpus=num_gpus_per_actor,
                    scheduling_strategy=scheduling_strategy,
                ).remote(
                    model_path=pretrain,
                    tp_size=tensor_parallel_size,
                    mem_fraction_static=gpu_memory_utilization,
                    random_seed=seed + i,
                    disable_radix_cache=not enable_prefix_caching,
                    dtype=model_dtype,
                    trust_remote_code=True,
                    max_prefill_tokens=max_num_batched_tokens,
                    max_running_requests=max_num_seqs,
                    # Borrowed from veRL's SGLang rollout
                    mm_attention_backend="fa3",
                    attention_backend="fa3",
                    enable_memory_saver=inference_engine_enable_sleep,
                    # Will be popped before instantiating sgl.Engine
                    distributed_executor_backend=distributed_executor_backend,
                    noset_visible_devices=noset_visible_devices,
                    bundle_indices=bundle_indices,
                    num_gpus=0.2 if use_hybrid_engine else 1,
                    tokenizer=tokenizer,
                    **engine_init_kwargs,
                )
                return engine

            engine = ray.get(get_sglang_engine.remote())

            inference_engine_actors.append(engine)

    engines = [RayWrappedInferenceEngine(actor_handle) for actor_handle in inference_engine_actors]

    if inference_engine_enable_sleep:
        if backend == "vllm":
            # NOTE(shu): set to 1 for LoRA
            sleep_level = 1 if enable_lora else sleep_level
            sleep_refs = [engine.inference_engine_actor.sleep.remote(level=sleep_level) for engine in engines]
        elif backend == "sglang":
            # NOTE(Charlie): we always need to sync weights after waking up: https://github.com/sgl-project/sglang/issues/7939
            assert sleep_level == 2, "SGLang always discards weights, so sleep_level is not applicable."
            sleep_refs = [engine.inference_engine_actor.sleep.remote() for engine in engines]
        ray.get(sleep_refs)

    return engines
