"""Retain every managed core and every outer DP actor's diagnostic response."""

import asyncio


def managed_core_identities(engine):
    """Validate the pinned client's local/external or internal-DP core set."""
    parallel = engine.vllm_config.parallel_config
    dp_rank = parallel.data_parallel_index
    count = parallel.data_parallel_size_local if parallel.local_engines_only else parallel.data_parallel_size
    ranks = [dp_rank] if parallel.data_parallel_rank_local is not None else list(range(dp_rank, dp_rank + count))
    expected = [rank.to_bytes(2, "little") for rank in ranks]
    core = engine.engine_core
    if (
        not ranks
        or any(rank < 0 or rank >= parallel.data_parallel_size for rank in ranks)
        or core.engine_ranks_managed != ranks
        or list(core.core_engines) != expected
    ):
        raise ValueError(
            "Receiver readback requires every configured managed DP core: "
            f"expected={ranks}, observed_ranks={core.engine_ranks_managed}, "
            f"observed_identities={[value.hex() for value in core.core_engines]}"
        )
    return expected, ranks


async def call_all_receiver_workers(engine, method: str, *, args: tuple = (), kwargs: dict | None = None):
    """Dispatch to every core this actor manages, retaining original worker results.

    The SkyRL factory creates one actor per external DP rank. Such an actor owns
    one managed core even though its native distributed world includes every DP
    rank. Internal DPLB clients instead manage multiple cores and deliberately
    return only the first public utility result, so use each private response.
    """
    if engine.vllm_config.parallel_config.data_parallel_size == 1:
        return await engine.collective_rpc(method, args=args, kwargs=kwargs)
    identities, ranks = managed_core_identities(engine)
    core = engine.engine_core
    per_core = await asyncio.gather(
        *[
            core._call_utility_async("collective_rpc", method, None, args, kwargs, engine=identity)
            for identity in identities
        ]
    )
    if any(not workers for workers in per_core):
        raise ValueError("Receiver core returned no worker readback")
    transport = {
        "core_client": type(core).__name__,
        "managed_dp_ranks": ranks,
        "managed_core_identities": [value.hex() for value in identities],
        "configured_dp_size": engine.vllm_config.parallel_config.data_parallel_size,
    }
    return [{**worker, "receiver_transport": transport} for workers in per_core for worker in workers]


def group_external_dp_workers(actor_rows: list[list[dict]], geometry: dict) -> list[list[dict]]:
    """Regroup the actual factory's ordered DP actors; reject missing/duplicate ranks."""
    dp = geometry["receiver_parallel"].get("data_parallel_size", 1)
    ranks_per_engine = geometry["receiver_ranks_per_engine"]
    if ranks_per_engine % dp or len(actor_rows) != geometry["receiver_engines"] * dp:
        raise ValueError("Receiver readback is missing configured external DP actors")
    ranks_per_actor = ranks_per_engine // dp
    engines = []
    for engine in range(geometry["receiver_engines"]):
        workers = []
        for dp_rank in range(dp):
            rows = actor_rows[engine * dp + dp_rank]
            expected = list(range(dp_rank * ranks_per_actor, (dp_rank + 1) * ranks_per_actor))
            if sorted(row["rank"] for row in rows) != expected:
                raise ValueError("External DP actor worker ranks differ from the configured factory slice")
            workers.extend(rows)
        engines.append(workers)
    return engines


class OrderedBucketDispatch:
    """Order diagnostic bucket RPCs before they enter synchronous vLLM workers.

    Ray and the async core client may reorder concurrent actor calls. A later
    bucket waits here, rather than blocking the worker that must receive its
    predecessor. The receipt joins CPU enqueue only; CUDA load events still
    govern buffer reuse and the final install completion.
    """

    def __init__(self, engine, manifest_id: str, bucket_count: int):
        self.engine = engine
        self.manifest_id = manifest_id
        self.bucket_count = bucket_count
        self.publication_id = None
        self.next_bucket = {False: 0, True: 0}
        self.pending = set()
        self.failure = None
        self.condition = asyncio.Condition()

    def begin(self, publication_id: int):
        self.require_idle()
        self.publication_id = publication_id
        self.next_bucket = {False: 0, True: 0}
        self.failure = None

    def require_idle(self):
        if self.pending:
            raise ValueError("Diagnostic bucket dispatch still has pending receives")

    async def receive(self, bucket_id: int, *, replay: bool, manifest_id, publication_id):
        active_manifest = self.manifest_id if self.publication_id is not None else None
        if manifest_id != active_manifest or publication_id != self.publication_id:
            raise ValueError("Bucket dispatch does not match the active manifest and publication")
        if type(bucket_id) is not int or not 0 <= bucket_id < self.bucket_count:
            raise ValueError("Bucket dispatch ID is outside the manifest")
        key = (replay, bucket_id)
        async with self.condition:
            if key in self.pending or bucket_id < self.next_bucket[replay]:
                raise ValueError("Duplicate diagnostic bucket dispatch")
            self.pending.add(key)
            try:
                await self.condition.wait_for(lambda: self.failure is not None or bucket_id == self.next_bucket[replay])
                if self.failure is not None:
                    raise RuntimeError(
                        "Diagnostic bucket dispatch failed earlier in this publication"
                    ) from self.failure
                result = await call_all_receiver_workers(
                    self.engine,
                    "receive_diagnostic_weight_sync_bucket",
                    args=(bucket_id,),
                    kwargs={"replay": replay, "manifest_id": manifest_id, "publication_id": publication_id},
                )
                self.next_bucket[replay] += 1
                return result
            except BaseException as error:
                self.failure = error
                raise
            finally:
                self.pending.remove(key)
                self.condition.notify_all()
