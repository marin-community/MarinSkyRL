"""Retain native shard preparation across configured learner publications."""

import time

import ray

from skyrl_train.weight_sync.readback_diagnostics import persist_readback
from skyrl_train.weight_sync.bucket_qualification import mark_measurement_once
from skyrl_train.weight_sync.shard_interval import GenerationBoundary, ShardLifecycle, run_shard_interval, settled
from skyrl_train.weight_sync.shard_preparation import PreparationOptions, ShardGeometry
from skyrl_train.weight_sync.shard_rendezvous import native_prepared_shard_diagnostic
from skyrl_train.weight_sync.shard_replay_rpc import replay_prepared_shards


class ShardTrainingPublication:
    def __init__(self, driver):
        self.driver = driver
        config = driver.cfg.generator.shard_sync
        self.geometry = ShardGeometry(
            config.policy_ranks,
            config.receiver_replicas,
            config.expert_parallel_size,
            tuple(tuple(layers) for layers in config.layers_by_pp),
            config.num_experts,
            config.hidden_size,
            config.intermediate_size,
        )
        self.geometry.validate()
        self.options = PreparationOptions(
            config.transfer_bytes, config.comparison_bytes, config.dense_chunk_bytes, config.minimum_free_bytes
        )
        self.options.validate()
        if not config.preparation_id or not config.output_uri:
            raise ValueError("Configured shard publication requires explicit preparation and output identities")
        if driver.cfg.generator.publication_stage_timing:
            raise ValueError("Shard phase receipts replace the legacy publication-stage timing adapter")
        self.preparation_id = config.preparation_id
        self.output_uri = config.output_uri
        self.timeout_seconds = config.timeout_seconds
        self.context = None
        self.plan = None
        self.manifest_id = None
        self.receipt_index = 0
        self.initial_publication = driver.global_step
        self.measurement_marker = None

    def capture(self, row):
        self.receipt_index += 1
        return persist_readback(self.output_uri, f"shard-driver-{self.preparation_id}-{self.receipt_index}", row)

    async def prepare(self):
        if self.context is not None:
            raise ValueError("Shard training publication is already prepared")
        context = native_prepared_shard_diagnostic(
            self.driver,
            self.preparation_id,
            self.geometry,
            self.options,
            store_node_id=ray.get_runtime_context().get_node_id(),
            backend="nccl",
            timeout_seconds=self.timeout_seconds,
            output_uri=self.output_uri,
            capture=self.capture,
        )
        started = time.perf_counter()
        plan, rows = await context.__aenter__()
        self.context = context
        self.plan = plan
        self.manifest_id = rows[0]["manifest_id"]
        self.driver.all_startup_timings["shard_one_time_preparation"] = time.perf_counter() - started

    async def replay(self, manifest_id, publication_id):
        return await replay_prepared_shards(
            self.driver,
            manifest_id,
            publication_id,
            policy_ranks=tuple(range(self.geometry.policy_ranks)),
            expected_receiver_bytes=self.plan.expected_receiver_bytes,
            expected_device_type="cuda",
            output_uri=self.output_uri,
            capture=self.capture,
        )

    async def observe(self, publication_id, moment):
        observation_id = f"shard-{publication_id}-{moment}"
        refs = self.driver.policy_model.async_run_ray_method(
            "pass_through", "read_weight_sync_observations", observation_id, self.output_uri
        )
        return await settled(
            settled(*refs),
            self.driver.inference_engine_client.read_weight_sync_observations(observation_id, self.output_uri),
        )

    async def publish(self, publication_id):
        started = time.perf_counter()
        try:
            if self.context is None:
                await self.prepare()
            if publication_id > self.initial_publication and self.measurement_marker is None:
                self.measurement_marker = mark_measurement_once(self.output_uri, component="shard")
            result = await run_shard_interval(
                self.driver,
                self.manifest_id,
                publication_id,
                replay=self.replay,
                policy_ranks=tuple(range(self.geometry.policy_ranks)),
                receiver_ranks=tuple(rank for rank, _ in self.plan.expected_receiver_bytes),
                expected_receiver_bytes=self.plan.expected_receiver_bytes,
                lifecycle=ShardLifecycle.RETAIN,
                generation_boundary=GenerationBoundary.DRIVER,
                capture=self.capture,
                observe=lambda moment: self.observe(publication_id, moment),
            )
            result["measurement_marker"] = self.measurement_marker
            result["total_seconds_including_proof"] = time.perf_counter() - started
            result["total_scope"] = (
                "controller preparation on first call and publication through verified finish; "
                "excludes completion receipt and outer driver pause/drain/resume"
            )
            result["durable_receipt"] = self.capture(
                {"phase": "publication-complete", "publication_id": publication_id, "result": result}
            )
            return result
        except BaseException as primary:
            try:
                self.capture(
                    {
                        "phase": "publication-failed",
                        "publication_id": publication_id,
                        "manifest_id": self.manifest_id,
                        "error_type": type(primary).__name__,
                        "error": str(primary)[:4096],
                    }
                )
            except BaseException as error:
                primary.add_note(f"Shard failure receipt: {type(error).__name__}: {error}")
            try:
                await self.close()
            except BaseException as error:
                primary.add_note(f"Shard training cleanup: {type(error).__name__}: {error}")
            raise

    async def close(self):
        if self.context is not None:
            context, self.context = self.context, None
            await context.__aexit__(None, None, None)
