"""Retain native shard preparation across configured learner publications."""

import asyncio
from copy import deepcopy
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
        self.proofs = config.get("proofs", True)
        if type(self.proofs) is not bool:
            raise ValueError("generator.shard_sync.proofs must be boolean")
        self.inline_diagnostic_updates = config.get("inline_diagnostic_updates", 0)
        if type(self.inline_diagnostic_updates) is not int or self.inline_diagnostic_updates < 0:
            raise ValueError("generator.shard_sync.inline_diagnostic_updates must be a nonnegative integer")
        self.before_observation = None
        self.pending_result = None
        self.deferred_rows = None
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
        if self.deferred_rows is not None:
            # The interval result gains finish/timing fields later. Preserve the
            # exact state at capture, never an alias to the mutable live result.
            self.deferred_rows.append(deepcopy(row))
            return None
        self.receipt_index += 1
        return persist_readback(self.output_uri, f"shard-driver-{self.preparation_id}-{self.receipt_index}", row)

    def diagnostics_outside_pause(self, publication_id):
        """Keep startup and full proof gates inline; optimize later proofs-off syncs."""
        return not self.proofs and publication_id > self.initial_publication + self.inline_diagnostic_updates

    async def before_pause(self, publication_id):
        """Read telemetry while generation is running, before entering the weight lease."""
        if not self.diagnostics_outside_pause(publication_id):
            return
        if self.pending_result is not None or self.before_observation is not None:
            raise ValueError("Previous shard diagnostics have not completed")
        if self.driver.inference_engine_client.generation_paused_event.is_set():
            raise ValueError("Shard before diagnostics require running inference")
        started = time.perf_counter()
        if self.measurement_marker is None:
            self.measurement_marker = (
                await settled(asyncio.to_thread(mark_measurement_once, self.output_uri, component="shard"))
            )[0]
        rows = await self.observe(publication_id, "before-pause")
        self.before_observation = (publication_id, rows, time.perf_counter() - started)

    async def after_resume(self, publication_id):
        """Persist completed-install evidence while generation can make progress."""
        if not self.diagnostics_outside_pause(publication_id):
            return
        if self.driver.inference_engine_client.generation_paused_event.is_set():
            raise ValueError("Shard durable completion requires resumed inference")
        if self.pending_result is None or self.pending_result[0] != publication_id:
            raise ValueError("Shard diagnostics lack a completed matching install")
        _, result, rows = self.pending_result
        started = time.perf_counter()
        result["physical_after"] = await self.observe(publication_id, "after-resume")
        result["outside_pause_seconds"]["observation_after"] = time.perf_counter() - started
        started = time.perf_counter()
        # Await I/O, so failure is visible and shutdown cannot lose writes. The
        # thread lets the driver's generation coroutines progress after resume.
        for row in rows:
            await settled(asyncio.to_thread(self.capture, row))
        result["outside_pause_seconds"]["interval_capture"] = time.perf_counter() - started
        started = time.perf_counter()
        result["durable_receipt"] = (
            await settled(
                asyncio.to_thread(
                    self.capture, {"phase": "publication-complete", "publication_id": publication_id, "result": result}
                )
            )
        )[0]
        result["outside_pause_seconds"]["completion_capture"] = time.perf_counter() - started
        self.driver.all_timings.update(
            {f"shard_sync/outside_pause/{name}": seconds for name, seconds in result["outside_pause_seconds"].items()}
        )
        self.pending_result = None

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
        # Native vLLM utility calls are serial with decode. Never perform object
        # store writes in those calls while generation is running.
        output_uri = None if moment in ("before-pause", "after-resume") else self.output_uri
        refs = self.driver.policy_model.async_run_ray_method(
            "pass_through", "read_weight_sync_observations", observation_id, output_uri
        )
        return await settled(
            settled(*refs),
            self.driver.inference_engine_client.read_weight_sync_observations(observation_id, output_uri),
        )

    async def publish(self, publication_id):
        started = time.perf_counter()
        outside_pause = self.diagnostics_outside_pause(publication_id)
        if self.pending_result is not None:
            raise ValueError("Previous shard diagnostics have not completed")
        if outside_pause:
            if self.before_observation is None or self.before_observation[0] != publication_id:
                raise ValueError("Shard before diagnostics are missing for this install")
            self.deferred_rows = []
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
                proofs=self.proofs,
                lifecycle=ShardLifecycle.RETAIN,
                generation_boundary=GenerationBoundary.DRIVER,
                capture=self.capture,
                observe=None if outside_pause else lambda moment: self.observe(publication_id, moment),
            )
            result["measurement_marker"] = self.measurement_marker
            result["total_seconds_including_proof"] = time.perf_counter() - started
            result["total_scope"] = (
                "controller preparation on first call and publication through finish with configured proofs; "
                "excludes completion receipt and outer driver pause/drain/resume"
            )
            result["diagnostics_scope"] = "outside-pause" if outside_pause else "paused-install"
            if outside_pause:
                _, result["physical_before"], seconds = self.before_observation
                result["outside_pause_seconds"] = {"observation_before": seconds}
                self.pending_result = (publication_id, result, self.deferred_rows)
                self.before_observation = None
                self.deferred_rows = None
                return result
            capture_started = time.perf_counter()
            result["durable_receipt"] = self.capture(
                {"phase": "publication-complete", "publication_id": publication_id, "result": result}
            )
            result["phase_seconds"]["completion_capture"] = time.perf_counter() - capture_started
            return result
        except BaseException as primary:
            # A failed installation stays paused. Retain the immutable partial
            # evidence before closing; success-path latency rules do not apply.
            rows, self.deferred_rows = self.deferred_rows, None
            for row in rows or ():
                try:
                    self.capture(row)
                except BaseException as error:
                    primary.add_note(f"Shard partial receipt: {type(error).__name__}: {error}")
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
        try:
            if self.pending_result is not None:
                publication_id, result, rows = self.pending_result
                self.pending_result = None
                for row in rows:
                    await settled(asyncio.to_thread(self.capture, row))
                await settled(
                    asyncio.to_thread(
                        self.capture,
                        {
                            "phase": "publication-diagnostics-incomplete",
                            "publication_id": publication_id,
                            "result": result,
                        },
                    )
                )
        finally:
            if self.context is not None:
                context, self.context = self.context, None
                await context.__aexit__(None, None, None)
