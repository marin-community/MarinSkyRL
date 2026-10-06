import asyncio
from collections.abc import Callable

from loguru import logger

from marinskyrl.runtime_options import BatchBuilder
from skyrl_train.batch_assembly import forward_input
from skyrl_train.batch_digest import BatchInputPhase, digest
from skyrl_train.batch_source import BatchDiagnostics, BatchSource, BatchSummary, DriverBatchSource, WorkerBatchSource
from skyrl_train.timing_observability import StepWallTime
from skyrl_train.utils import Timer


class VerifyBatchSource:
    """Compare all actor inputs with an isolated driver build before collective work."""

    def __init__(self, worker: WorkerBatchSource, driver: DriverBatchSource):
        self.worker = worker
        self.driver = driver

    async def admit(self, *, stall_timeout: float, diagnostics: BatchDiagnostics) -> dict[str, float]:
        return await self.worker.admit(stall_timeout=stall_timeout, diagnostics=diagnostics)

    async def prepare(self, *, global_step: int, step_wall: StepWallTime | None) -> None:
        await self.worker.prepare(global_step=global_step, step_wall=step_wall)
        metadata = self.worker.metadata
        groups = await self.worker.context.wait(
            self.worker.reader.read(metadata.batch_id, tuple(group.index for group in metadata.groups))
        )
        await self.driver.prepare_from_groups(
            groups,
            batch_id=metadata.batch_id,
            global_step=global_step,
            step_wall=None,
            diagnostics=BatchDiagnostics({}, {}),
        )

    async def _compare(self, phase: BatchInputPhase) -> None:
        batch = forward_input(self.driver.batch) if phase is BatchInputPhase.FORWARD else self.driver.batch
        expected = [digest(chunk) for chunk in batch.chunk(batch.batch_size // self.worker.plan.dp_size)]
        receipts = await self.worker.context.wait(
            asyncio.gather(
                *self.worker.policy.async_run_ray_method(
                    "pass_through", "digest_loaded", self.worker.plan.batch_id, phase
                )
            )
        )
        for actor, receipt in zip(self.worker.policy.actor_infos, receipts, strict=True):
            if receipt != expected[actor.rank.dp]:
                raise ValueError(f"worker {phase.value} input digest mismatch on rank {actor.rank}")
        logger.info("Worker {} input digests verified on {} ranks", phase.value, len(receipts))

    async def forward(self, *, step_wall: StepWallTime | None) -> None:
        await self._compare(BatchInputPhase.FORWARD)
        with Timer("verify_driver_preparation", self.worker.diagnostics.timings):
            await self.driver.forward(step_wall=None)
        await self.worker.forward(step_wall=step_wall)

    async def finalize(self, *, step_wall: StepWallTime | None) -> None:
        with Timer("verify_driver_finalization", self.worker.diagnostics.timings):
            await self.driver.finalize(step_wall=None)
        await self.worker.finalize(step_wall=step_wall)
        await self._compare(BatchInputPhase.TRAIN)
        await self.driver.dump(step_wall=None)

    def summary(self) -> BatchSummary:
        return self.worker.summary()

    async def train(self, *, step_wall: StepWallTime | None) -> dict:
        return await self.worker.train(step_wall=step_wall)

    async def release(self, error: BaseException | None = None) -> None:
        try:
            await self.worker.release(error)
        finally:
            await self.driver.release(error)


def make_batch_source(
    mode: BatchBuilder,
    *,
    driver: DriverBatchSource,
    worker_factory: Callable[[], WorkerBatchSource],
) -> BatchSource:
    """Select one lifecycle implementation before training starts."""
    if mode is BatchBuilder.DRIVER:
        return driver
    worker = worker_factory()
    return VerifyBatchSource(worker, driver) if mode is BatchBuilder.VERIFY else worker
