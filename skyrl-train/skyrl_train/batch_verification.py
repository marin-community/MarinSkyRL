from collections.abc import Callable

from marinskyrl.runtime_options import BatchBuilder
from skyrl_train.batch_source import BatchSource, DriverBatchSource, WorkerBatchSource


def make_batch_source(
    mode: BatchBuilder,
    *,
    driver: DriverBatchSource,
    worker_factory: Callable[[], WorkerBatchSource],
) -> BatchSource:
    """Select one lifecycle implementation before training starts."""
    if mode is BatchBuilder.DRIVER:
        return driver
    return worker_factory()
