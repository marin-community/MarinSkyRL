"""Dedicated opt-in worker; the normal WorkerWrap extension remains installed."""

from vllm.v1.worker.gpu_worker import Worker

from skyrl_train.distributed.weight_sync_environment import apply_weight_sync_environment


class InvariantWeightSyncWorker(Worker):
    def init_device(self):
        apply_weight_sync_environment(True, role="inference", rank=self.rank, local_rank=self.local_rank)
        return super().init_device()
