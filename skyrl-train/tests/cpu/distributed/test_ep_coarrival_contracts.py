import asyncio
import threading

import pytest
import torch
from omegaconf import OmegaConf

from skyrl_train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.workers import worker as worker_module
from skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase
from skyrl_train.workers.worker import DistributedTorchRayActor, PolicyWorkerBase, Worker


class EntryBarrierReached(RuntimeError):
    pass


@pytest.mark.parametrize(
    ("world_size", "context_parallel_size", "expert_parallel_size", "data_parallel_size"),
    [
        (32, 2, 8, 2),
        (12, 1, 4, 3),
    ],
)
def test_worker_replicates_dispatch_data_across_ep_and_cp_ranks(
    monkeypatch,
    world_size,
    context_parallel_size,
    expert_parallel_size,
    data_parallel_size,
):
    monkeypatch.setattr(worker_module, "init_worker_process_group_with_device", lambda **_kwargs: None)
    monkeypatch.setattr(torch.distributed.device_mesh, "init_device_mesh", lambda *_args, **_kwargs: object())
    config = OmegaConf.create(
        {
            "trainer": {
                "policy": {
                    "fsdp_config": {
                        "context_parallel_size": context_parallel_size,
                        "expert_model_parallel_size": expert_parallel_size,
                    }
                }
            }
        }
    )

    mesh_ranks = []
    for rank in range(world_size):
        worker = object.__new__(DistributedTorchRayActor)
        worker._world_size = world_size
        worker._rank = rank
        worker.sequence_parallel_size = 1
        worker.cfg = config
        worker.init_worker_process_group()
        mesh_ranks.append(worker.mesh_rank)

    replication_width = context_parallel_size * expert_parallel_size
    assert [rank.dp for rank in mesh_ranks] == [rank // replication_width for rank in range(world_size)]
    assert [rank.sp for rank in mesh_ranks] == [rank % replication_width for rank in range(world_size)]
    assert {rank.dp_size for rank in mesh_ranks} == {data_parallel_size}
    assert [rank.dp for rank in mesh_ranks if rank.is_collection_dp_rank()] == list(range(data_parallel_size))


@pytest.mark.asyncio
async def test_fully_async_step_finishes_policy_drain_before_forward():
    events = []

    class TrainingStep:
        def __init__(self):
            self.cfg = OmegaConf.create(
                {
                    "trainer": {
                        "algorithm": {
                            "use_kl_in_reward": False,
                            "advantage_batch_normalize": False,
                        },
                        "dump_data_batch": False,
                    }
                }
            )
            self.all_timings = {}

        async def _drain_policy_event_loops(self):
            events.append("drain-start")
            await asyncio.sleep(0)
            events.append("drain-finished")

        def fwd_logprobs_values_reward(self, training_input):
            events.append("forward")
            return training_input

        def compute_advantages_and_returns(self, training_input):
            events.append("advantages")
            return training_input

        def train_critic_and_policy(self, training_input):
            events.append("train")
            return {"status": "ok"}

    training_input = TrainingInputBatch({"rewards": torch.ones(1)})
    training_input.metadata = {"uids": ["trial"]}

    status = await FullyAsyncRayPPOTrainer._run_training(TrainingStep(), training_input)

    assert status == {"status": "ok"}
    assert events == ["drain-start", "drain-finished", "forward", "advantages", "train"]


@pytest.mark.asyncio
async def test_worker_drain_runs_cuda_barrier_off_event_loop(monkeypatch):
    monkeypatch.setenv("SKYRL_WEIGHTSYNC_DRAIN_BARRIER", "1")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    event_loop_thread = threading.get_ident()
    events = []

    def synchronize():
        events.append(("cuda", threading.get_ident()))

    def barrier():
        events.append(("barrier", threading.get_ident()))

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(torch.distributed, "barrier", barrier)
    worker = object.__new__(Worker)
    worker._rank = 0
    worker._world_size = 2

    await worker.barrier_all()

    assert [event for event, _thread in events] == ["cuda", "barrier"]
    assert all(thread != event_loop_thread for _event, thread in events)
    assert len({thread for _event, thread in events}) == 1


@pytest.mark.asyncio
async def test_fsdp_forward_runs_blocking_body_off_event_loop(monkeypatch):
    monkeypatch.setenv("SKYRL_FORWARD_DISPATCH_FIX", "1")
    event_loop_thread = threading.get_ident()
    body_threads = []
    expected = object()
    worker = object.__new__(FSDPPolicyWorkerBase)
    worker._rank = 0
    worker._forward_impl = lambda _data: body_threads.append(threading.get_ident()) or expected

    result = await worker.forward(TrainingInputBatch({"sequences": torch.ones((1, 1), dtype=torch.long)}))

    assert result is expected
    assert body_threads and body_threads[0] != event_loop_thread


def test_decentralized_router_replay_training_enters_barrier_before_training(monkeypatch):
    monkeypatch.setenv("SKYRL_R3_RESIDENT", "1")
    monkeypatch.setenv("SKYRL_R3_DECENTRAL", "1")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    events = []

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("cuda"))

    def barrier():
        events.append("barrier")
        raise EntryBarrierReached

    monkeypatch.setattr(torch.distributed, "barrier", barrier)
    worker = object.__new__(PolicyWorkerBase)
    worker._rank = 0
    worker._world_size = 2
    training_input = TrainingInputBatch(
        {"rollout_routed_experts": torch.zeros((1, 1, 1, 1), dtype=torch.int16)}
    )

    with pytest.raises(EntryBarrierReached):
        worker.ppo_train(training_input)

    assert events == ["cuda", "barrier"]
