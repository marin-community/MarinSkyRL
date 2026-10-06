from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

import ray
import torch
from skyrl_train.distributed.dispatch import ActorInfo, MeshRank
from skyrl_train.rollouts.buffer import RolloutGroup
from skyrl_train.training_batch import TrainingOutputBatch
from skyrl_train.trajectory_runners.base import TrajectoryRunner
from skyrl_train.trajectory_runners.types import TrajectoryBatch, TrajectoryRequestBatch


class FixedPromptDataset:
    def __init__(self, uids: Sequence[str]):
        self.uids = tuple(uids)

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, index: int) -> str:
        return self.uid(index)

    def uid(self, index: int) -> str:
        return self.uids[index]

    def collate_fn(self, items: list[str]) -> list[dict]:
        return [{"uid": uid, "prompt": [], "env_class": None, "env_extras": {}} for uid in items]


class FixedRolloutRunner(TrajectoryRunner):
    """Serve specified groups at the rollout boundary; reject unexpected generation."""

    def __init__(self, groups: Sequence[RolloutGroup] = ()):
        self.groups = {group.uid: group.trajectory_batch for group in groups}

    async def _run(self, input_batch: TrajectoryRequestBatch, disable_tqdm: bool = False) -> TrajectoryBatch:
        uid = input_batch["trajectory_ids"][0].instance_id
        return deepcopy(self.groups[uid])


class FixedPolicyGroup:
    """Complete model RPCs with fixed outputs while the trainer owns the lifecycle."""

    def __init__(self, *args, **kwargs):
        self.actor_infos = [
            ActorInfo(handle=None, rank=MeshRank(dp=0, sp=0, tp=0, pp=0, world_size=1, dp_size=1, pp_size=1))
        ]

    def async_init_model(self, *args, **kwargs):
        return [ray.put(None)]

    def async_run_ray_method(self, dispatch, method, *args, data=None, **kwargs):
        if method == "forward":
            output = TrainingOutputBatch(
                {"output": torch.full((data["sequences"].shape[0], data.metadata["response_length"]), -0.75)}
            )
        elif method == "ppo_train":
            output = TrainingOutputBatch({})
            output.metadata = {"train_status": {"policy_loss": 0.0}}
        elif method == "save_checkpoint":
            directory = Path(kwargs["ckpt_dir"])
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "rank0.pt").write_bytes(b"policy")
            output = None
        elif method in {
            "_set_pad_token_id",
            "init_weight_sync_state",
            "barrier_all",
            "empty_cache",
            "wait_checkpoint_upload",
            "load_checkpoint",
        }:
            output = None
        else:
            raise NotImplementedError(method)
        return [ray.put(output)]

    async def async_run_method(self, dispatch, method, *args, **kwargs):
        if method != "broadcast_to_inference_engines":
            raise NotImplementedError(method)
        return [None]

    def backload_to_gpu(self, **kwargs):
        return []

    def offload_to_cpu(self, **kwargs):
        return []

    def kill_actors(self):
        pass


class FixedInferenceEngines:
    engines = ()

    async def pause_generation(self):
        pass

    async def resume_generation(self):
        pass

    async def teardown(self):
        pass

    def shutdown_http_endpoint(self):
        pass


class RecordingTracker:
    def __init__(self):
        self.logs = []

    def log(self, metrics, step, commit=True):
        self.logs.append((dict(metrics), step))
