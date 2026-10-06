from collections.abc import Sequence
from copy import deepcopy

from skyrl_train.rollouts.buffer import RolloutGroup
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
