"""Shared state records for fully asynchronous rollout generation."""

from dataclasses import dataclass, field
from typing import List, Protocol

from skyrl_train.trajectory_runners.base import TrajectoryBatch


@dataclass
class GeneratedOutputGroup:
    """One prompt's rollout samples and the metadata needed to retry them."""

    trajectory_batch: TrajectoryBatch
    uid: str
    earliest_model_step: int
    source_prompts: List[dict]
    rollout_id: str | None = None
    rollout_store_path: str | None = None


@dataclass(frozen=True)
class RolloutReference:
    """A committed group whose payload lives in a FineStore archive."""

    rollout_id: str
    uid: str
    store_path: str


@dataclass
class GenerationBufferState:
    """Completed references, admitted groups, and retries stored with a checkpoint."""

    completed_groups: List[GeneratedOutputGroup]
    retry_prompts: List[List[dict]]
    admitted_groups: List[GeneratedOutputGroup] = field(default_factory=list)
    completed_rollouts: List[RolloutReference] = field(default_factory=list)

    def pending_uids(self) -> set[str]:
        """Return dataset UIDs whose work survives in this checkpoint."""
        uids = set()
        for group in self.completed_groups:
            if not isinstance(group.uid, str):
                raise ValueError("completed generation group uid must be a string")
            uids.add(group.uid)
        for reference in self.completed_rollouts:
            if not isinstance(reference.uid, str):
                raise ValueError("completed rollout reference uid must be a string")
            uids.add(reference.uid)
        for group in self.admitted_groups:
            if not isinstance(group.uid, str):
                raise ValueError("admitted generation group uid must be a string")
            uids.add(group.uid)
        for prompts in self.retry_prompts:
            for prompt in prompts:
                uid = prompt.get("uid")
                if not isinstance(uid, str):
                    raise ValueError("retry prompt uid must be a string")
                uids.add(uid)
        return uids


class GenerationQueuesProvider(Protocol):
    """Live generation queues that can provide checkpoint state."""

    def snapshot(self) -> GenerationBufferState: ...

    def shutdown_snapshot(self) -> GenerationBufferState: ...
