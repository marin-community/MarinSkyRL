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


@dataclass(frozen=True)
class RolloutBufferSnapshot:
    """Backend-owned state for pending work at a checkpoint boundary."""

    backend: str
    pending_uids: tuple[str, ...]
    state: object


@dataclass
class GenerationBufferState:
    """Completed buffer state, admitted groups, and retries stored with a checkpoint."""

    buffer: RolloutBufferSnapshot | None
    retry_prompts: List[List[dict]]
    admitted_groups: List[GeneratedOutputGroup] = field(default_factory=list)

    def has_pending_work(self) -> bool:
        return bool((self.buffer and self.buffer.pending_uids) or self.admitted_groups or self.retry_prompts)

    def pending_uids(self) -> set[str]:
        """Return dataset UIDs whose work survives in this checkpoint."""
        uids = set()
        if self.buffer is not None:
            for uid in self.buffer.pending_uids:
                if not isinstance(uid, str):
                    raise ValueError("completed rollout uid must be a string")
                uids.add(uid)
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
