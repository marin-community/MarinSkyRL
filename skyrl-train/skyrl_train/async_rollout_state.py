"""Shared state records for fully asynchronous rollout generation."""

from dataclasses import dataclass
from typing import List, Protocol

from skyrl_train.generators.base import GeneratorOutput


@dataclass
class GeneratedOutputGroup:
    """One prompt's rollout samples and the metadata needed to retry them."""

    generator_output: GeneratorOutput
    uid: str
    earliest_model_step: int
    source_prompts: List[dict]


@dataclass
class GenerationBufferState:
    """Completed rollout groups and pending source-prompt retries in a checkpoint."""

    completed_groups: List[GeneratedOutputGroup]
    retry_prompts: List[List[dict]]


class GenerationQueuesProvider(Protocol):
    """Live generation queues that can provide checkpoint state."""

    def snapshot(self) -> GenerationBufferState: ...
