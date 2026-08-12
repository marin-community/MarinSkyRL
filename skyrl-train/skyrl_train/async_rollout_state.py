"""Shared state records for fully asynchronous rollout generation."""

from dataclasses import dataclass
from typing import List

from skyrl_train.generators.base import GeneratorOutput


@dataclass
class GeneratedOutputGroup:
    """One prompt's rollout samples and the metadata needed to retry them."""

    generator_output: GeneratorOutput
    uid: str
    global_step_when_scheduled: int
    source_prompts: List[dict]


@dataclass
class GenerationBufferState:
    """Completed rollout groups and pending source-prompt retries in a checkpoint."""

    completed_groups: List[GeneratedOutputGroup]
    retry_prompts: List[List[dict]]
