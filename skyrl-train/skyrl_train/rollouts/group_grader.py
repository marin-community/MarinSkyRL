"""SkyRL-owned group-grader specifications."""

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from taskcompendium.environment import ExternalVerifierSpec
from taskcompendium.models import TaskSpec, VerifierKind

from skyrl_gym.envs.nemotron_ultra import GENRM_AGENTS

GENRM_GROUP_GRADER = "nemotron_genrm"


class GroupGraderSpec(BaseModel):
    """A trainer-owned grader for a group of attempts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    parameters_json: str


class GenRMGroupGraderParameters(BaseModel):
    """Private inputs for the Nemotron GenRM group grader."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    principle: str = Field(min_length=1)
    agent: str
    config: dict[str, Any]


def task_group_grader(task: TaskSpec) -> GroupGraderSpec | None:
    """Derive a group grader from the SkyRL environment payload."""
    if task.verifier.kind != VerifierKind.EXTERNAL:
        return None
    verifier = ExternalVerifierSpec.model_validate_json(task.verifier.parameters_json)
    extras = verifier.parameters["extras"]
    config = verifier.parameters["config"]
    ultra = (extras.get("extra_info") or {}).get("nemotron_ultra") or {}
    if (
        task.environment.interaction != "nemotron_ultra"
        or ultra.get("agent") not in GENRM_AGENTS
        or config.get("grading") == "skip"
    ):
        return None
    record = json.loads(ultra["record_json"])
    principle = record.get("principle")
    if not isinstance(principle, str) or not principle.strip():
        raise ValueError("GenRM tasks require a non-empty grading principle")
    grading = dict(config.get("genrm", {}))
    if config.get("verifyit_enabled", False):
        grading["verifyit_enabled"] = True
    return GroupGraderSpec(
        name=GENRM_GROUP_GRADER,
        parameters_json=GenRMGroupGraderParameters(
            principle=principle,
            agent=ultra["agent"],
            config=grading,
        ).model_dump_json(),
    )
