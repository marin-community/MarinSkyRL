"""Storage fields for prepared TaskSpecs."""

from dataclasses import dataclass

from taskcompendium.models import TaskSpec

TASK_SPEC_COLUMN = "task_spec"
TASKCOMPENDIUM_ENVIRONMENT = "taskcompendium"
HARBOR_ID_PREFIX = "harbor-id:"


@dataclass(frozen=True)
class CurriculumBin:
    name: str
    grade: int


@dataclass(frozen=True)
class TaskDatasetRow:
    task: TaskSpec
    data_source: str | None = None
    teacher_route: str | None = None
    metric_tags: tuple[str, ...] = ()
    curriculum_bin: CurriculumBin | None = None
