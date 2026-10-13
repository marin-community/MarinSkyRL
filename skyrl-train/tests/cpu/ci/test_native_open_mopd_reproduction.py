from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pytest
from skyrl_gym.source_task import session_parameters
from taskcompendium.models import TaskSpec

from marinskyrl.task_format import TASK_SPEC_COLUMN

SCRIPT_ROOT = Path(__file__).parents[3] / "ci" / "opd" / "open_mopd_repro"
SPEC = spec_from_file_location("open_mopd_dataset", SCRIPT_ROOT / "dataset.py")
assert SPEC is not None and SPEC.loader is not None
DATASET = module_from_spec(SPEC)
sys.modules["open_mopd_dataset"] = DATASET
SPEC.loader.exec_module(DATASET)


def test_open_mopd_rows_preserve_prompts_and_assign_explicit_teacher_routes():
    rows = [
        {
            "domain": "math",
            "data_source": "math_dapo_boxed",
            "prompt": [
                {"role": "system", "content": "Solve carefully."},
                {"role": "user", "content": "What is 2 + 2?"},
            ],
        },
        {
            "domain": "code",
            "data_source": "code_deepcoder",
            "prompt": [{"role": "user", "content": "Implement a sort."}],
        },
        {
            "domain": "if",
            "data_source": "if_nemotron",
            "prompt": [{"role": "user", "content": "Reply in JSON."}],
        },
    ]

    converted = DATASET.convert_rows(rows)

    prepared = converted.to_pylist()
    assert [row["teacher_route"] for row in prepared] == ["math", "code", "if"]
    tasks = [TaskSpec.model_validate_json(row[TASK_SPEC_COLUMN]) for row in prepared]
    assert [task.source.dataset for task in tasks] == ["math_dapo_boxed", "code_deepcoder", "if_nemotron"]
    assert [[event.content for event in task.context.events] for task in tasks] == [
        ["Solve carefully.", "What is 2 + 2?"],
        ["Implement a sort."],
        ["Reply in JSON."],
    ]
    assert [session_parameters(task).extras["source_index"] for task in tasks] == [0, 1, 2]


def test_open_mopd_rows_reject_unknown_teacher_domain():
    with pytest.raises(ValueError, match="unknown domain"):
        DATASET.convert_rows([{"domain": "swe", "prompt": [{"role": "user", "content": "Fix a bug."}]}])
