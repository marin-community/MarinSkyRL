from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pytest

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

    assert converted.to_pylist() == [
        {
            "data_source": row["data_source"],
            "prompt": row["prompt"],
            "env_class": "prompt_only",
            "teacher_route": row["domain"],
            "source_index": index,
        }
        for index, row in enumerate(rows)
    ]


def test_open_mopd_rows_reject_unknown_teacher_domain():
    with pytest.raises(ValueError, match="unknown domain"):
        DATASET.convert_rows([{"domain": "swe", "prompt": [{"role": "user", "content": "Fix a bug."}]}])
