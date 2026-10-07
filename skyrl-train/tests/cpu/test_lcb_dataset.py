import json

from examples.livecodebench.lcb_dataset import LIVECODEBENCH, process_example
from taskcompendium.importers.skyrl import ExternalVerifierSpec
from taskcompendium.submission import conversation_messages

from skyrl_train.dataset.tasks import source_row_task
from tests.cpu.task_specs import session_spec


REVERSE_SOLUTION = """```python
print(input()[::-1])
```"""


def test_lcb_example_builder_preserves_executable_reference_tests():
    row = process_example(
        {
            "problem": "Read one line and print it in reverse.",
            "tests": [{"input": "abc\n", "output": "cba\n", "testtype": "stdin"}],
            "completion": REVERSE_SOLUTION,
        },
        idx=3,
        dataset_name=LIVECODEBENCH,
        split="test",
    )

    task = source_row_task(
        row,
        0,
        source_name=LIVECODEBENCH,
        environment_configs={"session": session_spec().model_dump(exclude={"task_session"})},
    ).task
    public = conversation_messages(task.context)
    verifier = ExternalVerifierSpec.model_validate_json(task.verifier.parameters_json)
    assert "```python" in public[0]["content"]
    assert "cba" not in public[0]["content"]
    assert json.loads(verifier.parameters["extras"]["reward_model"]["ground_truth"]) == [
        {"input": "abc\n", "output": "cba\n", "testtype": "stdin"}
    ]
