import json

import pyarrow.parquet as pq

from examples.livecodebench.lcb_dataset import LIVECODEBENCH, process_dataset
from skyrl_gym.source_task import session_parameters
from taskcompendium.submission import conversation_messages
from taskcompendium.models import CommandSemantics, EnvironmentRequirements, TaskSpec

from marinskyrl.task_format import TASK_SPEC_COLUMN


REVERSE_SOLUTION = """```python
print(input()[::-1])
```"""


def test_lcb_example_builder_preserves_executable_reference_tests(tmp_path):
    source = tmp_path / "test" / "code"
    source.mkdir(parents=True)
    (source / "livecodebench.json").write_text(
        json.dumps(
            [
                {
                    "problem": "Read one line and print it in reverse.",
                    "tests": [{"input": "abc\n", "output": "cba\n", "testtype": "stdin"}],
                    "completion": REVERSE_SOLUTION,
                },
            ]
        )
    )
    image = "python@sha256:" + "a" * 64
    process_dataset(
        LIVECODEBENCH,
        "test",
        str(tmp_path),
        str(tmp_path),
        environment=EnvironmentRequirements(command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=image),
    )
    row = pq.read_table(tmp_path / "test_livecodebench.parquet").to_pylist()[0]
    task = TaskSpec.model_validate_json(row[TASK_SPEC_COLUMN])
    public = conversation_messages(task.context.events)
    parameters = session_parameters(task)
    assert "```python" in public[0]["content"]
    assert "cba" not in public[0]["content"]
    assert task.environment_requirements.docker_image == image
    assert json.loads(parameters.extras["reward_model"]["ground_truth"]) == [
        {"input": "abc\n", "output": "cba\n", "testtype": "stdin"}
    ]
