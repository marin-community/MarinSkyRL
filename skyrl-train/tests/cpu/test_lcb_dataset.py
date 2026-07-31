from omegaconf import DictConfig
import skyrl_gym

from examples.livecodebench.lcb_dataset import LIVECODEBENCH, process_example


REVERSE_SOLUTION = """```python
print(input()[::-1])
```"""


def test_lcb_example_builder_emits_a_launchable_contract_row():
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

    assert row is not None
    assert row["env_class"] == "lcb"
    assert "```python" in row["prompt"][0]["content"]

    env = skyrl_gym.make(
        row["env_class"],
        env_config=DictConfig({"env_class": row["env_class"]}),
        extras={"reward_model": row["reward_model"]},
    )
    assert env.step(REVERSE_SOLUTION)["reward"] == 1.0
