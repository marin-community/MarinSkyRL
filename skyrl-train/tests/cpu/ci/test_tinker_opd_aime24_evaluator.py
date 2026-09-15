import importlib.util
import sys
from pathlib import Path

import pytest
from datasets import Dataset

EVALUATOR_PATH = Path(__file__).parents[3] / "ci" / "opd" / "tinker_repro" / "evaluate_aime24.py"
SPEC = importlib.util.spec_from_file_location("tinker_opd_aime24_evaluator", EVALUATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
evaluator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluator
SPEC.loader.exec_module(evaluator)


def test_load_aime24_examples_pins_dataset_and_normalizes_rows() -> None:
    requested: dict[str, object] = {}

    def load_dataset(path: str, *, split: str, revision: str) -> Dataset:
        requested.update(path=path, split=split, revision=revision)
        return Dataset.from_dict(
            {
                "id": list(range(30)),
                "problem": [f" Problem {index} " for index in range(30)],
                "answer": [f" {index:03d} " for index in range(30)],
            }
        )

    examples = evaluator.load_aime24_examples(load_dataset)

    assert requested == {
        "path": "HuggingFaceH4/aime_2024",
        "split": "train",
        "revision": "2fe88a2f1091d5048c0f36abc874fb997b3dd99a",
    }
    assert examples[7] == evaluator.AIME24Example(
        example_id="aime_2024:7",
        problem="Problem 7",
        answer=7,
    )


def test_load_aime24_examples_rejects_dataset_drift() -> None:
    dataset = Dataset.from_dict({"id": [0], "problem": ["Problem"], "answer": ["123"]})

    with pytest.raises(ValueError, match="Expected 30 AIME 2024 examples, found 1"):
        evaluator.load_aime24_examples(lambda *args, **kwargs: dataset)


def test_validate_sampling_defaults_rejects_transitive_sdk_drift() -> None:
    class IncompatibleSamplingParams:
        top_p = 0.95
        top_k = 50

    with pytest.raises(RuntimeError, match="expected top_p=1.0 and top_k=-1"):
        evaluator.validate_sampling_defaults(IncompatibleSamplingParams)


@pytest.mark.parametrize(
    ("num_examples", "num_errors", "num_truncated", "expected_message"),
    [
        (29, 0, 0, "expected 30 scored samples, found 29"),
        (30, 1, 0, "encountered 1 sampling errors"),
        (30, 0, 2, "encountered 2 truncated responses"),
    ],
)
def test_validate_comparable_result_rejects_incomplete_evaluation(
    num_examples: int,
    num_errors: int,
    num_truncated: int,
    expected_message: str,
) -> None:
    config = evaluator.EvaluationConfig(
        checkpoint="tinker://released/sampler_weights/final",
        save_dir="artifacts/eval",
        max_examples=None,
        num_samples=1,
        concurrency=8,
    )

    with pytest.raises(RuntimeError, match=expected_message):
        evaluator.validate_comparable_result(
            config,
            num_examples=num_examples,
            num_errors=num_errors,
            num_truncated=num_truncated,
        )


def test_validate_comparable_result_accounts_for_smoke_limit_and_sample_count() -> None:
    config = evaluator.EvaluationConfig(
        checkpoint="tinker://released/sampler_weights/final",
        save_dir="artifacts/eval",
        max_examples=2,
        num_samples=4,
        concurrency=8,
    )

    evaluator.validate_comparable_result(
        config,
        num_examples=8,
        num_errors=0,
        num_truncated=0,
    )
