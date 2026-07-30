import json

import pytest

from infra.rl_data.preparation import PreparationOptions, prepare_artifact, write_artifact, write_bundle
from infra.rl_data.sources import dapo_math_source, rlvr_ifeval_source


class FakeContract:
    def __init__(self, env_id, instruction=""):
        self.env_id = env_id
        self.prompt_instruction = instruction

    def normalize_ground_truth(self, ground_truth):
        if not ground_truth:
            raise ValueError("empty ground truth")
        if isinstance(ground_truth, dict):
            return json.dumps(ground_truth, sort_keys=True)
        return str(ground_truth).replace("\\boxed{", "").replace("}", "")

    def validate_example(self, ground_truth, positive_response, negative_response):
        normalized = self.normalize_ground_truth(ground_truth)
        if normalized not in positive_response or negative_response:
            raise ValueError("invalid verifier examples")
        return normalized


def test_dapo_preparation_strips_boilerplate_deduplicates_and_records_provenance():
    artifact = prepare_artifact(
        dapo_math_source(),
        [
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": "Solve the following math problem step by step.\n\n2 + 2\nRemember to put your answer",
                    }
                ],
                "reward_model": {"ground_truth": "\\boxed{4}"},
            },
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": "Solve the following math problem step by step.\n\n2 + 2\nRemember to put your answer",
                    }
                ],
                "reward_model": {"ground_truth": "\\boxed{4}"},
            },
        ],
        FakeContract("aime", " Answer: \\boxed{ANSWER}"),
        token_count=lambda text: len(text.split()),
        options=PreparationOptions(source_revision="fixture", max_prompt_tokens=20, minimum_unique_rows=1, seed=7),
    )

    assert artifact.rows == [
        {
            "data_source": "BytedTsinghua-SIA/DAPO-Math-17k",
            "prompt": [{"role": "user", "content": "2 + 2 Answer: \\boxed{ANSWER}"}],
            "env_class": "aime",
            "reward_model": {"ground_truth": "4"},
            "extra_info": {"split": "train", "index": 0},
        }
    ]
    assert artifact.provenance["counts"] == {"raw_rows": 2, "unique_rows": 1, "emitted_rows": 1}
    assert artifact.provenance["verification"] == "two_sided"


def test_preparation_rejects_overlength_prompts_before_artifact_write(tmp_path):
    with pytest.raises(ValueError, match="max_prompt_tokens"):
        prepare_artifact(
            dapo_math_source(),
            [
                {
                    "prompt": [
                        {
                            "role": "user",
                            "content": "Solve the following math problem step by step.\n\none two three\nRemember to put your answer",
                        }
                    ],
                    "reward_model": {"ground_truth": "4"},
                }
            ],
            FakeContract("aime", " Answer: \\boxed{ANSWER}"),
            token_count=lambda text: len(text.split()),
            options=PreparationOptions(source_revision="fixture", max_prompt_tokens=2, minimum_unique_rows=1),
        )

    assert not (tmp_path / "artifact").exists()


def test_preparation_rejects_dedup_collapse():
    examples = [
        {
            "prompt": [
                {
                    "role": "user",
                    "content": "Solve the following math problem step by step.\n\n2 + 2\nRemember to put your answer",
                }
            ],
            "reward_model": {"ground_truth": "4"},
        },
        {
            "prompt": [
                {
                    "role": "user",
                    "content": "Solve the following math problem step by step.\n\n2 + 2\nRemember to put your answer",
                }
            ],
            "reward_model": {"ground_truth": "4"},
        },
    ]

    with pytest.raises(ValueError, match="minimum_unique_rows"):
        prepare_artifact(
            dapo_math_source(),
            examples,
            FakeContract("aime", " Answer: \\boxed{ANSWER}"),
            token_count=lambda text: len(text.split()),
            options=PreparationOptions(source_revision="fixture", max_prompt_tokens=20, minimum_unique_rows=2),
        )


def test_ifeval_preparation_uses_its_constraint_contract():
    artifact = prepare_artifact(
        rlvr_ifeval_source(),
        [
            {
                "messages": [{"role": "user", "content": "Include alpha."}],
                "ground_truth": {"func_name": "verify_keywords", "keyword_list": ["alpha"]},
                "constraint_type": "keywords",
            }
        ],
        FakeContract("ifeval"),
        token_count=lambda text: len(text.split()),
        options=PreparationOptions(source_revision="fixture", max_prompt_tokens=20, minimum_unique_rows=1),
    )

    assert artifact.rows[0]["env_class"] == "ifeval"
    assert json.loads(artifact.rows[0]["reward_model"]["ground_truth"]) == {
        "func_name": "verify_keywords",
        "keyword_list": ["alpha"],
    }
    assert artifact.provenance["verification"] == "schema_only"


def test_artifact_writer_cleans_failed_staging_directory(tmp_path):
    artifact = prepare_artifact(
        dapo_math_source(),
        [
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": "Solve the following math problem step by step.\n\n2 + 2\nRemember to put your answer",
                    }
                ],
                "reward_model": {"ground_truth": "4"},
            }
        ],
        FakeContract("aime", " Answer: \\boxed{ANSWER}"),
        token_count=lambda text: len(text.split()),
        options=PreparationOptions(source_revision="fixture", max_prompt_tokens=20, minimum_unique_rows=1),
    )

    def fail_after_writing(rows, path):
        path.write_text(json.dumps(rows))
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError, match="disk full"):
        write_artifact(artifact, tmp_path / "artifact", parquet_writer=fail_after_writing)

    assert not (tmp_path / "artifact").exists()


def test_bundle_writer_publishes_train_validation_and_combined_provenance(tmp_path):
    train = prepare_artifact(
        dapo_math_source(),
        [
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": "Solve the following math problem step by step.\n\n2 + 2\nRemember to put your answer",
                    }
                ],
                "reward_model": {"ground_truth": "4"},
            }
        ],
        FakeContract("aime", " Answer: \\boxed{ANSWER}"),
        token_count=lambda text: len(text.split()),
        options=PreparationOptions(source_revision="fixture", max_prompt_tokens=20, minimum_unique_rows=1),
    )
    validation = prepare_artifact(
        dapo_math_source(),
        [
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": "Solve the following math problem step by step.\n\n3 + 3\nRemember to put your answer",
                    }
                ],
                "reward_model": {"ground_truth": "6"},
            }
        ],
        FakeContract("aime", " Answer: \\boxed{ANSWER}"),
        token_count=lambda text: len(text.split()),
        options=PreparationOptions(
            source_revision="fixture", max_prompt_tokens=20, minimum_unique_rows=1, artifact_split="validation"
        ),
    )

    def write_rows(rows, path):
        path.write_text(json.dumps(rows))

    write_bundle(train, validation, tmp_path / "artifact", parquet_writer=write_rows)

    assert (tmp_path / "artifact" / "train.parquet").exists()
    assert (tmp_path / "artifact" / "validation.parquet").exists()
    assert set(json.loads((tmp_path / "artifact" / "provenance.json").read_text())) == {"train", "validation"}
    assert validation.rows[0]["extra_info"]["split"] == "validation"
