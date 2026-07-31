import json

import pytest

from infra.rl_data.preparation import PreparationOptions, prepare_artifact, write_artifact, write_bundle
from infra.rl_data.sources import (
    apps_source,
    dapo_math_source,
    deepscaler_source,
    gpqa_source,
    gsm8k_source,
    hh_rlhf_source,
    kto_mix_source,
    openscience_source,
    rlvr_math_source,
    rlvr_ifeval_source,
    verifiable_code_source,
)


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


def test_preparation_skips_a_majority_of_malformed_rows_with_provenance() -> None:
    examples = [
        {"messages": [{"role": "user", "content": "Question: bad one"}], "ground_truth": ""},
        {"messages": [{"role": "user", "content": "Question: 2 + 2"}], "ground_truth": "4"},
        {"messages": [{"role": "user", "content": "Question: bad two"}], "ground_truth": ""},
    ]

    with pytest.warns(UserWarning, match=r"skipped 2 of 3 rows"):
        artifact = prepare_artifact(
            rlvr_math_source(),
            examples,
            FakeContract("aime", " Answer: \\boxed{ANSWER}"),
            token_count=lambda text: len(text.split()),
            options=PreparationOptions(source_revision="fixture", max_prompt_tokens=20, minimum_unique_rows=1),
        )

    assert len(artifact.rows) == 1
    assert artifact.provenance["counts"]["malformed_rows_skipped"] == 2
    assert artifact.provenance["conversion_failures"] == [
        {"index": 0, "error": "ValueError: empty ground truth"},
        {"index": 2, "error": "ValueError: empty ground truth"},
    ]


def test_preparation_can_require_a_minimum_conversion_yield() -> None:
    examples = [
        {"messages": [{"role": "user", "content": "Question: bad one"}], "ground_truth": ""},
        {"messages": [{"role": "user", "content": "Question: 2 + 2"}], "ground_truth": "4"},
        {"messages": [{"role": "user", "content": "Question: bad two"}], "ground_truth": ""},
    ]

    with pytest.warns(UserWarning, match=r"skipped 2 of 3 rows"):
        with pytest.raises(ValueError, match=r"conversion yield 1/3 .* below minimum_yield_fraction=0.5"):
            prepare_artifact(
                rlvr_math_source(),
                examples,
                FakeContract("aime", " Answer: \\boxed{ANSWER}"),
                token_count=lambda text: len(text.split()),
                options=PreparationOptions(
                    source_revision="fixture",
                    max_prompt_tokens=20,
                    minimum_unique_rows=1,
                    minimum_yield_fraction=0.5,
                ),
            )


@pytest.mark.parametrize("minimum_yield_fraction", [-0.01, 1.01])
def test_preparation_rejects_an_invalid_minimum_yield_fraction(minimum_yield_fraction) -> None:
    with pytest.raises(ValueError, match="minimum_yield_fraction must be between 0 and 1"):
        prepare_artifact(
            rlvr_math_source(),
            [{"messages": [{"role": "user", "content": "Question: 2 + 2"}], "ground_truth": "4"}],
            FakeContract("aime", " Answer: \\boxed{ANSWER}"),
            token_count=lambda text: len(text.split()),
            options=PreparationOptions(
                source_revision="fixture",
                max_prompt_tokens=20,
                minimum_unique_rows=1,
                minimum_yield_fraction=minimum_yield_fraction,
            ),
        )


class SchemaOnlyCodeContract:
    env_id = "lcb"
    prompt_instruction = "\nReturn fenced Python."

    def normalize_ground_truth(self, ground_truth):
        return json.dumps(ground_truth, sort_keys=True)

    def validate_example(self, ground_truth, positive_response, negative_response):
        raise AssertionError("Code-source preparation must not execute downloaded solutions.")


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
    assert artifact.provenance["counts"] == {
        "raw_rows": 2,
        "converted_rows": 2,
        "malformed_rows_skipped": 0,
        "unique_rows": 1,
        "emitted_rows": 1,
    }
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


# ---------------------------------------------------------------------------
# New sources
# ---------------------------------------------------------------------------

_OPTS = {"source_revision": "fixture", "max_prompt_tokens": 200, "minimum_unique_rows": 1}


def test_deepscaler_preparation():
    artifact = prepare_artifact(
        deepscaler_source(),
        [{"problem": "What is 2 + 2?", "answer": "4", "solution": "2+2=4"}],
        FakeContract("aime", " Answer: \\boxed{ANSWER}"),
        token_count=lambda text: len(text.split()),
        options=PreparationOptions(**_OPTS),
    )
    assert artifact.rows[0]["data_source"] == "agentica-org/DeepScaleR-Preview-Dataset"
    assert artifact.rows[0]["reward_model"]["ground_truth"] == "4"
    assert artifact.rows[0]["env_class"] == "aime"


def test_gsm8k_preparation_strips_answer_delimiter():
    artifact = prepare_artifact(
        gsm8k_source(),
        [{"question": "What is 3 * 4?", "answer": "3 * 4 = 12\n#### 12"}],
        FakeContract("gsm8k", "\nThe final answer must appear after ####"),
        token_count=lambda text: len(text.split()),
        options=PreparationOptions(**_OPTS),
    )
    assert artifact.rows[0]["reward_model"]["ground_truth"] == "12"
    assert artifact.rows[0]["env_class"] == "gsm8k"


def test_gsm8k_rejects_missing_delimiter():
    with pytest.raises(ValueError, match="####"):
        prepare_artifact(
            gsm8k_source(),
            [{"question": "Bad row", "answer": "no delimiter here"}],
            FakeContract("gsm8k", "\n####"),
            token_count=lambda text: len(text.split()),
            options=PreparationOptions(**_OPTS),
        )


def test_verifiable_code_preparation():
    contract = SchemaOnlyCodeContract()
    artifact = prepare_artifact(
        verifiable_code_source(),
        [
            {
                "problem_statement": "Write a function that adds two numbers.",
                "verification_info": (
                    "{'language': 'python', 'test_cases': "
                    "[{'fn_name': None, 'input': '1 2\\n', 'output': '3\\n', 'type': 'stdin_stdout'}]}"
                ),
            }
        ],
        contract,
        token_count=lambda text: len(text.split()),
        options=PreparationOptions(**_OPTS),
    )
    assert artifact.rows[0]["data_source"] == "open-r1/verifiable-coding-problems-python"
    assert "test_cases" in artifact.rows[0]["reward_model"]["ground_truth"]
    assert artifact.rows[0]["prompt"][0]["content"].endswith(contract.prompt_instruction)
    assert artifact.provenance["verification"] == "schema_only"


def test_apps_preparation():
    contract = SchemaOnlyCodeContract()
    artifact = prepare_artifact(
        apps_source(),
        [
            {
                "question": "Reverse a string.",
                "input_output": '{"inputs": ["abc\\n"], "outputs": ["cba\\n"]}',
            }
        ],
        contract,
        token_count=lambda text: len(text.split()),
        options=PreparationOptions(**_OPTS),
    )
    assert artifact.rows[0]["data_source"] == "codeparrot/apps"
    assert "inputs" in artifact.rows[0]["reward_model"]["ground_truth"]
    assert artifact.rows[0]["prompt"][0]["content"].endswith(contract.prompt_instruction)
    assert artifact.provenance["verification"] == "schema_only"


def test_gpqa_preparation_builds_mcq():
    artifact = prepare_artifact(
        gpqa_source(),
        [
            {
                "Question": "What is the speed of light?",
                "Correct Answer": "3e8 m/s",
                "Incorrect Answer 1": "3e6 m/s",
                "Incorrect Answer 2": "3e10 m/s",
                "Incorrect Answer 3": "3e4 m/s",
            }
        ],
        FakeContract("mcq", "\nAnswer: \\boxed{ANSWER}"),
        token_count=lambda text: len(text.split()),
        options=PreparationOptions(**_OPTS),
    )
    row = artifact.rows[0]
    assert row["data_source"] == "Idavidrein/gpqa"
    assert row["reward_model"]["ground_truth"] in ("A", "B", "C", "D")
    assert "3e8 m/s" in row["prompt"][0]["content"]


def test_openscience_preparation_extracts_letter():
    artifact = prepare_artifact(
        openscience_source(),
        [
            {
                "input": "What is photosynthesis?\nA: Process of light absorption\nB: Process of DNA replication",
                "output": "Photosynthesis is about light.\n\\boxed{A}",
            }
        ],
        FakeContract("mcq", "\nAnswer: \\boxed{ANSWER}"),
        token_count=lambda text: len(text.split()),
        options=PreparationOptions(**_OPTS),
    )
    assert artifact.rows[0]["reward_model"]["ground_truth"] == "A"


def test_kto_mix_preparation_keeps_preferred():
    artifact = prepare_artifact(
        kto_mix_source(),
        [
            {
                "prompt": [{"role": "user", "content": "Hello"}],
                "completion": [{"role": "assistant", "content": "Hi there!"}],
                "label": "True",
            }
        ],
        FakeContract("preference"),
        token_count=lambda text: len(text.split()),
        options=PreparationOptions(**_OPTS),
    )
    assert artifact.rows[0]["data_source"] == "trl-lib/kto-mix-14k"
    assert artifact.rows[0]["reward_model"]["ground_truth"] == "Hi there!"


def test_kto_mix_skips_dispreferred():
    with pytest.raises(ValueError, match="label=False"):
        prepare_artifact(
            kto_mix_source(),
            [
                {
                    "prompt": [{"role": "user", "content": "Hello"}],
                    "completion": [{"role": "assistant", "content": "Bad reply"}],
                    "label": "False",
                }
            ],
            FakeContract("preference"),
            token_count=lambda text: len(text.split()),
            options=PreparationOptions(**_OPTS),
        )


def test_hh_rlhf_preparation_splits_chosen_rejected():
    artifact = prepare_artifact(
        hh_rlhf_source(),
        [
            {
                "chosen": "\n\nHuman: Hi\n\nAssistant: Hello!",
                "rejected": "\n\nHuman: Hi\n\nAssistant: Go away",
            }
        ],
        FakeContract("preference"),
        token_count=lambda text: len(text.split()),
        options=PreparationOptions(**_OPTS),
    )
    row = artifact.rows[0]
    assert row["data_source"] == "Anthropic/hh-rlhf"
    assert row["reward_model"]["ground_truth"] == "Hello!"
    assert row["extra_info"]["rejected"] == "Go away"


def test_hh_rlhf_rejects_identical_pairs():
    with pytest.raises(ValueError, match="identical"):
        prepare_artifact(
            hh_rlhf_source(),
            [
                {
                    "chosen": "\n\nHuman: Hi\n\nAssistant: Same",
                    "rejected": "\n\nHuman: Hi\n\nAssistant: Same",
                }
            ],
            FakeContract("preference"),
            token_count=lambda text: len(text.split()),
            options=PreparationOptions(**_OPTS),
        )
