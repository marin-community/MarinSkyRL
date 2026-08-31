import json

import pytest

from infra.rl_data.preparation import PreparationOptions, prepare_artifact, write_artifact, write_bundle
from infra.rl_data.mixtures import MixtureSlice, MixtureSpec, load_mixture_spec, prepare_mixture
from infra.rl_data.sources import (
    Source,
    aime_1983_2024_source,
    aime24_source,
    apps_source,
    asdiv_source,
    dapo_math_source,
    deepscaler_source,
    gpqa_source,
    gsm8k_source,
    hardmath_source,
    hendrycks_math_source,
    hh_rlhf_source,
    kto_mix_source,
    eurus2_code_source,
    generate_reasoning_gym_rows,
    load_source_rows,
    math500_source,
    nemotron_if_source,
    numina_math_source,
    openscience_source,
    rlvr_math_source,
    rlvr_ifeval_source,
    source_by_name,
    svamp_source,
    verifiable_code_source,
)
from skyrl_gym import get_data_contract
from skyrl_gym.envs.ifeval import utils as ifeval_utils


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


def _mostly_malformed_rlvr_math_examples():
    return [
        {"messages": [{"role": "user", "content": "Question: bad one"}], "ground_truth": ""},
        {"messages": [{"role": "user", "content": "Question: 2 + 2"}], "ground_truth": "4"},
        {"messages": [{"role": "user", "content": "Question: bad two"}], "ground_truth": ""},
    ]


def _prepare_mostly_malformed_rlvr_math(minimum_yield_fraction=None):
    return prepare_artifact(
        rlvr_math_source(),
        _mostly_malformed_rlvr_math_examples(),
        FakeContract("aime", " Answer: \\boxed{ANSWER}"),
        token_count=lambda text: len(text.split()),
        options=PreparationOptions(
            source_revision="fixture",
            max_prompt_tokens=20,
            minimum_unique_rows=1,
            minimum_yield_fraction=minimum_yield_fraction,
        ),
    )


def test_preparation_skips_a_majority_of_malformed_rows_with_provenance() -> None:
    with pytest.warns(UserWarning, match=r"skipped 2 of 3 rows"):
        artifact = _prepare_mostly_malformed_rlvr_math()

    assert len(artifact.rows) == 1
    assert artifact.provenance["counts"]["malformed_rows_skipped"] == 2
    assert artifact.provenance["conversion_failures"] == [
        {"index": 0, "error": "ValueError: empty ground truth"},
        {"index": 2, "error": "ValueError: empty ground truth"},
    ]


def test_preparation_can_require_a_minimum_conversion_yield() -> None:
    with pytest.warns(UserWarning, match=r"skipped 2 of 3 rows"):
        with pytest.raises(ValueError, match=r"conversion yield 1/3 .* below minimum_yield_fraction=0.5"):
            _prepare_mostly_malformed_rlvr_math(minimum_yield_fraction=0.5)


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


def test_aime24_preparation_uses_the_aime_verifier_contract():
    artifact = prepare_artifact(
        aime24_source(),
        [{"problem": "What is 100 + 24?", "answer": "124"}],
        FakeContract("aime", " Answer: \\boxed{ANSWER}"),
        token_count=lambda text: len(text.split()),
        options=PreparationOptions(
            source_revision="fixture",
            max_prompt_tokens=20,
            minimum_unique_rows=1,
            artifact_split="validation",
        ),
    )

    assert artifact.rows == [
        {
            "data_source": "HuggingFaceH4/aime_2024",
            "prompt": [{"role": "user", "content": "What is 100 + 24? Answer: \\boxed{ANSWER}"}],
            "env_class": "aime",
            "reward_model": {"ground_truth": "124"},
            "extra_info": {"split": "validation", "index": 0},
        }
    ]
    assert artifact.provenance["verification"] == "two_sided"


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


@pytest.mark.parametrize(
    ("source", "example", "expected_answer", "expected_metadata"),
    [
        (
            hendrycks_math_source(),
            {"problem": "Compute 2 + 2.", "solution": "Therefore \\boxed{4}.", "level": "Level 1", "subject": "algebra"},
            "4",
            {"level": "Level 1", "subject": "algebra"},
        ),
        (
            aime_1983_2024_source(),
            {"Question": "Compute 100 + 24.", "Answer": 124},
            "124",
            {},
        ),
        (
            asdiv_source(),
            {"Body": "Sam has four apples.", "Question": "How many?", "Answer": "4 (apples)", "Grade": "2"},
            "4",
            {"grade": "2"},
        ),
        (
            svamp_source(),
            {"Body": "Sam has four apples.", "Question": "How many?", "Answer": 4.0},
            "4",
            {},
        ),
        (
            numina_math_source(),
            {"problem": "Compute one half.", "solution": "Therefore \\boxed{\\frac{1}{2}}.", "source": "cn_k12"},
            "\\frac{1}{2}",
            {"source": "cn_k12"},
        ),
        (
            hardmath_source(),
            {"question": "Approximate epsilon.", "ground_truths": "The result is \\boxed{\\epsilon \\approx 4.16}."},
            "4.16",
            {},
        ),
    ],
)
def test_curriculum_math_sources_emit_verifier_ready_rows(source, example, expected_answer, expected_metadata):
    artifact = prepare_artifact(
        source,
        [example],
        get_data_contract("aime"),
        token_count=lambda text: len(text.split()),
        options=PreparationOptions(**_OPTS),
    )

    row = artifact.rows[0]
    assert row["data_source"] == source.dataset_id
    assert row["env_class"] == "aime"
    assert row["reward_model"]["ground_truth"] == expected_answer
    assert {key: row["extra_info"][key] for key in expected_metadata} == expected_metadata
    assert artifact.provenance["verification"] == "two_sided"


def test_hendrycks_math_loader_combines_selected_subjects(monkeypatch):
    def load_dataset(dataset_id, subject, *, split, revision, streaming):
        return [{"problem": f"Problem from {subject}", "solution": "\\boxed{1}", "level": "Level 1"}]

    monkeypatch.setattr("datasets.load_dataset", load_dataset)

    rows = list(
        load_source_rows(
            hendrycks_math_source(),
            "revision-1",
            {"subjects": ["algebra", "geometry"], "skip": 1},
        )
    )

    assert rows == [
        {
            "problem": "Problem from geometry",
            "solution": "\\boxed{1}",
            "level": "Level 1",
            "subject": "geometry",
        }
    ]


def test_asdiv_loader_reads_pinned_xml(monkeypatch):
    class Response:
        content = b"""<ASDiv><ProblemSet><Problem ID='1' Grade='3'><Body>Sam has four apples.</Body><Question>How many?</Question><Answer>4 (apples)</Answer></Problem></ProblemSet></ASDiv>"""

        def raise_for_status(self):
            return None

    requested = []

    def get(url, *, timeout):
        requested.append((url, timeout))
        return Response()

    monkeypatch.setattr("infra.rl_data.sources.requests.get", get)

    rows = list(load_source_rows(asdiv_source(), "commit-123", {}))

    assert rows == [
        {"Body": "Sam has four apples.", "Question": "How many?", "Answer": "4 (apples)", "Grade": "3"}
    ]
    assert requested == [
        ("https://raw.githubusercontent.com/chaochun/nlu-asdiv-dataset/commit-123/dataset/ASDiv.xml", 60)
    ]


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


def _fake_source(name: str, env_id: str) -> Source:
    def prepare(example, index, contract):
        ground_truth = contract.normalize_ground_truth(example["answer"])
        return {
            "data_source": f"fixture/{name}",
            "prompt": [{"role": "user", "content": example["prompt"]}],
            "env_class": env_id,
            "reward_model": {"ground_truth": ground_truth},
            "extra_info": {"split": "train", "index": index},
        }

    return Source(name, f"fixture/{name}", env_id, "train", False, "schema_only", prepare)


def test_mixture_yaml_parses_source_slice_controls(tmp_path):
    path = tmp_path / "mixture.yaml"
    path.write_text(
        """\
train:
  - source: eurus2_code
    revision: abc123
    split: validation
    cap: 3
    minimum_unique_rows: 2
    parameters:
      skip: 7
validation:
  - source: reasoning_gym
    revision: 0.1.25
    parameters:
      tasks: [leg_counting]
      rows_per_task: 100
"""
    )

    spec = load_mixture_spec(path)

    assert spec.train == (
        MixtureSlice(
            source="eurus2_code",
            revision="abc123",
            cap=3,
            minimum_unique_rows=2,
            parameters={"skip": 7},
            split="validation",
        ),
    )
    assert spec.validation[0].parameters == {"tasks": ["leg_counting"], "rows_per_task": 100}


def test_mixture_preparation_preserves_source_verifiers_caps_and_validation_slices():
    sources = {"math": _fake_source("math", "aime"), "if": _fake_source("if", "ifeval")}
    rows = {
        "math": [{"prompt": f"a{index}", "answer": index + 1} for index in range(4)],
        "if": [{"prompt": f"b{index}", "answer": f"constraint-{index}"} for index in range(3)],
    }
    contracts = {"aime": FakeContract("aime"), "ifeval": FakeContract("ifeval")}
    spec = MixtureSpec(
        train=(MixtureSlice("math", "math-rev", cap=3), MixtureSlice("if", "if-rev", cap=2)),
        validation=(MixtureSlice("math", "math-val", cap=1), MixtureSlice("if", "if-val", cap=1)),
    )

    def build():
        return prepare_mixture(
            spec,
            token_count=lambda text: len(text),
            max_prompt_tokens=100,
            seed=17,
            source_lookup=sources.__getitem__,
            contract_lookup=contracts.__getitem__,
            row_loader=lambda source, revision, parameters: rows[source.name],
        )

    train, validation = build()

    assert [row["prompt"][0]["content"] for row in train.rows] == ["a1", "a3", "a2", "b1", "b2"]
    assert [row["env_class"] for row in train.rows] == ["aime", "aime", "aime", "ifeval", "ifeval"]
    assert len(validation.rows) == 2
    assert {row["env_class"] for row in validation.rows} == {"aime", "ifeval"}
    assert build() == (train, validation)
    assert [source["source"]["revision"] for source in train.provenance["sources"]] == ["math-rev", "if-rev"]
    assert [source["counts"]["emitted_rows"] for source in train.provenance["sources"]] == [3, 2]
    assert [source["share"] for source in train.provenance["sources"]] == pytest.approx([0.6, 0.4])


@pytest.mark.parametrize(
    ("source_factory", "source_name", "label"),
    [(aime24_source, "aime24", "AIME24"), (math500_source, "math500", "MATH-500")],
)
def test_mixture_rejects_test_only_math_training_without_explicit_permission(source_factory, source_name, label):
    spec = MixtureSpec(
        train=(MixtureSlice(source_name, "fixture", cap=1),),
        validation=(MixtureSlice(source_name, "fixture", cap=1),),
    )

    with pytest.raises(ValueError, match=f"{label} is test-only"):
        prepare_mixture(
            spec,
            token_count=lambda text: len(text),
            max_prompt_tokens=100,
            seed=1,
            source_lookup=lambda name: source_factory(),
            contract_lookup=lambda env_id: FakeContract(env_id, " answer"),
            row_loader=lambda source, revision, parameters: [{"problem": "2+2", "answer": "4"}],
        )


def test_eurus_code_adapter_normalizes_apps_tests():
    artifact = prepare_artifact(
        eurus2_code_source(),
        [
            {
                "ability": "code",
                "prompt": [{"role": "user", "content": "Read two integers and print their sum."}],
                "reward_model": {
                    "ground_truth": json.dumps({"inputs": ["1 2\n", "4 5\n"], "outputs": ["3\n", "9\n"]})
                },
            }
        ],
        get_data_contract("lcb"),
        token_count=lambda text: len(text.split()),
        options=PreparationOptions(**_OPTS),
    )

    tests = json.loads(artifact.rows[0]["reward_model"]["ground_truth"])
    assert tests == [
        {"input": "1 2\n", "output": "3\n", "testtype": "stdin"},
        {"input": "4 5\n", "output": "9\n", "testtype": "stdin"},
    ]
    assert artifact.rows[0]["env_class"] == "lcb"


def test_reasoning_gym_generation_is_deterministic_verifiable_and_disjoint():
    train_rows = list(
        generate_reasoning_gym_rows(
            tasks=("leg_counting", "knights_knaves"), rows_per_task=3, seed=41, start_index=0
        )
    )
    rebuilt_rows = list(
        generate_reasoning_gym_rows(
            tasks=("leg_counting", "knights_knaves"), rows_per_task=3, seed=41, start_index=0
        )
    )
    holdout_rows = list(
        generate_reasoning_gym_rows(
            tasks=("leg_counting", "knights_knaves"), rows_per_task=100, seed=41, start_index=3
        )
    )

    assert rebuilt_rows == train_rows
    assert {row["question"] for row in train_rows}.isdisjoint(row["question"] for row in holdout_rows)
    contract = get_data_contract("reasoning_gym")
    artifact = prepare_artifact(
        source=source_by_name("reasoning_gym"),
        examples=train_rows,
        contract=contract,
        token_count=lambda text: len(text.split()),
        options=PreparationOptions(**_OPTS),
    )
    for row in artifact.rows:
        ground_truth = row["reward_model"]["ground_truth"]
        assert contract.is_correct(json.loads(ground_truth)["entry"]["answer"], ground_truth)
        assert not contract.is_correct("definitely wrong", ground_truth)


def test_nemotron_if_adapter_builds_fractional_ifeval_constraints():
    artifact = prepare_artifact(
        nemotron_if_source(),
        [
            {
                "input": [{"role": "user", "content": "Write three sentences and add a P.S."}],
                "args": {
                    "instruction_id_list": [
                        "length_constraints:number_sentences",
                        "detectable_content:postscript",
                    ],
                    "instruction_kwargs": [
                        {"num_sentences": 3, "relation": "at least"},
                        {"postscript_marker": "P.S."},
                    ],
                },
            }
        ],
        get_data_contract("ifeval"),
        token_count=lambda text: len(text.split()),
        options=PreparationOptions(**_OPTS),
    )

    constraints = json.loads(artifact.rows[0]["reward_model"]["ground_truth"])
    assert constraints == [
        {"N": 3, "func_name": "verify_sentence_constraint", "quantifier": "at least"},
        {"func_name": "verify_postscript", "postscript_marker": "P.S."},
    ]
    assert [json.loads(ifeval_utils.normalize_ground_truth(constraint)) for constraint in constraints] == constraints
