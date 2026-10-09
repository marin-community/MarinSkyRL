import json

import pyarrow.parquet as pq
import pytest

from examples.nupa.nupa_dataset import (
    answer_format_from_task_name,
    build_complement_records,
    build_panel_identities,
    panel_manifest_digest,
    split_prompt_answer,
    unique_texts_by_stratum,
    validate_records_against_verifier,
    write_parquet,
)
from rolloutengine.contracts import ModelTurn
from skyrl_gym.answer_tasks import grade_nupa

SYNTHETIC_SOURCE = {
    "add_Float_Float_Float": {
        "3": [
            "1.5 + 2.5 =4.0",
            "1.5 + 2.5 =4.0",
            "2.5 + 3.5 =6.0",
            "3.5 + 4.5 =8.0",
            "4.5 + 5.5 =10.0",
        ],
        "4": [
            "10.5 + 20.5 =31.0",
            "11.5 + 21.5 =33.0",
            "12.5 + 22.5 =35.0",
            "13.5 + 23.5 =37.0",
            "14.5 + 24.5 =39.0",
            "15.5 + 25.5 =41.0",
        ],
    },
    "mod_int": {
        "2": [
            "7 mod 3 =1",
            "8 mod 3 =2",
            "9 mod 3 =0",
            "10 mod 3 =1",
            "11 mod 3 =2",
        ],
    },
}


@pytest.fixture
def synthetic_source(tmp_path):
    source = tmp_path / "test.json"
    source.write_text(json.dumps(SYNTHETIC_SOURCE))
    return source


def test_unique_texts_by_stratum_deduplicates_repeated_texts(synthetic_source):
    unique = unique_texts_by_stratum(synthetic_source)

    assert sum(len(texts) for texts in unique.values()) == 15
    assert len(unique[("add_Float_Float_Float", 3)]) == 4


def test_panel_selection_is_deterministic_and_stays_inside_unique_records(synthetic_source):
    unique = unique_texts_by_stratum(synthetic_source)

    panel = build_panel_identities(unique, panel_size=7)
    again = build_panel_identities(unique_texts_by_stratum(synthetic_source), panel_size=7)

    assert len(panel) == 7
    assert len(set(panel)) == 7
    assert panel_manifest_digest(panel) == panel_manifest_digest(again)
    assert all(identity.sha256 in unique[(identity.task_name, identity.digit)] for identity in panel)


def test_complement_excludes_panel_and_keeps_identity_disjointness(synthetic_source):
    unique = unique_texts_by_stratum(synthetic_source)
    panel = build_panel_identities(unique, panel_size=7)
    held_out = {(identity.task_name, identity.digit, identity.sha256) for identity in panel}

    records = list(build_complement_records(unique, panel))

    assert len(records) == 15 - 7
    identities = {
        (row["extra_info"]["task_name"], row["extra_info"]["digit"], row["extra_info"]["source_sha256"])
        for row in records
    }
    assert not identities & held_out
    assert len(identities) == len(records)


@pytest.mark.parametrize(
    "task_name, answer_format",
    [("add_Float_Float_Float", "Float"), ("mod_int", "Integer"), ("to_Fraction", "Fraction")],
)
def test_answer_format_follows_the_task_name(task_name, answer_format):
    assert answer_format_from_task_name(task_name) == answer_format


def test_every_row_rewards_its_reference_answer_through_the_task_grader(synthetic_source):
    unique = unique_texts_by_stratum(synthetic_source)
    panel = build_panel_identities(unique, panel_size=7)

    for row in build_complement_records(unique, panel):
        assert row["prompt"][0]["content"].endswith(" =")
        answer = json.loads(row["reward_spec"]["ground_truth"])["answer"]
        for response, reward in [(answer, 1.0), (answer + "0", 0.0)]:
            turn = ModelTurn({"role": "assistant", "content": response}, (), (1,), None, "stop", text=response)
            assert grade_nupa(turn, {}, {"reward_spec": row["reward_spec"]}).reward == reward


def test_validate_records_against_verifier_counts_records_and_checks_each_task(synthetic_source):
    unique = unique_texts_by_stratum(synthetic_source)
    panel = build_panel_identities(unique, panel_size=7)

    assert validate_records_against_verifier(build_complement_records(unique, panel)) == (8, 2)


def test_write_parquet_round_trips_rows(synthetic_source, tmp_path):
    unique = unique_texts_by_stratum(synthetic_source)
    panel = build_panel_identities(unique, panel_size=7)
    output = tmp_path / "nupa" / "train.parquet"

    write_parquet(build_complement_records(unique, panel), output)

    table = pq.read_table(output)
    assert table.num_rows == 8
    assert table.schema.names == ["data_source", "prompt", "env_class", "reward_spec", "extra_info"]
    assert all(row == "nupa" for row in table.column("env_class").to_pylist())


def test_split_prompt_answer_rejects_text_without_delimiter():
    with pytest.raises(ValueError):
        split_prompt_answer("no delimiter here")
