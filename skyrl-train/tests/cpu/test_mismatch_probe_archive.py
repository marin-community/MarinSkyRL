import pytest
from finestore import mismatch

from skyrl_train.config.mismatch_probe import GENERATION_SCORING
from skyrl_train.mismatch_probe.archive import MismatchArchive, read_frozen_probe


def test_reuse_reads_completed_frozen_tokens_and_generation_scores(tmp_path):
    schema = mismatch
    uri = str(tmp_path / "probe")
    probe = schema.ProbeRow(
        probe_hash="probe-hash",
        sample_id="sample-0",
        prompt_id="prompt-0",
        prompt_token_ids=[3, 4],
        trainer_prompt_ids=[3, 4],
        vllm_output_ids=[7, 9],
        trainer_input_ids=[7, 9],
        response_mask=[True, True],
        loss_mask=[True, True],
        request_seed=91,
        batch_position=0,
    )
    generation = schema.ScoreRow(
        probe_hash="probe-hash",
        sample_id="sample-0",
        scoring=GENERATION_SCORING,
        update=0,
        weights_hash="weights-hash",
        logprobs=[-0.125, -2.75],
    )
    manifest = schema.ManifestRow(
        archive=uri,
        status="building",
        probe_hash="probe-hash",
        starting_weights_hash="weights-hash",
        architecture="tiny-grug",
        vllm_enforce_eager=False,
        optimizer_steps_per_update=0,
        seed=17,
        bootstrap_seed=18,
        created_at_utc="2026-09-26T00:00:00Z",
        config_json="{}",
        software_json="{}",
        hardware_json="{}",
        batch_layout_json="{}",
        timing_json="{}",
        step_metrics_json="{}",
    )
    archive = MismatchArchive(uri, writer_id="test")
    try:
        archive.write(probes=[probe], scores=[generation], manifest=manifest)
    finally:
        archive.close()
    with pytest.raises(ValueError, match="not a single complete"):
        read_frozen_probe(uri)

    archive = MismatchArchive(uri, writer_id="test-complete")
    try:
        archive.write(manifest=manifest.model_copy(update={"status": "complete"}))
    finally:
        archive.close()
    source = read_frozen_probe(uri)
    assert source.manifest.starting_weights_hash == "weights-hash"
    assert source.probes == [probe]
    assert source.generations["sample-0"] == generation
