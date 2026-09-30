from types import SimpleNamespace

import numpy as np
import pytest
from finestore import mismatch_probe as mismatch
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from skyrl_train.config.utils import get_default_config
from skyrl_train.group_admission import GroupAdvantageInvariant
from skyrl_train.inference_engines.vllm_teacher_oracle import tokenizer_vocabulary_fingerprint
from skyrl_train.mismatch_probe.archive import MismatchArchive, read_frozen_probe
from skyrl_train.mismatch_probe.collect import ProbeCollector
from skyrl_train.trainer import RayPPOTrainer


def test_reuse_reads_completed_frozen_tokens_and_generation_scores(tmp_path):
    uri = str(tmp_path / "probe")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({str(i): i for i in range(32)}, unk_token="0")),
        pad_token="0",
        unk_token="0",
    )
    cfg = get_default_config()
    cfg.trainer.mismatch_probe.archive_uri = uri
    cfg.trainer.mismatch_probe.prompts.count = 3
    cfg.trainer.mismatch_probe.prompts.samples_per_prompt = 1
    cfg.trainer.policy.megatron_config.moe_router_replay = True
    cfg.trainer.algorithm.advantage_estimator = "uniform"
    cfg.trainer.algorithm.use_tis = False
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = cfg
    trainer.group_advantage_invariant = GroupAdvantageInvariant.no_group_advantage(physical_group_size=1)
    trainer.tokenizer = tokenizer
    trainer.inference_engine_client = SimpleNamespace(tokenizer=tokenizer)
    trainer.policy_model = SimpleNamespace(actor_infos=[SimpleNamespace(rank=SimpleNamespace(dp_size=2))])
    trainer.critic_model = trainer.ref_model = None
    trainer.all_metrics = {}
    trainer._num_experts_cache = 8
    routes = np.asarray([[[1, 2]], [[0, 0]], [[1, 2]], [[0, 0]]], dtype=np.int32)
    trajectory = {
        "prompt_token_ids": [[3, 4], [5], [6, 7, 8]],
        "response_ids": [[7, 9, 11, 13], [9, 10, 11, 12], [15, 16, 17, 18]],
        "rewards": [1.0, 0.0, 0.5],
        "loss_masks": [[1] * 4 for _ in range(3)],
        "rollout_logprobs": [np.asarray([-0.125, -2.75, -1.5, -0.5], dtype=np.float32) for _ in range(3)],
        "rollout_routed_experts": [routes.copy() for _ in range(3)],
    }
    prompt_ids = [f"prompt-{i}" for i in range(3)]
    sample_ids = [f"sample-{i}" for i in range(3)]
    collector = ProbeCollector(cfg)
    collector._collate_and_freeze(trainer, trajectory, prompt_ids, sample_ids, [91, 92, 93])
    probes = collector.probes
    assert collector.training_input.batch_size == 4
    assert not collector.training_input["loss_mask"][-1].any()
    for row in probes:
        assert row.route_valid_mask == [[True], [False], [True], [False]]
    generation = [
        mismatch.ScoreRow(
            probe_hash=collector.probe_hash,
            sample_id=row.sample_id,
            scorer="vllm.generate",
            update=0,
            weights_hash="weights-hash",
            logprobs=values.tolist(),
        )
        for row, values in zip(probes, trajectory["rollout_logprobs"], strict=True)
    ]
    manifest = mismatch.ManifestRow(
        archive=uri,
        status=mismatch.ArchiveStatus.BUILDING,
        probe_hash=collector.probe_hash,
        starting_weights_hash="weights-hash",
        tokenizer_fingerprint=tokenizer_vocabulary_fingerprint(tokenizer),
        starting_global_step=1,
        scored_updates=[0],
        scored_global_steps=[1],
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
        archive.write(probes=probes, scores=generation, manifest=manifest)
        with pytest.raises(ValueError, match="not a single complete"):
            read_frozen_probe(uri)
        archive.write(manifest=manifest.model_copy(update={"status": mismatch.ArchiveStatus.COMPLETE}))
    finally:
        archive.close()
    source = read_frozen_probe(uri)
    assert source.manifest.starting_weights_hash == "weights-hash"
    assert source.probes == probes
    assert list(source.generations.values()) == generation
    cfg.trainer.mismatch_probe.reuse_probe = uri
    reused = ProbeCollector(cfg)
    samples = reused._from_source()
    reused._collate_and_freeze(trainer, samples.trajectory, samples.prompt_ids, samples.sample_ids, samples.seeds)
    assert reused.probes == probes
    for original, restored in zip(trajectory["rollout_logprobs"], reused.generation_scores, strict=True):
        np.testing.assert_array_equal(restored, original)
