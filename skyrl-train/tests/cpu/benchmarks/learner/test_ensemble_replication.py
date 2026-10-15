import collections

import numpy as np

from skyrl_train.benchmarks.learner.batch_artifact import list_ensemble, read_ensemble_batch, write_routes
from skyrl_train.benchmarks.learner.ensemble_replication import main as replicate
from skyrl_train.benchmarks.learner.route_pass import route_requests
from skyrl_train.dataset.routed_expert_batch import RoutedExpertRows

from .test_matched_bank import PROMPTS, RESPONSES, _write_ensemble

CHECKPOINT = "hero-test"
LAYERS = 3
TOP_K = 2
NUM_EXPERTS = 384


def _sample_routes(sample: int, length: int) -> np.ndarray:
    """Routes that identify their sample and response position."""
    token = np.arange(length).reshape(length, 1, 1)
    layer = np.arange(LAYERS).reshape(1, LAYERS, 1)
    choice = np.arange(TOP_K).reshape(1, 1, TOP_K)
    return (sample * 40 + token * 5 + layer * 2 + choice).astype(np.int16)


def _route_source(tmp_path, monkeypatch) -> str:
    root = _write_ensemble(tmp_path, monkeypatch)
    for path in list_ensemble(root, "v0"):
        batch, manifest = read_ensemble_batch(path, routes_checkpoint=None)
        samples = [int(row_id.split("/")[1]) for row_id in manifest.source["row_ids"]]
        rows = tuple(_sample_routes(sample, len(RESPONSES[sample])) for sample in samples)
        write_routes(path, CHECKPOINT, RoutedExpertRows(rows, manifest.response_length, NUM_EXPERTS))
    return root


def test_replicated_updates_carry_each_copy_routes(tmp_path, monkeypatch):
    source = _route_source(tmp_path, monkeypatch)
    out = str(tmp_path / "replicated")
    replicate(
        [
            "--source-root",
            source,
            "--source-version",
            "v0",
            "--out-root",
            out,
            "--ensemble-version",
            "v1",
            "--profile",
            "short",
            "--routes-checkpoint",
            CHECKPOINT,
            "--update-sizes",
            "4",
            "12",
            "--copies",
            "2",
            "--seed",
            "3",
            "--pad-token-id",
            "128001",
        ]
    )

    paths = list_ensemble(out, "v1")
    assert [path.rsplit("_", 1)[1] for path in paths] == ["40", "41", "120", "121"]
    uses: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    for path in paths:
        batch, manifest = read_ensemble_batch(path, routes_checkpoint=CHECKPOINT)
        size = manifest.source["update_sequences"]
        assert manifest.num_sequences == size and manifest.source["duplicated"]
        samples = [int(row_id.split("/")[1]) for row_id in manifest.source["row_ids"]]
        uses[size].update(samples)
        requests = route_requests(batch)
        for request, sample, rows in zip(requests, samples, batch.routed_expert_rows.rows, strict=True):
            assert request.token_ids == tuple(PROMPTS[sample] + RESPONSES[sample])
            np.testing.assert_array_equal(rows, _sample_routes(sample, len(RESPONSES[sample])))
        assert manifest.response_tokens == sum(len(RESPONSES[sample]) for sample in samples)
        assert manifest.total_tokens == sum(len(PROMPTS[sample]) + len(RESPONSES[sample]) for sample in samples)
    # Across both copies, 8 updates' worth of draws use each of the 8 samples once; 24 use each three times.
    assert set(uses[4].values()) == {1}
    assert set(uses[12].values()) == {3}
