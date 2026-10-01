"""Shared initialization survives JSON persistence without regenerating validation."""

import json

from omegaconf import OmegaConf
import pytest

from skyrl_train.evaluate import evaluate


@pytest.mark.asyncio
@pytest.mark.parametrize('revision', ['pinned-model', 'different-model'])
async def test_initialization_cache_round_trip(tmp_path, revision):
    cache = tmp_path / 'baseline.json'
    sampling = {'temperature': 0.0, 'top_p': 1.0}
    identity = {
        'model': 'test-model', 'revision': revision, 'split': 'validation-hash', 'quick': 'quick-hash',
        'sampling': sampling,
        'verifiers': ('tool_name:expected_tool_name_match:v1', 'nemo:word_count_threshold=0:v1',
                      'exact:strict_recursive_arguments:v1'),
    }
    metrics = {'eval/full/tool_name/accuracy': 0.5}
    cache.write_text(json.dumps({'identity': identity, 'metrics': metrics}))
    cfg = OmegaConf.create({
        'trainer': {'policy': {'model': {'path': 'test-model'}}, 'pivot_pilot': {
            'baseline_cache': str(cache), 'split_hash': 'validation-hash', 'quick_hash': 'quick-hash'}},
        'generator': {'trajectory_retention': {'model_source_identity': 'pinned-model'},
                      'eval_sampling_params': sampling},
    })
    if revision == 'different-model':
        with pytest.raises(ValueError, match='Initialization cache identity mismatch'):
            await evaluate(None, None, cfg, 0, None)
    else:
        assert await evaluate(None, None, cfg, 0, None) == metrics
