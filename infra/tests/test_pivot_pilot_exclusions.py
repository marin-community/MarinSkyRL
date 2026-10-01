"""Unusable demonstrations are excluded consistently from every frozen training arm."""

import json

import pyarrow as pa
import pyarrow.parquet as pq

from infra.rl_data.pivot_pilot import freeze


def test_unusable_selected_demonstration_is_excluded_from_all_arms(tmp_path):
    split = tmp_path / 'split'
    split.mkdir()
    rows = [{'extra_info': {'source_id': str(i), 'nemotron_ultra': {
        'record_json': json.dumps({'metadata': {'instance_id': f'task-{i}'}})}}} for i in range(3)]
    pq.write_table(pa.Table.from_pylist(rows), split / 'candidates.parquet')
    (split / 'manifest.json').write_text('{}')
    profile = tmp_path / 'rescored.jsonl'
    records = [{'source_id': str(i), 'record_id': f'{i}:{rep}', 'repetition_id': rep,
                'profiling_attempt': 0, 'status': 'verified',
                'scores': dict.fromkeys(['tool_name', 'nemo', 'exact'], int(i < 2 and rep == 0))}
               for i in range(3) for rep in range(8)]
    profile.write_text(''.join(json.dumps(record) + '\n' for record in records))
    output = tmp_path / 'frozen'
    manifest = freeze(split, [profile], output, 'grug', {'0': 'demonstration exceeds context window'})
    assert manifest['counts'] == {'all_train': 2, 'train': 1, 'random_train': 1}
    assert manifest['explicit_exclusions'] == {'0': 'demonstration exceeds context window'}
    for name in manifest['counts']:
        ids = {row['extra_info']['source_id'] for row in pq.read_table(output / f'{name}.parquet').to_pylist()}
        assert '0' not in ids
    assert {row['extra_info']['source_id'] for row in pq.read_table(output / 'train.parquet').to_pylist()} == {'1'}
