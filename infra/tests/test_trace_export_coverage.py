from infra.rl_cleanup.make_and_upload_trace_dataset import prepare_trace_stage, trace_export_manifest


def test_trace_export_manifest_records_partial_result_coverage():
    manifest = trace_export_manifest(result_count=100, row_count=97, shard_names=["train-00000.parquet"])

    assert manifest == {
        "result_count": 100,
        "row_count": 97,
        "result_coverage": 0.97,
        "shards": ["train-00000.parquet"],
    }


def test_trace_export_manifest_treats_empty_source_as_complete():
    manifest = trace_export_manifest(result_count=0, row_count=0, shard_names=[])

    assert manifest["result_coverage"] == 1.0


def test_prepare_trace_stage_removes_only_previous_trace_shards(tmp_path):
    stage_dir = tmp_path / "stage"
    data_dir = stage_dir / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "train-00000.parquet").write_bytes(b"old")
    (data_dir / "notes.parquet").write_bytes(b"keep")

    prepare_trace_stage(stage_dir)

    assert not (data_dir / "train-00000.parquet").exists()
    assert (data_dir / "notes.parquet").read_bytes() == b"keep"
