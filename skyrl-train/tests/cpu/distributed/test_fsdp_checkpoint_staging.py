from contextlib import contextmanager, nullcontext
from io import BytesIO
import json
from pathlib import Path
import struct

import pytest
import torch

from skyrl_train.distributed import fsdp_strategy as fsdp_module
from skyrl_train.distributed.fsdp_strategy import FSDPStrategy


@pytest.mark.parametrize("fail_optimizer_save", [False, True])
def test_s3_checkpoint_streams_complete_rank_files_without_local_staging(monkeypatch, fail_optimizer_save):
    strategy = object.__new__(FSDPStrategy)
    strategy.world_size = 2
    strategy.fsdp_strategy = "fsdp2"
    strategy.is_lora = False
    monkeypatch.setattr(strategy, "get_rank", lambda: 1)
    monkeypatch.setattr(strategy, "is_rank_0", lambda: False)
    monkeypatch.setattr(strategy, "get_rng_state", lambda: {})
    monkeypatch.setattr(strategy, "log", lambda *args: None)
    monkeypatch.setattr(fsdp_module.dist, "barrier", lambda: None)
    monkeypatch.setattr(fsdp_module, "get_fsdp_state_ctx", lambda *args, **kwargs: nullcontext())

    def reject_local_staging(*args):
        raise AssertionError("S3 rank files must not stage locally")

    monkeypatch.setattr(fsdp_module.io, "local_output_dir", reject_local_staging)

    objects = {}

    @contextmanager
    def write_object(path):
        stream = BytesIO()
        yield stream
        objects[path] = stream.getvalue()

    monkeypatch.setattr(fsdp_module, "open_s3_checkpoint_write_stream", write_object)
    if fail_optimizer_save:
        save = torch.save

        def fail_optimizer(obj, file):
            if isinstance(obj, dict) and "param_groups" in obj:
                raise OSError("optimizer serialization failed")
            return save(obj, file)

        monkeypatch.setattr(fsdp_module.torch, "save", fail_optimizer)

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint_dir = "s3://bucket/checkpoints/global_step_1/policy"
    if fail_optimizer_save:
        with pytest.raises(OSError, match="optimizer serialization failed"):
            strategy.save_checkpoint(model, checkpoint_dir, node_local_rank=1, optimizer=optimizer)
        assert len(objects) == 1
        assert next(iter(objects)).endswith("model_world_size_2_rank_1.pt")
    else:
        strategy.save_checkpoint(model, checkpoint_dir, node_local_rank=1, optimizer=optimizer)
        assert len(objects) == 3
        assert "weight" in torch.load(BytesIO(objects[f"{checkpoint_dir}/model_world_size_2_rank_1.pt"]))
        assert "param_groups" in torch.load(BytesIO(objects[f"{checkpoint_dir}/optim_world_size_2_rank_1.pt"]))
        extra = torch.load(BytesIO(objects[f"{checkpoint_dir}/extra_state_world_size_2_rank_1.pt"]))
        assert extra["world_size"] == 2


def test_cloud_checkpoint_load_stages_only_its_rank_shards(monkeypatch, tmp_path):
    strategy = object.__new__(FSDPStrategy)
    strategy.world_size = 8
    strategy.fsdp_strategy = "fsdp"
    monkeypatch.setattr(strategy, "get_rank", lambda: 3)
    monkeypatch.setattr(strategy, "log", lambda *args: None)

    staged_paths = []

    @contextmanager
    def stage_requested_files(paths):
        staged_paths.extend(paths)
        local_paths = [tmp_path / Path(path).name for path in paths]
        torch.save(torch.nn.Linear(1, 1).state_dict(), local_paths[0])
        torch.save({}, local_paths[1])
        torch.save(
            {
                "client_state": {},
                "fsdp_strategy": "fsdp",
                "world_size": 8,
                "rank": 3,
            },
            local_paths[2],
        )
        yield [str(path) for path in local_paths]

    @contextmanager
    def reject_directory_staging(path):
        raise AssertionError(f"checkpoint load must not stage the whole directory: {path}")
        yield

    monkeypatch.setattr(fsdp_module.io, "exists", lambda path: True)
    monkeypatch.setattr(fsdp_module.io, "local_read_files", stage_requested_files)
    monkeypatch.setattr(fsdp_module.io, "local_read_dir", reject_directory_staging)
    monkeypatch.setattr(fsdp_module, "get_fsdp_state_ctx", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(fsdp_module.dist, "barrier", lambda: None)

    strategy.load_checkpoint(torch.nn.Linear(1, 1), "s3://bucket/checkpoints/step_28/policy")

    assert staged_paths == [
        "s3://bucket/checkpoints/step_28/policy/model_world_size_8_rank_3.pt",
        "s3://bucket/checkpoints/step_28/policy/optim_world_size_8_rank_3.pt",
        "s3://bucket/checkpoints/step_28/policy/extra_state_world_size_8_rank_3.pt",
    ]


def test_export_checkpoint_load_does_not_require_training_state(monkeypatch, tmp_path):
    strategy = object.__new__(FSDPStrategy)
    strategy.world_size = 8
    strategy.fsdp_strategy = "fsdp"
    monkeypatch.setattr(strategy, "get_rank", lambda: 3)
    monkeypatch.setattr(strategy, "log", lambda *args: None)
    staged_paths = []

    @contextmanager
    def stage_requested_files(paths):
        staged_paths.extend(paths)
        model_path = tmp_path / Path(paths[0]).name
        torch.save(torch.nn.Linear(1, 1).state_dict(), model_path)
        yield [str(model_path)]

    monkeypatch.setattr(fsdp_module.io, "exists", lambda path: "extra_state" not in path and "optim" not in path)
    monkeypatch.setattr(fsdp_module.io, "local_read_files", stage_requested_files)
    monkeypatch.setattr(fsdp_module, "get_fsdp_state_ctx", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(fsdp_module.dist, "barrier", lambda: None)

    strategy.load_checkpoint(
        torch.nn.Linear(1, 1),
        "s3://bucket/checkpoints/step_28/policy",
        load_training_state=False,
    )

    assert staged_paths == ["s3://bucket/checkpoints/step_28/policy/model_world_size_8_rank_3.pt"]


def test_hf_export_serialization_has_no_trailing_barrier(monkeypatch, tmp_path):
    barrier_entered = False

    class Config:
        def save_pretrained(self, output_dir):
            Path(output_dir, "config.json").write_text("{}")
            Path(output_dir, "tokenizer.json").write_text("{}")

    class Model(torch.nn.Module):
        config = Config()

        def save_pretrained(self, output_dir, **kwargs):
            header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
            Path(output_dir, "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)

    strategy = object.__new__(FSDPStrategy)
    strategy.fsdp_strategy = "fsdp2"
    monkeypatch.setattr(strategy, "is_rank_0", lambda: True)
    monkeypatch.setattr(strategy, "get_rank", lambda: 0)
    monkeypatch.setattr(strategy, "log", lambda *args: None)
    monkeypatch.setattr(strategy, "_unwrap_model", lambda model: model)
    monkeypatch.setattr(strategy, "_fix_fsdp_config", lambda config: config)
    monkeypatch.setattr(fsdp_module, "fsdp_version", lambda model: 2)
    monkeypatch.setattr(fsdp_module, "fsdp2_get_full_state_dict", lambda *args, **kwargs: {})

    def record_barrier():
        nonlocal barrier_entered
        barrier_entered = True

    monkeypatch.setattr(fsdp_module.dist, "barrier", record_barrier)

    strategy.save_hf_model(Model(), str(tmp_path / "export"))

    assert not barrier_entered
