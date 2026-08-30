from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch

from skyrl_train.distributed import fsdp_strategy as fsdp_module
from skyrl_train.distributed.fsdp_strategy import FSDPStrategy


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

    class Model(torch.nn.Module):
        config = Config()

        def save_pretrained(self, output_dir, **kwargs):
            Path(output_dir, "model.safetensors").write_bytes(b"weights")

    strategy = object.__new__(FSDPStrategy)
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
