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
    monkeypatch.setattr(strategy, "print", lambda *args: None)

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
