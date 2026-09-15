# SPDX-FileCopyrightText: 2026 NovaSkyAI
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
from omegaconf import OmegaConf

from skyrl_train.entrypoints.fully_async_levanter_snowball import validate_fully_async_levanter_config

from tests.cpu.util import example_dummy_config


def _valid_config():
    cfg = copy.deepcopy(example_dummy_config())
    updates = {
        "environment.env_class": "gsm8k",
        "trainer.train_batch_size": 4,
        "trainer.policy_mini_batch_size": 4,
        "trainer.fully_async.max_staleness_steps": 4,
        "trainer.fully_async.num_parallel_generation_workers": 20,
        "trainer.fully_async.max_buffered_groups": 4,
        "generator.batched": False,
        "generator.enable_http_endpoint": False,
        "generator.use_conversation_multi_turn": False,
    }
    for path, value in updates.items():
        OmegaConf.update(cfg, path, value, force_add=True)
    return cfg


def test_direct_async_levanter_config_passes_before_allocation():
    validate_fully_async_levanter_config(_valid_config())


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        ("environment.env_class", "aime", "env_class=gsm8k"),
        ("generator.batched", True, "batched=false"),
        ("generator.enable_http_endpoint", True, "direct token evidence"),
        ("generator.use_conversation_multi_turn", True, "use_conversation_multi_turn=false"),
        ("trainer.fully_async.max_staleness_steps", -1, "nonnegative max staleness"),
        ("trainer.fully_async.num_parallel_generation_workers", 3, "generation workers"),
        ("trainer.fully_async.max_buffered_groups", None, "explicit positive max_buffered_groups"),
        ("trainer.fully_async.max_buffered_groups", 21, "no greater than the worker count"),
        ("trainer.train_batch_size", 8, "train_batch_size equal"),
    ],
)
def test_unsupported_async_schedule_fails_before_allocation(path, value, match):
    cfg = _valid_config()
    OmegaConf.update(cfg, path, value)

    with pytest.raises(ValueError, match=match):
        validate_fully_async_levanter_config(cfg)
