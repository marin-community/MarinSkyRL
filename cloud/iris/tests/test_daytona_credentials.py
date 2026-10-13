"""Daytona credential selection for task backends."""

import pytest
from omegaconf import OmegaConf

import cloud.iris.iris_backend as launcher


def test_resolve_daytona_rl_api_key_prefers_explicit_rl_key(monkeypatch):
    monkeypatch.setenv("DAYTONA_RL_API_KEY", "rl-key")
    monkeypatch.setenv("DAYTONA_API_KEY", "generic-key")
    monkeypatch.setattr(launcher, "_daytona_rl_api_key_from_secret_manager", lambda: "secret-manager-key")

    assert launcher._resolve_daytona_rl_api_key() == "rl-key"


def test_resolve_daytona_rl_api_key_rejects_generic_key(monkeypatch):
    monkeypatch.delenv("DAYTONA_RL_API_KEY", raising=False)
    monkeypatch.setenv("DAYTONA_API_KEY", "generic-key")
    monkeypatch.setattr(launcher, "_daytona_rl_api_key_from_secret_manager", lambda: None)

    with pytest.raises(SystemExit, match="no Daytona RL key available"):
        launcher._resolve_daytona_rl_api_key()


@pytest.mark.parametrize(
    ("harbor", "expected"),
    [
        (None, False),
        ({"harbor": {"environment_type": "daytona"}}, True),
        ({"harbor": {"environment_type": "docker"}}, False),
    ],
)
def test_daytona_preflight_follows_task_backend(harbor, expected):
    config = OmegaConf.create(
        {
            "skyrl": {
                "terminal_bench_config": harbor,
            },
        }
    )

    assert launcher._rl_config_uses_daytona(config) is expected


@pytest.mark.parametrize("harbor_source", [False, True])
def test_daytona_preflight_ignores_unused_harbor_settings_in_nemotron_runs(harbor_source):
    config = OmegaConf.create(
        {
            "skyrl": {
                "terminal_bench_config": {"harbor": {"environment_type": "daytona"}},
                "environment": {
                    "task_sessions": {
                        "session": {"task_session": "nemotron_ultra"},
                        "sources": {"harbor/fixture": {"session": {"task_session": "shellbox"}}}
                        if harbor_source
                        else {},
                    }
                },
            }
        }
    )
    assert launcher._rl_config_uses_daytona(config) is harbor_source
