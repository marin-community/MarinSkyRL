"""The Taskforge gen1000 reference recipes compose into the SkyRL config they declare."""

from pathlib import Path

import pytest
import yaml

from cloud.iris.rl_config_translation import compose_skyrl_config, parse_rl_config
from cloud.iris.role_plan import derive_num_nodes, derive_role_plan

CONFIGS = Path(__file__).parents[1] / "configs"


class _HPCStub:
    gpus_per_node = 8


@pytest.mark.parametrize(
    ("name", "entrypoint"),
    [
        ("taskforge_gen1000_qwen3_30b_a3b_thinking", "skyrl_train.entrypoints.taskcompendium"),
        ("taskforge_gen1000_qwen3_30b_a3b_thinking_generate", "skyrl_train.entrypoints.taskcompendium_generate"),
    ],
)
def test_taskforge_recipe_enables_local_machines_and_the_2507_template(name, entrypoint):
    path = CONFIGS / f"{name}.yaml"
    parsed = parse_rl_config(str(path), model_override="Qwen/Qwen3-30B-A3B-Thinking-2507")
    compiled = compose_skyrl_config(
        parsed, {"job_name": name, "experiments_dir": "/tmp/exp", "num_nodes": 3}, _HPCStub()
    )
    config = compiled.config

    assert compiled.entrypoint == entrypoint
    assert dict(config.environment.task_sessions.local) == {}
    assert dict(config.environment.task_sessions.guided_shell) == {}
    assert config.generator.chat_template.name_or_path == "chat_templates/qwen3_thinking_2507_acc.jinja2"
    assert config.generator.engine_init_kwargs.tool_call_parser == "hermes"
    assert config.generator.sampling_params.max_generate_length == 32768
    batch = config.trainer.train_batch_size * config.generator.n_samples_per_prompt
    workers = config.trajectory_runner.rollout_workers.num_workers
    assert config.trajectory_runner.max_concurrent_tasks * workers >= batch
    assert derive_num_nodes(derive_role_plan(yaml.safe_load(path.read_text()))) == 3
