# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay the actual native stage-2 driver arguments and Hydra composition on CPU."""

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import tempfile
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch

from cloud.iris import iris_backend as backend
from cloud.iris import training_driver as driver
from cloud.iris.protocol import job_spec
from cloud.iris.rl_config_translation import apply_context_budget_overrides, build_skyrl_hydra_args, parse_rl_config
from cloud.iris.runtime_bundle import runtime_bundle_inputs
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def compose_packet(report: dict, output: Path) -> None:
    source_root = Path(importlib.util.find_spec("cloud").submodule_search_locations[0]).parent
    for name, digest in report.get("msr_source_hashes", {}).items():
        if "/tests/" in name or name in {"pyproject.toml", "uv.lock"}:
            continue
        target = source_root / name
        if not target.exists():
            installed = name.removeprefix("skyrl-train/").removeprefix("skyrl-gym/")
            target = source_root / installed
        assert target.is_file(), ("Missing packaged production source", name)
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest, name
    request = report["request"]
    execution = {**report["execution"], "job_name": request["run_id"].replace("/", "-") + "-" + request["attempt_id"]}
    spec = job_spec({"schema_version": 2, "request": request, "execution": execution})
    if os.environ.get("READBACK_LOCAL_SOURCE"):
        with contextlib.chdir(os.environ["READBACK_LOCAL_SOURCE"]):
            runtime_bundle_inputs(report["msr"])
    else:
        runtime_bundle_inputs(report["msr"])
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as config:
        config.write(request["config_yaml"])
        config.flush()
        args = backend.resolved_launch_args(backend.job_launch_argv(spec, config.name))
        # Observe the exact argv before its shell escaping; no command is executed.
        with patch.object(backend, "_build_task_shell", side_effect=lambda _args, command, _path: command):
            controller = backend.build_task_command(args)
        command = controller[controller.index("--") + 1 :]
        assert command[1:3] == ["-m", "cloud.iris.training_driver"]
        values = vars(driver.create_parser().parse_args(command[3:]))
        values.update(
            rl_config_path=config.name,
            train_data=driver.parse_list_arg(values["train_data"]),
            val_data=driver.parse_list_arg(values["val_data"]),
            skyrl_overrides=values["skyrl_override"] or [],
        )
        cfg = driver.LocalRLConfig(**{f.name: values[f.name] for f in fields(driver.LocalRLConfig) if f.name in values})
        runner = driver.LocalRLRunner(cfg)
        parsed = parse_rl_config(config.name, model_override=cfg.model_path)
        parsed, overrides = apply_context_budget_overrides(parsed, cfg.skyrl_overrides)
        generated = build_skyrl_hydra_args(
            parsed, runner._build_exp_args(), driver._LocalHPCStub(gpus_per_node=cfg.gpus, cpus_per_node=cfg.cpus)
        )
        generated.extend(overrides)
        location = importlib.util.find_spec("skyrl_train").submodule_search_locations[0]
        config_dir = str(Path(location) / "config")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            composed = compose(config_name="ppo_base_config", overrides=generated)

        assert parsed.entrypoint == "skyrl_train.entrypoints.fully_async"
        assert (
            composed.trainer.max_steps == 25
            and composed.trainer.eval_interval == 25
            and composed.trainer.eval_before_train
        )
        assert composed.trainer.ckpt_interval == composed.trainer.hf_save_interval == -1
        assert composed.trainer.seed == 17 and composed.data.num_workers == 0 and composed.data.epoch_seeded_shuffle
        assert (
            composed.trainer.fully_async.weight_sync_interval == 2
            and composed.trainer.fully_async.max_staleness_steps == 1
        )
        assert (
            composed.trainer.fully_async.eval_mode == "blocking"
            and not composed.trainer.fully_async.eval_on_installed_weights
        )
        assert composed.generator.non_agentic_parser_protocol == "post-thinking-native-v1"
        assert (
            composed.generator.sampling_params.max_generate_length == 4096
            and composed.generator.eval_sampling_params.max_generate_length == 4096
        )
        assert list(composed.generator.sampling_params.stop_token_ids) == [128001, 128009]
        assert composed.generator.engine_init_kwargs.logprobs_mode == "raw_logprobs"
        assert composed.generator.max_input_length == 4096 and composed.trainer.max_prompt_length == 4096
        assert composed.trainer.placement.policy_num_nodes == 4 and composed.generator.num_inference_engines == 1
        assert (
            composed.generator.inference_engine_data_parallel_size
            == composed.generator.inference_engine_expert_parallel_size
            == 8
        )
        assert (
            composed.trainer.algorithm.policy_loss_type == "behavior_clip"
            and not composed.trainer.algorithm.use_kl_loss
            and not composed.trainer.algorithm.use_kl_in_reward
        )
        assert list(composed.generator.non_agentic_eval_endpoints) == report["endpoint_evidence"]["generated_modes"]
        assert composed.trainer.dump_data_batch is True
        assert composed.generator.trajectory_reward_shaping.overlong.l_max == 4096
        assert composed.generator.trajectory_reward_shaping.overlong.l_cache == 512
        kind = report["arm"]
        if kind in ("force_close", "repetition_stop"):
            intervention = composed.generator.non_agentic_intervention
            assert intervention.kind == kind and intervention.force_close_after == 3072
            assert (
                intervention.repetition_window == 256
                and intervention.repetition_ngram == 16
                and intervention.repetition_fraction == 0.5
            )
            assert intervention.thinking_end_id == 128003 and intervention.eos_id == 128001
        else:
            assert composed.generator.non_agentic_intervention is None
        assert composed.generator.trajectory_reward_shaping.enabled == (kind == "soft_overlong")
        assert composed.trainer.algorithm.non_agentic_truncated_advantage_cap == (
            -1.0 if kind == "negative_truncation_advantage" else None
        )
        result = {
            "arm": report["arm"],
            "runtime": report["msr"],
            "request_hash": report["request_hash"],
            "hydra_args": generated,
            "composed": OmegaConf.to_container(composed, resolve=True),
        }
        output.write_text(json.dumps(result, indent=2))
        print("E61_STAGE2_ACTUAL_NATIVE_HYDRA_PASS", report["arm"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compose_packet(json.loads(args.packet.read_bytes()), args.output)
