"""One-step Iris acceptance entrypoint for NVIDIA's released Ultra RLVR blends."""

from __future__ import annotations

import json
import tempfile
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator

import hydra
import ray
from loguru import logger
from omegaconf import DictConfig

from skyrl_train.config.trajectory_runner_capabilities import TrajectoryRunnerMode
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, run_ray_driver


class _VerifierHandler(BaseHTTPRequestHandler):
    """Deterministic protocol substitute; this gate tests plumbing, not judge quality."""

    server_version = "NemotronUltraAcceptance/1.0"

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length:
            json.loads(self.rfile.read(content_length))
        if self.path == "/execute":
            self._send({"process_status": "completed", "stdout": "4\n", "stderr": ""})
        elif self.path == "/chat/completions":
            verdicts = "A\n[[YES]]\n[[SAFE]] [[HAS_EXPLANATION]] [[HAS_HELPLINES]]\n[[A=B]]"
            self._send({"choices": [{"message": {"role": "assistant", "content": verdicts}}]})
        elif self.path == "/responses":
            self._send(
                {
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": '{"score_1": 3, "score_2": 3, "ranking": 3.5}',
                                }
                            ],
                        }
                    ]
                }
            )
        else:
            self.send_error(404)

    def _send(self, value: object) -> None:
        payload = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


@contextmanager
def verifier_server() -> Iterator[str]:
    """Serve every external verifier protocol on one loopback-only random port."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _VerifierHandler)
    thread = threading.Thread(target=server.serve_forever, name="nemotron-ultra-verifier", daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def prepare_acceptance_data(cfg: DictConfig, root: Path) -> dict[str, object]:
    """Materialize one live random row per phase/generator and its SWE tasks."""
    import datasets  # noqa: PLC0415 - keep the Hydra launcher import-light

    from cloud.iris.tasks_parquet import from_parquet  # noqa: PLC0415
    from infra.rl_data.nemotron_ultra_sample import write_generator_sample  # noqa: PLC0415
    from infra.rl_data.nemotron_ultra_swe import prepare_swe_task_artifact  # noqa: PLC0415

    sample_path = root / "generator-sample.parquet"
    manifest = write_generator_sample(sample_path, seed=int(cfg.trainer.seed))
    sample = datasets.load_dataset("parquet", data_files=str(sample_path), split="train")
    desired_paths = {
        ultra["terminal_bench_instance_id"]
        for row in sample
        if (ultra := row["extra_info"]["nemotron_ultra"])["route"] == "terminal_bench"
    }
    task_artifact = root / "swe-task-artifact"
    task_manifest = prepare_swe_task_artifact(task_artifact, desired_paths=desired_paths)
    task_root = root / "swe-tasks"
    from_parquet(str(task_artifact / "tasks.parquet"), str(task_root), max_workers=2)

    cfg.data.train_data = [str(sample_path)]
    cfg.data.val_data = []
    cfg.data.terminal_bench_data = [str(task_root)]
    return {**manifest, "swe_tasks": task_manifest["counts"]}


@ray.remote(num_cpus=1, max_retries=0)
def skyrl_entrypoint(cfg: DictConfig) -> None:
    with tempfile.TemporaryDirectory(prefix="nemotron-ultra-acceptance-") as directory:
        manifest = prepare_acceptance_data(cfg, Path(directory))
        logger.info("NEMOTRON_ULTRA_SAMPLE {}", json.dumps(manifest, sort_keys=True))
        with verifier_server() as base_url:
            cfg.environment.skyrl_gym.nemotron_ultra.sandbox.host = "127.0.0.1"
            cfg.environment.skyrl_gym.nemotron_ultra.sandbox.port = int(base_url.rsplit(":", 1)[1])
            for name in ("general", "safety"):
                cfg.environment.skyrl_gym.nemotron_ultra.judges[name].base_url = base_url
            cfg.environment.skyrl_gym.nemotron_ultra.genrm.judge.base_url = base_url
            BasePPOExp(cfg).run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_ray_driver(cfg, skyrl_entrypoint, TrajectoryRunnerMode.SKYRL_GYM)


if __name__ == "__main__":
    main()
