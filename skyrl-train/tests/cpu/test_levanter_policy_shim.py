import io
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
import ray
import torch
from omegaconf import OmegaConf

from skyrl_train.distributed.dispatch import concatenate_outputs_after_mesh_dispatch
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.workers.levanter_policy import LevanterPolicyActorGroup


class _PolicyHandler(BaseHTTPRequestHandler):
    step = 0

    def log_message(self, _format, *_args) -> None:
        return None

    def do_POST(self) -> None:
        if self.path == "/broadcast_weights":
            body = json.dumps({"step": self.step}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        length = int(self.headers["content-length"])
        with np.load(io.BytesIO(self.rfile.read(length)), allow_pickle=False) as batch:
            action_count = int(batch["action_count"])
            action_log_probs = -batch["sequences"][:, -action_count:].astype(np.float32)

        output = io.BytesIO()
        if self.path == "/ppo_train":
            type(self).step += 1
            np.savez(
                output,
                action_log_probs=action_log_probs,
                loss=np.asarray(0.25, dtype=np.float32),
                step=np.asarray(self.step, dtype=np.int64),
            )
        else:
            np.save(output, action_log_probs, allow_pickle=False)
        body = output.getvalue()
        self.send_response(200)
        self.send_header("content-type", "application/octet-stream")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@contextmanager
def _policy_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _PolicyHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.usefixtures("ray_init")
def test_levanter_actor_group_preserves_skyrl_forward_train_and_publish_contract() -> None:
    with _policy_server() as endpoint:
        cfg = OmegaConf.create(
            {
                "trainer": {
                    "policy": {
                        "levanter": {
                            "endpoint_urls": [endpoint],
                            "data_parallel_size": 1,
                            "request_timeout_seconds": 10,
                        }
                    }
                },
                "generator": {"r3_transport": "by_value", "r3_dispatch_put_timeout_seconds": 10},
            }
        )
        group = LevanterPolicyActorGroup(cfg)
        batch = TrainingInputBatch(
            {
                "sequences": torch.tensor([[1, 2, 3, 4]], dtype=torch.int64),
                "action_log_probs": torch.zeros((1, 2)),
                "advantages": torch.ones((1, 2)),
                "loss_mask": torch.ones((1, 2)),
            }
        )
        batch.metadata = {"response_length": 2}

        forward_refs = group.async_run_ray_method("mesh", "forward", data=batch)
        forward = concatenate_outputs_after_mesh_dispatch(group.actor_infos, ray.get(forward_refs))
        train_refs = group.async_run_ray_method("mesh", "ppo_train", batch)
        trained = ray.get(train_refs)[0]
        publish_refs = group.async_run_ray_method("pass_through", "broadcast_to_inference_engines", None)
        published_step = ray.get(publish_refs)[0]

        np.testing.assert_array_equal(forward["output"].numpy(), [[-3.0, -4.0]])
        assert trained.metadata["train_status"]["policy_loss"] == 0.25
        assert trained.metadata["train_status"]["levanter_step"] == 1.0
        assert published_step == 1
