"""Replay checked-in responses through original and verifyit framework dispatch."""

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import skyrl_gym
from omegaconf import OmegaConf

from skyrl_gym.envs.nemotron_ultra.mcqa import grade_mcqa
from skyrl_gym.envs.nemotron_ultra.judge import OpenAIJudge
from skyrl_gym.envs.nemotron_ultra.math_with_judge import grade_math
from skyrl_gym.envs.nemotron_ultra.math_judge_verifyit import grade_math_verifyit


class FixtureJudge(BaseHTTPRequestHandler):
    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        reply = json.dumps({"id": "offline-fixture", "object": "chat.completion", "created": 0,
                            "model": request["model"], "choices": [{"index": 0, "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "[[A!=B]]"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixtures = json.loads(Path(__file__).with_name("fixtures.json").read_text())
    results = []
    for row in fixtures["mcqa"]:
        original = grade_mcqa(row["response"], row["record"])
        cutover = grade_mcqa(row["response"], row["record"], verifyit_enabled=True)
        results.append({"route": "mcqa", "input": row, "original": original, "verifyit": cutover})
        if cutover != original:
            raise RuntimeError(f"MCQA parity mismatch: {results[-1]}")
    for row in fixtures["reasoning_gym"]:
        extras = {"reward_model": {"ground_truth": {"task": "simple_equations", "entry": row["entry"]}}}
        original = skyrl_gym.make("reasoning_gym", env_config=OmegaConf.create({}), extras=extras)
        cutover = skyrl_gym.make(
            "reasoning_gym", env_config=OmegaConf.create({"verifyit_enabled": True}), extras=extras
        )
        source_result = original.step(row["response"])
        cutover_result = cutover.step(row["response"])
        results.append(
            {"route": "reasoning_gym", "input": row, "original": source_result, "verifyit": cutover_result}
        )
        if cutover_result["reward"] != source_result["reward"]:
            raise RuntimeError(f"Reasoning Gym parity mismatch: {results[-1]}")
    server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureJudge)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        judge = OpenAIJudge(base_url=f"http://127.0.0.1:{server.server_port}/v1", model="offline-fixture")
        for row in fixtures["math"]:
            original = grade_math(row["response"], row["record"], judge=judge)
            cutover = grade_math_verifyit(row["response"], row["record"], judge=judge)
            results.append({"route": "math", "input": row, "original": original, "verifyit": cutover})
            if cutover != original:
                raise RuntimeError(f"Math parity mismatch: {results[-1]}")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(f"Matched {len(results)} original/cutover fixture pairs; evidence: {args.output}")


if __name__ == "__main__":
    main()
