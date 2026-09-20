"""Plain BF16 Hero score reference for an existing trained replay rollout.

Run in a one-H100 Iris task with HERO_REPLAY_TRAINED_URI,
HERO_REPLAY_REFERENCE_INPUT_URI (the live replay result key), and
HERO_REPLAY_REFERENCE_RESULT_URI. This is observational and does not train.
"""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from safetensors import safe_open
import torch

from skyrl_train.models.grug_moe import GrugMoeForCausalLM
from tests.gpu.hero_replay_checkpoint import split_stacked_hero_checkpoint
from tests.gpu.test_hero_router_replay_live import (
    _put_s3_json,
    _s3_client,
    _s3_target,
    _stage_trained_checkpoint,
)


def _read_json(uri: str) -> dict:
    bucket, key = _s3_target(uri)
    return json.loads(_s3_client().get_object(Bucket=bucket, Key=key)["Body"].read())


def _selected_checkpoint_diffs(model: GrugMoeForCausalLM, path: Path) -> dict[str, float]:
    names = (
        "model.layers.0.mlp.router.weight",
        "model.layers.0.mlp.router.bias",
        "model.layers.0.mlp.experts.3.gate_proj.weight",
        "model.layers.15.mlp.experts.383.gate_proj.weight",
        "model.layers.0.mlp.latent_down_proj.weight",
        "model.layers.0.shared_experts.1.up_proj.weight",
        "model.layers.0.self_attn.sconv_k.weight",
    )
    weight_map = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    state = model.state_dict()
    result = {}
    for name in names:
        with safe_open(path / weight_map[name], framework="pt", device="cpu") as reader:
            expected = reader.get_tensor(name)
        actual = state[name].detach().cpu()
        assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
        result[name] = (actual.float() - expected.float()).abs().max().item()
    return result


def main() -> None:
    trained_uri = os.environ["HERO_REPLAY_TRAINED_URI"]
    input_uri = os.environ["HERO_REPLAY_REFERENCE_INPUT_URI"]
    output_uri = os.environ["HERO_REPLAY_REFERENCE_RESULT_URI"]
    rollout = _read_json(input_uri.removesuffix(".json") + "-rollout.json")
    live_result = _read_json(input_uri)
    prompts = rollout["prompts"]
    responses = rollout["response_ids"]
    assert len(prompts) == len(responses) == 4
    assert len({len(prompt) for prompt in prompts}) == 1
    assert all(len(response) == 4 for response in responses)
    prompt_length = len(prompts[0])
    sequences = torch.tensor([prompt + response for prompt, response in zip(prompts, responses, strict=True)])

    with TemporaryDirectory(prefix="hero-hf-reference-") as directory:
        source = Path(directory) / "source"
        destination = Path(directory) / "split"
        source.mkdir()
        staged_bytes = _stage_trained_checkpoint(trained_uri, source)
        split_count = split_stacked_hero_checkpoint(source, destination)
        model = (
            GrugMoeForCausalLM.from_pretrained(destination, dtype=torch.bfloat16, attn_implementation="eager")
            .eval()
            .to("cuda")
        )
        checkpoint_diffs = _selected_checkpoint_diffs(model, destination)
        tokens = sequences.to("cuda")
        position_ids = torch.arange(tokens.shape[1], device=tokens.device)[None, :].expand_as(tokens)
        with torch.no_grad():
            logits = model(tokens, position_ids=position_ids, use_cache=False).logits
            next_token_logprobs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
            next_token_logprobs = next_token_logprobs.gather(-1, tokens[:, 1:, None]).squeeze(-1)
            hf_scores = next_token_logprobs[:, prompt_length - 1 :].cpu()

    serving = torch.tensor(live_result["score_diagnostic"]["serving_response_logprobs"])
    native = torch.tensor(live_result["score_diagnostic"]["native_response_logprobs"])
    replay = torch.tensor(live_result["score_diagnostic"]["replay_response_logprobs"])
    report = {
        "model": trained_uri,
        "rollout_result": input_uri,
        "staged_checkpoint_bytes": staged_bytes,
        "converted_split_expert_tensors": split_count,
        "selected_checkpoint_max_abs_diffs": checkpoint_diffs,
        "response_ids": responses,
        "hf_response_logprobs": hf_scores.tolist(),
        "hf_vs_serving_max_abs": (hf_scores - serving).abs().max().item(),
        "hf_vs_megatron_native_max_abs": (hf_scores - native).abs().max().item(),
        "hf_vs_megatron_replay_max_abs": (hf_scores - replay).abs().max().item(),
    }
    _put_s3_json(output_uri, report)
    print("HERO_TRAINED_HF_REFERENCE=" + json.dumps(report, sort_keys=True), flush=True)
    assert all(diff == 0 for diff in checkpoint_diffs.values()), checkpoint_diffs


if __name__ == "__main__":
    main()
