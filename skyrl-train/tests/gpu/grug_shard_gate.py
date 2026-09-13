"""Independent byte oracle and native worker operations for the tiny Grug gate."""

import torch


def tensor_bytes(value):
    return value.detach().contiguous().view(torch.uint8).reshape(-1).cpu()


def expected_receiver_parameters(hf_weights, expert_maps):
    """Construct full native parameters from the independent HF export and observed ownership."""
    expected = {}
    expert_names = set()
    for prefix, mapping in expert_maps.items():
        hf_prefix = prefix.removesuffix(".routed_experts")
        gate, up, down = (hf_weights[f"{hf_prefix}.{projection}_proj.weight"] for projection in ("gate", "up", "down"))
        assert len(mapping) == gate.shape[0] == up.shape[0] == down.shape[0]
        owners = sorted((slot, expert) for expert, slot in enumerate(mapping) if slot >= 0)
        assert [slot for slot, _ in owners] == list(range(len(owners)))
        expected[prefix + ".w13_weight"] = torch.stack(
            [torch.cat((gate[expert], up[expert]), dim=0) for _, expert in owners]
        ).to(torch.bfloat16)
        expected[prefix + ".w2_weight"] = torch.stack([down[expert] for _, expert in owners]).to(torch.bfloat16)
        expert_names.update(f"{hf_prefix}.{projection}_proj.weight" for projection in ("gate", "up", "down"))
    for name, value in hf_weights.items():
        if name in expert_names:
            continue
        expected[name] = value if name.endswith(".mlp.router.bias") else value.to(torch.bfloat16)
        if name.endswith(".mlp.router.weight"):
            expected[name] = expected[name].float()
    return expected


def compare_receiver_parameters(hf_weights, snapshot, *, expected_ep, experts):
    """Require exact complete model coverage and all bytes, including dense/shared/router weights."""
    assert snapshot["ep_size"] == expected_ep
    local = experts // expected_ep
    mapping = [-1] * experts
    begin = snapshot["ep_rank"] * local
    mapping[begin : begin + local] = list(range(local))
    assert snapshot["expert_maps"] and all(value == mapping for value in snapshot["expert_maps"].values())
    expected = expected_receiver_parameters(hf_weights, snapshot["expert_maps"])
    actual = snapshot["parameters"]
    assert set(actual) == set(expected), (set(actual) - set(expected), set(expected) - set(actual))
    compared = 0
    for name, wanted in expected.items():
        observed = actual[name]
        assert observed.dtype == wanted.dtype and observed.shape == wanted.shape, name
        assert torch.equal(tensor_bytes(observed), tensor_bytes(wanted)), name
        compared += wanted.numel() * wanted.element_size()
    return compared


def policy_local_snapshot(worker):
    """Copy each actual learner parameter without a Bridge gather that could hide non-root writes."""
    return {
        f"{index}/{name}": value.detach().cpu().clone()
        for index, module in enumerate(worker.actor_module)
        for name, value in module.named_parameters()
    }


def receiver_snapshot(worker):
    from vllm.distributed import get_ep_group

    model = worker.model_runner.model
    maps = {}
    for name, module in model.named_modules():
        if name.endswith(".experts.routed_experts"):
            maps[name] = module.expert_map.detach().cpu().tolist()
    ep = get_ep_group()
    return {
        "ep_rank": ep.rank_in_group,
        "ep_size": ep.world_size,
        "expert_maps": maps,
        "parameters": {name: value.detach().cpu().clone() for name, value in model.named_parameters()},
    }


def corrupt_receiver_byte(worker):
    """Emulate a silent device-byte error, without changing the Parameter's version counter."""
    from vllm.distributed import get_ep_group

    ep = get_ep_group()
    if ep.rank_in_group != 1:
        return {"changed": False, "ep_rank": ep.rank_in_group}
    name, parameter = next(
        (name, value)
        for name, value in worker.model_runner.model.named_parameters()
        if name.endswith(".experts.routed_experts.w2_weight")
    )
    version = parameter._version
    raw = parameter.data.view(torch.uint8).reshape(-1)
    raw[-1].bitwise_xor_(1)
    torch.cuda.synchronize(parameter.device)
    assert parameter._version == version
    return {"changed": True, "ep_rank": ep.rank_in_group, "parameter": name, "byte_offset": raw.numel() - 1}


def assert_source_preserved(before, after):
    assert len(before) == len(after)
    count = 0
    for previous, current in zip(before, after, strict=True):
        assert set(previous) == set(current)
        for name, value in previous.items():
            assert torch.equal(tensor_bytes(value), tensor_bytes(current[name])), name
            count += value.numel() * value.element_size()
    return count
