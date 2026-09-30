import torch
from torch import nn

from skyrl_train.mismatch_probe.capture import capture_layer_regions, write_capture


class _Scale(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, hidden_states):
        return hidden_states * self.factor


class _Router(nn.Module):
    def forward(self, hidden_states):
        return hidden_states.softmax(-1), hidden_states > 0


class _MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = _Router()
        self.shared_experts = _Scale(0.5)

    def forward(self, hidden_states):
        self.router(hidden_states)
        return hidden_states + self.shared_experts(hidden_states), None


class _Layer(nn.Module):
    def __init__(self, layer_number):
        super().__init__()
        self.layer_number = layer_number
        self.input_layernorm = _Scale(2.0)
        self.self_attention = _Scale(3.0)
        self.pre_mlp_layernorm = _Scale(5.0)
        self.mlp = _MoE()

    def forward(self, hidden_states):
        residual = hidden_states + self.self_attention(self.input_layernorm(hidden_states))
        mlp_output, _ = self.mlp(self.pre_mlp_layernorm(residual))
        return residual + mlp_output, None


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(1), _Layer(2)])


class _Wrapped(nn.Module):
    """Stand-in for DDP(Float16Module(GPTModel)) nesting."""

    def __init__(self):
        super().__init__()
        self.module = nn.Module()
        self.module.decoder = _Decoder()


def test_capture_records_each_region_of_the_first_pass_through_the_listed_layer(tmp_path):
    model = _Wrapped()
    first, second = torch.tensor([[1.0, -2.0]]), torch.tensor([[4.0, 4.0]])
    with capture_layer_regions([model], [1], str(tmp_path), enabled=True) as captured:
        for batch in (first, second):
            hidden = batch
            for layer in model.module.decoder.layers:
                hidden, _ = layer(hidden)
    regions = captured[1]
    # Each layer maps x to 7x after attention and to 7x + 52.5x = 59.5x after the MoE block.
    layer_input = first * 59.5
    assert torch.equal(regions["input"], layer_input)
    assert torch.equal(regions["attention_norm"], layer_input * 2)
    assert torch.equal(regions["attention"], layer_input * 6)
    assert torch.equal(regions["residual_after_attention"], layer_input * 7)
    assert torch.equal(regions["mlp_norm"], layer_input * 35)
    assert torch.equal(regions["router_map"], layer_input > 0)
    assert torch.equal(regions["shared_expert"], layer_input * 17.5)
    assert torch.equal(regions["mlp"], layer_input * 52.5)
    assert torch.equal(regions["output"], layer_input * 59.5)
    assert 0 not in captured

    destination = str(tmp_path / "pp-0.pt")
    write_capture(destination, captured, {"sequences": torch.tensor([[3, 4]])})
    restored = torch.load(destination)
    assert torch.equal(restored["layers"][1]["mlp"], regions["mlp"])
    assert restored["sequences"].tolist() == [[3, 4]]


def test_capture_is_inert_on_ranks_that_do_not_record(tmp_path):
    model = _Wrapped()
    with capture_layer_regions([model], [0, 1], str(tmp_path), enabled=False) as captured:
        model.module.decoder.layers[0](torch.ones(1, 2))
    assert captured == {}
    assert not model.module.decoder.layers[0]._forward_hooks
