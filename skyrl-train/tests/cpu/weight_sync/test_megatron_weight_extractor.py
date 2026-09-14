from types import SimpleNamespace

import pytest
import torch

from skyrl_train.workers.megatron.weight_extractor import MegatronWeightExtractor


class _GroupedExpertBridge:
    """Model the bridge contract that emits an expert tensor only from a complete task stream."""

    names = (
        "model.layers.0.mlp.experts.gate_proj.weight",
        "model.layers.0.mlp.experts.up_proj.weight",
        "model.layers.0.mlp.experts.down_proj.weight",
    )
    emitted_names = names

    def get_conversion_tasks(self, actor_module):
        del actor_module
        return [
            SimpleNamespace(
                mapping=SimpleNamespace(
                    is_grouped_export=True,
                    hf_param={"gate": self.names[0], "up": self.names[1]},
                )
            ),
            SimpleNamespace(mapping=SimpleNamespace(is_grouped_export=True, hf_param=self.names[2])),
        ]

    def export_hf_weights(self, actor_module, *, show_progress, conversion_tasks):
        del actor_module, show_progress
        if len(conversion_tasks) != 2:
            return
        for index, name in enumerate(self.emitted_names):
            yield name, torch.arange(index * 6, (index + 1) * 6, dtype=torch.float32)


class _IncompleteGroupedExpertBridge(_GroupedExpertBridge):
    emitted_names = _GroupedExpertBridge.names[:-1]


def test_bucketed_extraction_preserves_grouped_expert_exports(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    extractor = MegatronWeightExtractor(
        bridge=_GroupedExpertBridge(),
        actor_module=object(),
        model_type="test",
        enable_bucketing=True,
        bucket_size_threshold_GB=16 / 1024**3,
    )

    chunks = list(extractor.extract_weights(torch.float32))

    assert [name for chunk in chunks for name in chunk.names] == [
        "model.layers.0.mlp.experts.gate_proj.weight",
        "model.layers.0.mlp.experts.up_proj.weight",
        "model.layers.0.mlp.experts.down_proj.weight",
    ]
    assert all(chunk.total_size_bytes <= 24 for chunk in chunks)


def test_bucketed_extraction_rejects_missing_grouped_expert_exports(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    extractor = MegatronWeightExtractor(
        bridge=_IncompleteGroupedExpertBridge(),
        actor_module=object(),
        model_type="test",
        enable_bucketing=True,
    )

    with pytest.raises(RuntimeError, match="omitted grouped weight exports"):
        list(extractor.extract_weights(torch.float32))
