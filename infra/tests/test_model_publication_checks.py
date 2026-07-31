import pytest

from infra.rl_cleanup.publication_checks import model_publication_status, require_complete_model_publication


def test_model_publication_status_requires_logs_and_trace_link(tmp_path):
    (tmp_path / "README.md").write_text("# Model\n")

    status = model_publication_status(tmp_path)

    assert status == {
        "weights": "absent",
        "tokenizer_configuration": "absent",
        "training_logs": "absent",
        "training_traces_link": "absent",
    }
    with pytest.raises(ValueError):
        require_complete_model_publication(tmp_path)


def test_model_publication_status_accepts_complete_remote_download(tmp_path):
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (tmp_path / "config.json").write_text("{}\n")
    (tmp_path / "tokenizer.json").write_text("{}\n")
    logs = tmp_path / "training_logs"
    logs.mkdir()
    (logs / "metrics.csv").write_text("step,reward\n1,1\n")
    (tmp_path / "README.md").write_text(
        "# Model\n\n## Training Traces\n\nhttps://huggingface.co/datasets/laion/example-traces\n"
    )

    assert require_complete_model_publication(tmp_path) == {
        "weights": "present",
        "tokenizer_configuration": "present",
        "training_logs": "present",
        "training_traces_link": "present",
    }
