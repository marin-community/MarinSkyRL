import pytest
from loguru import logger

from skyrl_train.utils.logging_utils import log_example


@pytest.fixture
def records():
    captured = []
    sink_id = logger.add(captured.append, format="{level}|{message}", colorize=False)
    yield captured
    logger.remove(sink_id)


def test_log_example_logs_braces_and_markup_verbatim(records):
    """Model text containing str.format braces and loguru color tags must not be interpreted (#781)."""
    prompt = [{"role": "user", "content": "fill {answer} in <red>bold</red>"}]
    response = "def f():\n    return {'a': 1} </green> {0}"

    log_example(logger, prompt=prompt, response=response, reward=[0.25, 0.5])

    assert len(records) == 1
    level, message = records[0].rstrip("\n").split("|", 1)
    assert level == "INFO"
    assert str(prompt) in message
    assert "def f():\n    return {'a': 1} </green> {0}" in message
    assert "0.7500" in message
