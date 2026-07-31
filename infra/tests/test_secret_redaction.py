import pytest

from infra.rl_cleanup.secret_redaction import redact, redact_record, redact_tree
from infra.rl_cleanup.trace_dataset_hygiene import sanitize_trace_record


JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJydW4ifQ.signature"


@pytest.mark.parametrize(
    ("shape", "value"),
    [
        ("pem_private_key", "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----"),
        ("iris_proxy_url", f"https://iris.oa.dev/proxy/t/{JWT}"),
        ("jwt", JWT),
        ("aws_access_key", "AKIA1234567890ABCDEF"),
        ("openai_key", "sk-12345678901234567890"),
        ("huggingface_token", "hf_12345678901234567890"),
        ("github_token", "ghp_12345678901234567890"),
    ],
)
def test_redact_recognizes_each_supported_credential_shape(shape, value):
    cleaned, findings = redact(value)

    assert cleaned == f"<redacted:{shape}>"
    assert [finding.shape for finding in findings] == [shape]


def test_redact_replaces_jwt_and_is_idempotent():
    cleaned, findings = redact(f"proxy token: {JWT}")

    assert cleaned == "proxy token: <redacted:jwt>"
    assert [finding.shape for finding in findings] == ["jwt"]
    assert redact(cleaned) == (cleaned, [])


def test_redact_record_cleans_nested_trace_content():
    record = {"conversations": [{"content": f"token={JWT}"}]}

    cleaned, findings = redact_record(record)

    assert cleaned["conversations"][0]["content"] == "token=<redacted:jwt>"
    assert findings[0].location == "conversations[0].content"


def test_redact_tree_rewrites_staged_training_logs(tmp_path):
    log = tmp_path / "training_logs" / "trainer.out"
    log.parent.mkdir()
    log.write_text(f"token={JWT}\n")

    findings = redact_tree(tmp_path)

    assert log.read_text() == "token=<redacted:jwt>\n"
    assert [finding.shape for finding in findings] == ["jwt"]


def test_sanitize_trace_record_redacts_publishable_conversations():
    record = {"conversations": [{"content": f"token={JWT}"}]}

    cleaned = sanitize_trace_record(record)

    assert cleaned["conversations"][0]["content"] == "token=<redacted:jwt>"
