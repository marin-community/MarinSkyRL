# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle events must actually serialise.

Rigging's ``event()`` takes an ``EventBody``; a bare dict raises ``AttributeError`` inside
``event_fields`` (``dict(body.fields)``), and the exporter swallows that into ``lost_records``. So
the failure is invisible: no exception reaches the caller and the event simply never appears. Both
``lifecycle`` and ``terminal`` were dropped on every run until this was fixed.
"""

from __future__ import annotations

import pytest
from rigging.telemetry import serialization

from skyrl_train.telemetry import _event_body


def test_event_body_is_accepted_by_the_serialiser():
    body = _event_body({"state": "started"})
    assert isinstance(body, serialization.EventBody)
    assert serialization.event_fields(body, 1024)


def test_a_bare_dict_is_rejected_which_is_why_the_wrapper_exists():
    """Guards the regression: passing the mapping straight through loses the event."""
    with pytest.raises(AttributeError):
        serialization.event_fields({"state": "started"}, 1024)  # type: ignore[arg-type]


def test_none_valued_fields_are_dropped_not_raised():
    """policy_step, queue_depth and the progress timestamp are all None before the first step.

    event_fields accepts only str/int/float/bool, so keeping a None would raise and lose the WHOLE
    event rather than the one field.
    """
    body = _event_body(
        {
            "status": "completed",
            "export_lost_records": 3,
            "policy_step": None,
            "queue_depth": None,
            "last_progress_time_seconds": None,
            "ratio": 1.5,
            "ok": True,
        }
    )
    assert dict(body.fields) == {
        "status": "completed",
        "export_lost_records": 3,
        "ratio": 1.5,
        "ok": True,
    }
    assert serialization.event_fields(body, 1024)


def test_the_inert_fallback_mirrors_the_real_signature():
    """The inert stub named its second parameter `fields: Mapping[str, object]`, which type-checked
    a caller passing a bare dict and hid the AttributeError on the live path."""
    import inspect

    from skyrl_train import inert_telemetry

    assert list(inspect.signature(inert_telemetry.event).parameters) == ["name", "body", "attributes"]
