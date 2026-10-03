"""Parse dotted settings using the mounted schema field."""

import json
from collections.abc import Mapping
from typing import Annotated, Any, get_args, get_origin

from pydantic import TypeAdapter

from .model import FrozenMap, Section


def parse_setting(model: type[Section], key: str, raw: str) -> Any:
    """Parse a setting without validating an incomplete section."""
    parts = key.split(".")
    if any(not part for part in parts):
        raise ValueError(f"invalid setting path {key!r}")
    for index, name in enumerate(parts):
        info = model.model_fields.get(name)
        if info is None:
            raise ValueError(f"unknown setting {key!r}: {model.__name__} has no field {name!r}")
        annotation = info.annotation
        options = tuple(_annotations(annotation))
        if index == len(parts) - 1:
            if info.metadata:
                annotation = Annotated[annotation, *info.metadata]
            return _parse_value(annotation, key, raw)
        section = next((option for option in options if isinstance(option, type) and issubclass(option, Section)), None)
        if section is not None:
            model = section
            continue
        if FrozenMap in options:
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
        mapping = next((option for option in options if get_origin(option) in (dict, Mapping)), None)
        if mapping is not None:
            if index != len(parts) - 2:
                raise ValueError(f"setting {key!r} descends into a scalar mapping value")
            return _parse_value(get_args(mapping)[1], key, raw)
        raise ValueError(f"setting {key!r} descends into a scalar at {name!r}")
    raise ValueError(f"invalid setting path {key!r}")


def _parse_value(annotation: Any, key: str, raw: str) -> Any:
    options = tuple(_annotations(annotation))
    if Any in options:
        raise ValueError(f"setting {key!r} targets an untyped field; add its sidecar annotation")
    if raw == "null" or any(
        option is FrozenMap
        or get_origin(option) in (tuple, dict, Mapping)
        or isinstance(option, type)
        and issubclass(option, Section)
        for option in options
    ):
        return json.loads(raw)
    adapter = TypeAdapter(annotation)
    parsed = adapter.validate_strings(raw)
    if int in options and float in options:
        try:
            numeric = json.loads(raw)
        except json.JSONDecodeError:
            return parsed
        if isinstance(numeric, int | float) and not isinstance(numeric, bool):
            return adapter.validate_python(numeric, strict=True)
    return parsed


def _annotations(annotation: Any):
    yield annotation
    for child in get_args(annotation):
        yield from _annotations(child)
