"""JSON-backed loader for :class:`PipelineDefaults` tunables.

The dataclass in :mod:`ai_orchestra.defaults` stores the in-code mirror of
``config/default.json``. ``config/default.json`` is the source of truth for
the project's official tunables; the dataclass keeps matching fallback
values so that import-time code never has to perform disk I/O.
"""

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from .defaults import PipelineDefaults

_TUPLE_FIELDS = ("frequencies_hz", "amplitude_range", "phase_range")

DEFAULT_CONFIG_PATH: Path = (
    Path(__file__).resolve().parents[2] / "config" / "default.json"
)


def load_pipeline_defaults(
    path: str | Path | None = None,
) -> PipelineDefaults:
    """Load and validate :class:`PipelineDefaults` from a JSON config file.

    When ``path`` is ``None`` the project default at
    ``config/default.json`` is used.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)

    with config_path.open(encoding="utf-8") as file_handle:
        raw_config: Any = json.load(file_handle)
    if not isinstance(raw_config, dict):
        msg = "Config root must be a JSON object."
        raise ValueError(msg)

    valid_field_names = {field_info.name for field_info in fields(PipelineDefaults)}
    unknown_fields = set(raw_config) - valid_field_names
    if unknown_fields:
        msg = f"Unknown config fields: {sorted(unknown_fields)}"
        raise ValueError(msg)

    typed_config = _coerce_tuple_fields(raw_config)
    try:
        defaults = PipelineDefaults(**typed_config)
    except TypeError as error:
        msg = f"Invalid config field types: {error}"
        raise ValueError(msg) from error
    defaults.validate()
    return defaults


def _coerce_tuple_fields(raw_config: dict[str, Any]) -> dict[str, Any]:
    """Convert JSON lists into tuples for tuple-typed dataclass fields."""
    coerced: dict[str, Any] = dict(raw_config)
    for tuple_field in _TUPLE_FIELDS:
        if tuple_field in coerced:
            value = coerced[tuple_field]
            if not isinstance(value, list):
                msg = f"Config field '{tuple_field}' must be a JSON array."
                raise ValueError(msg)
            coerced[tuple_field] = tuple(value)
    return coerced
