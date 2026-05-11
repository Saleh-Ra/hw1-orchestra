"""Load experiment variant specs and apply them to a base config."""

import json
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

from .defaults import PipelineDefaults

_TUPLE_FIELDS = ("frequencies_hz", "amplitude_range", "phase_range")

DEFAULT_EXPERIMENTS_PATH: Path = (
    Path(__file__).resolve().parents[2] / "config" / "experiments.json"
)


@dataclass(frozen=True)
class ExperimentSpec:
    """One named override on top of a base :class:`PipelineDefaults`."""

    name: str
    description: str
    overrides: dict[str, Any]


def load_experiments(
    path: str | Path | None = None,
) -> dict[str, ExperimentSpec]:
    """Load every experiment from a JSON file, keyed by ``name``."""
    experiments_path = (
        Path(path) if path is not None else DEFAULT_EXPERIMENTS_PATH
    )
    if not experiments_path.is_file():
        msg = f"Experiments file not found: {experiments_path}"
        raise FileNotFoundError(msg)

    with experiments_path.open(encoding="utf-8") as file_handle:
        raw: Any = json.load(file_handle)
    if not isinstance(raw, dict) or "experiments" not in raw:
        msg = "Experiments file must be a JSON object with an 'experiments' list."
        raise ValueError(msg)

    experiments: dict[str, ExperimentSpec] = {}
    for entry in raw["experiments"]:
        spec = _parse_experiment_entry(entry)
        if spec.name in experiments:
            msg = f"Duplicate experiment name: '{spec.name}'."
            raise ValueError(msg)
        experiments[spec.name] = spec
    return experiments


def apply_overrides(
    base: PipelineDefaults,
    overrides: dict[str, Any],
) -> PipelineDefaults:
    """Return a validated copy of ``base`` with ``overrides`` applied."""
    valid_field_names = {field_info.name for field_info in fields(PipelineDefaults)}
    unknown_fields = set(overrides) - valid_field_names
    if unknown_fields:
        msg = f"Unknown override fields: {sorted(unknown_fields)}"
        raise ValueError(msg)

    coerced: dict[str, Any] = {}
    for field_name, value in overrides.items():
        if field_name in _TUPLE_FIELDS and isinstance(value, list):
            coerced[field_name] = tuple(value)
        else:
            coerced[field_name] = value

    try:
        merged = replace(base, **coerced)
    except TypeError as error:
        msg = f"Invalid override types: {error}"
        raise ValueError(msg) from error
    merged.validate()
    return merged


def _parse_experiment_entry(entry: Any) -> ExperimentSpec:
    if not isinstance(entry, dict):
        msg = "Each experiment entry must be a JSON object."
        raise ValueError(msg)
    for required in ("name", "description", "overrides"):
        if required not in entry:
            msg = f"Experiment entry is missing required field '{required}'."
            raise ValueError(msg)
    if not isinstance(entry["overrides"], dict):
        msg = "Experiment 'overrides' must be a JSON object."
        raise ValueError(msg)
    return ExperimentSpec(
        name=str(entry["name"]),
        description=str(entry["description"]),
        overrides=dict(entry["overrides"]),
    )
