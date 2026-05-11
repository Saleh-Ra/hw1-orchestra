import json
from pathlib import Path

import pytest

from ai_orchestra import (
    DEFAULT_EXPERIMENTS_PATH,
    DEFAULTS,
    ExperimentSpec,
    PipelineDefaults,
    apply_overrides,
    load_experiments,
)


def test_default_experiments_file_loads() -> None:
    experiments = load_experiments()

    assert len(experiments) >= 1
    assert all(isinstance(spec, ExperimentSpec) for spec in experiments.values())
    assert all(spec.overrides for spec in experiments.values())


def test_default_experiments_file_path_exists() -> None:
    assert DEFAULT_EXPERIMENTS_PATH.is_file()


def test_load_experiments_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Experiments file not found"):
        load_experiments(tmp_path / "absent.json")


def test_load_experiments_rejects_non_object_root(tmp_path: Path) -> None:
    path = tmp_path / "list_root.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    with pytest.raises(ValueError, match="experiments"):
        load_experiments(path)


def test_load_experiments_rejects_duplicate_names(tmp_path: Path) -> None:
    payload = {
        "experiments": [
            {"name": "dup", "description": "A", "overrides": {"batch_size": 32}},
            {"name": "dup", "description": "B", "overrides": {"batch_size": 16}},
        ]
    }
    path = tmp_path / "dup.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate experiment name"):
        load_experiments(path)


def test_load_experiments_rejects_missing_required_field(tmp_path: Path) -> None:
    payload = {"experiments": [{"name": "x", "overrides": {"batch_size": 32}}]}
    path = tmp_path / "missing_desc.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="description"):
        load_experiments(path)


def test_apply_overrides_returns_new_validated_defaults() -> None:
    merged = apply_overrides(DEFAULTS, {"hidden_size": 64, "num_layers": 2})

    assert isinstance(merged, PipelineDefaults)
    assert merged.hidden_size == 64
    assert merged.num_layers == 2
    assert merged.batch_size == DEFAULTS.batch_size
    assert DEFAULTS.hidden_size == 32


def test_apply_overrides_rejects_unknown_field() -> None:
    with pytest.raises(ValueError, match="Unknown override fields"):
        apply_overrides(DEFAULTS, {"mystery": 1.0})


def test_apply_overrides_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        apply_overrides(DEFAULTS, {"learning_rate": -0.1})


def test_apply_overrides_coerces_list_into_tuple_fields() -> None:
    merged = apply_overrides(DEFAULTS, {"frequencies_hz": [1.0, 2.0]})

    assert merged.frequencies_hz == (1.0, 2.0)


def test_default_experiments_cover_milestone_categories() -> None:
    names = set(load_experiments())

    assert "noise_low" in names
    assert "noise_medium" in names
    assert "noise_high" in names
    assert "hidden_small" in names
    assert "hidden_large" in names
    assert "layers_one" in names
    assert "layers_two" in names
    assert "window_10" in names
    assert "window_large" in names
    assert "split_80_20" in names
    assert "split_70_30" in names
    assert "split_90_10" in names
