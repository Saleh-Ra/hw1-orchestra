import json
from dataclasses import fields
from pathlib import Path

import pytest

from ai_orchestra import DEFAULT_CONFIG_PATH, DEFAULTS, load_pipeline_defaults
from ai_orchestra.defaults import PipelineDefaults


def _baseline_payload() -> dict:
    return {
        "frequencies_hz": [1.0, 3.0, 5.0, 7.0],
        "sampling_frequency_hz": 1000,
        "num_samples": 10000,
        "duration_seconds": 10.0,
        "window_size": 10,
        "num_signal_sets": 50,
        "train_ratio": 0.8,
        "batch_size": 64,
        "random_seed": 42,
        "amplitude_range": [0.5, 1.5],
        "phase_range": [0.0, 6.283185307179586],
        "amplitude_noise_std": 0.1,
        "phase_noise_std": 0.1,
        "additive_noise_std": 0.02,
        "epochs": 3,
        "learning_rate": 0.001,
        "hidden_size": 32,
        "num_layers": 1,
    }


def test_default_config_matches_hardcoded_defaults() -> None:
    loaded = load_pipeline_defaults()

    assert loaded == DEFAULTS


def test_default_config_covers_every_pipeline_field() -> None:
    payload = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    expected = {field_info.name for field_info in fields(PipelineDefaults)}

    assert set(payload.keys()) == expected


def test_load_pipeline_defaults_from_custom_path(tmp_path: Path) -> None:
    payload = _baseline_payload()
    payload["num_samples"] = 200
    payload["batch_size"] = 16
    payload["epochs"] = 7
    config_path = tmp_path / "custom.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_pipeline_defaults(config_path)

    assert loaded.num_samples == 200
    assert loaded.batch_size == 16
    assert loaded.epochs == 7
    assert loaded.frequencies_hz == (1.0, 3.0, 5.0, 7.0)


def test_load_pipeline_defaults_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_pipeline_defaults(tmp_path / "does_not_exist.json")


def test_load_pipeline_defaults_rejects_unknown_field(tmp_path: Path) -> None:
    payload = _baseline_payload()
    payload["mystery_knob"] = 1.0
    config_path = tmp_path / "extra.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown config fields"):
        load_pipeline_defaults(config_path)


def test_load_pipeline_defaults_rejects_non_array_tuple_field(
    tmp_path: Path,
) -> None:
    payload = _baseline_payload()
    payload["amplitude_range"] = "0.5,1.5"
    config_path = tmp_path / "bad_tuple.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="amplitude_range"):
        load_pipeline_defaults(config_path)


def test_load_pipeline_defaults_validates_values(tmp_path: Path) -> None:
    payload = _baseline_payload()
    payload["learning_rate"] = -0.01
    config_path = tmp_path / "invalid_lr.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Learning rate"):
        load_pipeline_defaults(config_path)


def test_load_pipeline_defaults_rejects_non_object_root(tmp_path: Path) -> None:
    config_path = tmp_path / "list_root.json"
    config_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        load_pipeline_defaults(config_path)
