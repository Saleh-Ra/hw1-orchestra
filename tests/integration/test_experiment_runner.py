import json
from pathlib import Path

import pytest


def _make_tiny_experiments_file(path: Path) -> Path:
    payload = {
        "experiments": [
            {
                "name": "tiny_hidden",
                "description": "Smaller hidden size for fast tests.",
                "overrides": {"hidden_size": 8},
            }
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_run_experiment_writes_all_expected_artifacts(tmp_path: Path) -> None:
    pytest.importorskip("torch")

    from ai_orchestra import PipelineDefaults
    from ai_orchestra.experiment_runner import run_experiment

    experiments_path = _make_tiny_experiments_file(tmp_path / "experiments.json")
    base = PipelineDefaults(num_samples=100, batch_size=16, epochs=1)

    summary = run_experiment(
        "tiny_hidden",
        base_defaults=base,
        experiments_path=experiments_path,
        results_dir=tmp_path / "results",
        signal_set_count=1,
        device="cpu",
    )

    assert summary["name"] == "tiny_hidden"
    assert summary["output_dir"].is_dir()
    for key in (
        "config_path",
        "metrics_json",
        "metrics_csv",
        "overall_plot",
        "per_frequency_plot",
    ):
        assert summary[key].exists(), f"missing artifact: {key}"
    assert set(summary["overall_mse"].keys()) == {"FC", "RNN", "LSTM"}

    saved_config = json.loads(summary["config_path"].read_text(encoding="utf-8"))
    assert saved_config["name"] == "tiny_hidden"
    assert saved_config["overrides"] == {"hidden_size": 8}
    assert saved_config["resolved_defaults"]["hidden_size"] == 8
    assert saved_config["resolved_defaults"]["num_samples"] == 100


def test_run_experiment_rejects_unknown_name(tmp_path: Path) -> None:
    pytest.importorskip("torch")

    from ai_orchestra.experiment_runner import run_experiment

    experiments_path = _make_tiny_experiments_file(tmp_path / "experiments.json")

    with pytest.raises(KeyError, match="Unknown experiment"):
        run_experiment(
            "does_not_exist",
            experiments_path=experiments_path,
            results_dir=tmp_path / "results",
        )


def test_list_known_experiments_returns_sorted_names(tmp_path: Path) -> None:
    from ai_orchestra.experiment_runner import list_known_experiments

    path = tmp_path / "experiments.json"
    payload = {
        "experiments": [
            {"name": "b", "description": "", "overrides": {"batch_size": 8}},
            {"name": "a", "description": "", "overrides": {"batch_size": 16}},
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    names = list_known_experiments(path)

    assert names == ["a", "b"]
