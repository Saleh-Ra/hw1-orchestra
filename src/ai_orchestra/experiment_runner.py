"""Run a single named experiment variant and save its artifacts to disk."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .comparison import compare_models
from .comparison_io import save_comparison_csv, save_comparison_json
from .comparison_plotting import plot_overall_mse, plot_per_frequency_mse
from .defaults import DEFAULTS, PipelineDefaults
from .experiments import (
    DEFAULT_EXPERIMENTS_PATH,
    ExperimentSpec,
    apply_overrides,
    load_experiments,
)

DEFAULT_EXPERIMENT_RESULTS_DIR: Path = Path("results/experiments")


def run_experiment(
    name: str,
    base_defaults: PipelineDefaults = DEFAULTS,
    experiments_path: str | Path | None = None,
    results_dir: str | Path = DEFAULT_EXPERIMENT_RESULTS_DIR,
    signal_set_count: int = 1,
    device: str | None = None,
) -> dict[str, Any]:
    """Run experiment ``name`` end-to-end and write artifacts under ``results_dir``.

    Returns a small summary dict with the resolved output paths and the
    overall test MSE per model.
    """
    experiments = load_experiments(experiments_path)
    if name not in experiments:
        msg = (
            f"Unknown experiment '{name}'. "
            f"Known names: {sorted(experiments)}"
        )
        raise KeyError(msg)
    spec = experiments[name]
    merged = apply_overrides(base_defaults, spec.overrides)

    comparison = compare_models(
        defaults=merged,
        signal_set_count=signal_set_count,
        epochs=merged.epochs,
        learning_rate=merged.learning_rate,
        hidden_size=merged.hidden_size,
        num_layers=merged.num_layers,
        seed=merged.random_seed,
        device=device,
    )

    experiment_dir = Path(results_dir) / name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    _write_resolved_config(experiment_dir, spec, merged)
    metrics_json = save_comparison_json(comparison, experiment_dir / "metrics.json")
    metrics_csv = save_comparison_csv(comparison, experiment_dir / "metrics.csv")
    overall_plot = plot_overall_mse(
        comparison, experiment_dir / "overall_test_mse.png",
    )
    per_frequency_plot = plot_per_frequency_mse(
        comparison, experiment_dir / "per_frequency_test_mse.png",
    )

    return {
        "name": name,
        "output_dir": experiment_dir,
        "config_path": experiment_dir / "config.json",
        "metrics_json": metrics_json,
        "metrics_csv": metrics_csv,
        "overall_plot": overall_plot,
        "per_frequency_plot": per_frequency_plot,
        "overall_mse": comparison.overall_mse,
    }


def list_known_experiments(
    experiments_path: str | Path | None = None,
) -> list[str]:
    """Return the sorted list of experiment names from the given file."""
    return sorted(load_experiments(experiments_path))


def _write_resolved_config(
    experiment_dir: Path,
    spec: ExperimentSpec,
    merged: PipelineDefaults,
) -> Path:
    """Save the merged config so each result is reproducible from disk."""
    resolved_path = experiment_dir / "config.json"
    payload = {
        "name": spec.name,
        "description": spec.description,
        "overrides": spec.overrides,
        "experiments_source": str(DEFAULT_EXPERIMENTS_PATH.name),
        "resolved_defaults": _serialize_defaults(merged),
    }
    resolved_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return resolved_path


def _serialize_defaults(defaults: PipelineDefaults) -> dict[str, Any]:
    """Convert ``PipelineDefaults`` into JSON-serializable primitives."""
    payload = asdict(defaults)
    for field_name, value in payload.items():
        if isinstance(value, tuple):
            payload[field_name] = list(value)
    return payload
