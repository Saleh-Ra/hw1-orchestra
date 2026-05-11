"""Small SDK entry points for the current reconstruction pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .baseline import FcBaselineResult
from .dataset import FullyConnectedSignalDataset
from .defaults import DEFAULTS, PipelineDefaults
from .model_entries import (
    plot_fc_predictions,
    plot_lstm_predictions,
    plot_rnn_predictions,
    train_fc_baseline,
    train_lstm_baseline,
    train_rnn_baseline,
)
from .signal_sets import SignalSet, generate_signal_sets

__all__ = [
    "MinimalFcPipelineResult",
    "build_fc_dataset",
    "plot_fc_predictions",
    "plot_lstm_predictions",
    "plot_rnn_predictions",
    "run_minimal_fc_pipeline",
    "run_signal_generation",
    "summarize_minimal_result",
    "train_fc_baseline",
    "train_lstm_baseline",
    "train_rnn_baseline",
]


@dataclass(frozen=True)
class MinimalFcPipelineResult:
    """Result returned by the minimal runnable FC pipeline."""

    baseline: FcBaselineResult
    plot_paths: list[Path]


def run_signal_generation(
    defaults: PipelineDefaults = DEFAULTS,
    signal_set_count: int | None = None,
    seed: int = DEFAULTS.random_seed,
) -> list[SignalSet]:
    """Generate the signal sets used by the FC baseline."""
    return generate_signal_sets(defaults, count=signal_set_count, seed=seed)


def build_fc_dataset(
    signal_sets: list[SignalSet],
    defaults: PipelineDefaults = DEFAULTS,
) -> FullyConnectedSignalDataset:
    """Build the lazy fully connected dataset for generated signal sets."""
    return FullyConnectedSignalDataset(signal_sets, defaults)


def run_minimal_fc_pipeline(
    defaults: PipelineDefaults | None = None,
    signal_set_count: int = 1,
    epochs: int = 5,
    output_dir: str | Path = "results/figures/fc_baseline",
    device: str = "cpu",
) -> MinimalFcPipelineResult:
    """Run the current minimal FC pipeline and save its plots."""
    run_defaults = defaults or PipelineDefaults(num_samples=100, batch_size=32)
    baseline = train_fc_baseline(
        defaults=run_defaults,
        signal_set_count=signal_set_count,
        epochs=epochs,
        prediction_count=len(run_defaults.frequencies_hz),
        seed=run_defaults.random_seed,
        device=device,
    )
    plot_paths = plot_fc_predictions(baseline, run_defaults, output_dir)
    return MinimalFcPipelineResult(baseline=baseline, plot_paths=plot_paths)


def summarize_minimal_result(result: MinimalFcPipelineResult) -> dict[str, Any]:
    """Return a compact serializable summary for scripts and notebooks."""
    return {
        "train_losses": result.baseline.train_losses,
        "test_losses": result.baseline.test_losses,
        "class_indices": result.baseline.class_indices,
        "plot_paths": [str(path) for path in result.plot_paths],
    }
