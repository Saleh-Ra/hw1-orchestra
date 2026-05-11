"""Small SDK entry points for the current reconstruction pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .baseline import FcBaselineResult, run_fc_baseline
from .dataset import FullyConnectedSignalDataset
from .defaults import DEFAULTS, PipelineDefaults
from .fc_plotting import plot_fc_baseline_result
from .signal_sets import SignalSet, generate_signal_sets


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


def train_fc_baseline(
    defaults: PipelineDefaults = DEFAULTS,
    signal_set_count: int | None = None,
    epochs: int = 3,
    learning_rate: float = 1e-3,
    prediction_count: int = 4,
    seed: int = DEFAULTS.random_seed,
    device: str | None = None,
) -> FcBaselineResult:
    """Train the current FC baseline through the package entry point."""
    return run_fc_baseline(
        defaults=defaults,
        signal_set_count=signal_set_count,
        epochs=epochs,
        learning_rate=learning_rate,
        prediction_count=prediction_count,
        seed=seed,
        device=device,
    )


def plot_fc_predictions(
    baseline: FcBaselineResult,
    defaults: PipelineDefaults = DEFAULTS,
    output_dir: str | Path = "results/figures/fc_baseline",
    model_name: str = "FC",
) -> list[Path]:
    """Save FC prediction and loss plots for a baseline result."""
    return plot_fc_baseline_result(
        baseline,
        defaults.frequencies_hz,
        output_dir,
        model_name,
    )


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
