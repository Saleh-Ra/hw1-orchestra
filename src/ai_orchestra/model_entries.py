"""Thin per-model SDK wrappers for training and plotting baselines."""

from pathlib import Path

from .baseline import FcBaselineResult, run_fc_baseline
from .defaults import DEFAULTS, PipelineDefaults
from .fc_plotting import plot_fc_baseline_result
from .lstm_baseline import LstmBaselineResult, run_lstm_baseline
from .lstm_plotting import plot_lstm_baseline_result
from .rnn_baseline import RnnBaselineResult, run_rnn_baseline
from .rnn_plotting import plot_rnn_baseline_result


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


def train_rnn_baseline(
    defaults: PipelineDefaults = DEFAULTS,
    signal_set_count: int | None = None,
    epochs: int = 3,
    learning_rate: float = 1e-3,
    hidden_size: int = 32,
    prediction_count: int = 4,
    seed: int = DEFAULTS.random_seed,
    device: str | None = None,
) -> RnnBaselineResult:
    """Train the RNN baseline through the package entry point."""
    return run_rnn_baseline(
        defaults=defaults,
        signal_set_count=signal_set_count,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        prediction_count=prediction_count,
        seed=seed,
        device=device,
    )


def plot_rnn_predictions(
    baseline: RnnBaselineResult,
    defaults: PipelineDefaults = DEFAULTS,
    output_dir: str | Path = "results/figures/rnn_baseline",
    model_name: str = "RNN",
) -> list[Path]:
    """Save RNN prediction and loss plots for a baseline result."""
    return plot_rnn_baseline_result(
        baseline,
        defaults.frequencies_hz,
        output_dir,
        model_name,
    )


def train_lstm_baseline(
    defaults: PipelineDefaults = DEFAULTS,
    signal_set_count: int | None = None,
    epochs: int = 3,
    learning_rate: float = 1e-3,
    hidden_size: int = 32,
    prediction_count: int = 4,
    seed: int = DEFAULTS.random_seed,
    device: str | None = None,
) -> LstmBaselineResult:
    """Train the LSTM baseline through the package entry point."""
    return run_lstm_baseline(
        defaults=defaults,
        signal_set_count=signal_set_count,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        prediction_count=prediction_count,
        seed=seed,
        device=device,
    )


def plot_lstm_predictions(
    baseline: LstmBaselineResult,
    defaults: PipelineDefaults = DEFAULTS,
    output_dir: str | Path = "results/figures/lstm_baseline",
    model_name: str = "LSTM",
) -> list[Path]:
    """Save LSTM prediction and loss plots for a baseline result."""
    return plot_lstm_baseline_result(
        baseline,
        defaults.frequencies_hz,
        output_dir,
        model_name,
    )
