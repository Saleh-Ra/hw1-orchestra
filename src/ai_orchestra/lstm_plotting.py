"""Plotting helpers for LSTM baseline results."""

from pathlib import Path

from .lstm_baseline import LstmBaselineResult
from .sequence_plotting import plot_sequence_baseline_result


def plot_lstm_baseline_result(
    result: LstmBaselineResult,
    frequencies_hz: tuple[float, ...],
    output_dir: str | Path = "results/figures/lstm_baseline",
    model_name: str = "LSTM",
) -> list[Path]:
    """Save prediction examples and loss curves from an LSTM baseline result."""
    return plot_sequence_baseline_result(
        result=result,
        frequencies_hz=frequencies_hz,
        output_dir=output_dir,
        model_name=model_name,
        prediction_filename_prefix="lstm_prediction",
        loss_filename="lstm_loss_curve.png",
    )
