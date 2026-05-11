"""Plotting helpers for RNN baseline results."""

from pathlib import Path

from .rnn_baseline import RnnBaselineResult
from .sequence_plotting import plot_sequence_baseline_result


def plot_rnn_baseline_result(
    result: RnnBaselineResult,
    frequencies_hz: tuple[float, ...],
    output_dir: str | Path = "results/figures/rnn_baseline",
    model_name: str = "RNN",
) -> list[Path]:
    """Save prediction examples and loss curves from an RNN baseline result."""
    return plot_sequence_baseline_result(
        result=result,
        frequencies_hz=frequencies_hz,
        output_dir=output_dir,
        model_name=model_name,
        prediction_filename_prefix="rnn_prediction",
        loss_filename="rnn_loss_curve.png",
    )
