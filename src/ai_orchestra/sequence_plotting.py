"""Shared plotting helper for sequence-format baseline results."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .fc_plotting import plot_loss_curves, plot_prediction_window
from .sequence_baseline import SequenceBaselineResult


def plot_sequence_baseline_result(
    result: SequenceBaselineResult,
    frequencies_hz: tuple[float, ...],
    output_dir: str | Path,
    model_name: str,
    prediction_filename_prefix: str,
    loss_filename: str,
) -> list[Path]:
    """Save prediction examples and loss curves from a sequence baseline run."""
    output_root = Path(output_dir)
    paths = [
        _plot_seq_prediction(
            prediction,
            target,
            class_index,
            frequencies_hz,
            output_root,
            model_name,
            prediction_filename_prefix,
        )
        for prediction, target, class_index in zip(
            result.predictions,
            result.targets,
            result.class_indices,
            strict=True,
        )
    ]
    paths.append(
        plot_loss_curves(
            result.train_losses,
            result.test_losses,
            output_root / loss_filename,
            model_name,
        ),
    )
    return paths


def _plot_seq_prediction(
    prediction: NDArray[np.float64],
    target: NDArray[np.float64],
    class_index: int,
    frequencies_hz: tuple[float, ...],
    output_root: Path,
    model_name: str,
    prediction_filename_prefix: str,
) -> Path:
    frequency = frequencies_hz[class_index]
    file_name = (
        f"{prediction_filename_prefix}_s{class_index + 1}_{frequency:g}hz.png"
    )
    return plot_prediction_window(
        prediction,
        target,
        frequency,
        output_root / file_name,
        model_name,
    )
