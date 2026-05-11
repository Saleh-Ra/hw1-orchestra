"""Plotting helpers for fully connected baseline results."""

from pathlib import Path

import matplotlib
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .baseline import FcBaselineResult


def plot_prediction_window(
    prediction: NDArray[np.float64],
    target: NDArray[np.float64],
    frequency_hz: float,
    output_path: str | Path,
    model_name: str = "FC",
) -> Path:
    """Save one target-vs-prediction window plot."""
    prediction_values, target_values = _validate_prediction_pair(prediction, target)
    output = _prepare_output_path(output_path)
    sample_indices = np.arange(len(target_values))

    fig, axis = plt.subplots(figsize=(8, 4))
    axis.plot(sample_indices, target_values, label="Target")
    axis.plot(sample_indices, prediction_values, label="Prediction")
    axis.set_title(f"{model_name} Prediction - {frequency_hz:g} Hz")
    axis.set_xlabel("Window sample")
    axis.set_ylabel("Amplitude")
    axis.legend()
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_loss_curves(
    train_losses: list[float],
    test_losses: list[float],
    output_path: str | Path,
    model_name: str = "FC",
) -> Path:
    """Save train and test loss curves for a baseline run."""
    if not train_losses or not test_losses:
        msg = "Train and test losses must not be empty."
        raise ValueError(msg)

    output = _prepare_output_path(output_path)
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.plot(range(1, len(train_losses) + 1), train_losses, label="Train")
    axis.plot(range(1, len(test_losses) + 1), test_losses, label="Test")
    axis.set_title(f"{model_name} Loss Curves")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE loss")
    axis.legend()
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_fc_baseline_result(
    result: FcBaselineResult,
    frequencies_hz: tuple[float, ...],
    output_dir: str | Path = "results/figures/fc_baseline",
    model_name: str = "FC",
) -> list[Path]:
    """Save prediction examples and loss curves from an FC baseline result."""
    output_root = Path(output_dir)
    paths = [
        _plot_result_prediction(
            prediction,
            target,
            class_index,
            frequencies_hz,
            output_root,
            model_name,
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
            output_root / "fc_loss_curve.png",
            model_name,
        ),
    )
    return paths


def _plot_result_prediction(
    prediction: NDArray[np.float64],
    target: NDArray[np.float64],
    class_index: int,
    frequencies_hz: tuple[float, ...],
    output_root: Path,
    model_name: str,
) -> Path:
    frequency = frequencies_hz[class_index]
    output_path = output_root / f"fc_prediction_s{class_index + 1}_{frequency:g}hz.png"
    return plot_prediction_window(
        prediction,
        target,
        frequency,
        output_path,
        model_name,
    )


def _validate_prediction_pair(
    prediction: NDArray[np.float64],
    target: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    prediction_values = np.asarray(prediction, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    if prediction_values.ndim != 1 or target_values.ndim != 1:
        msg = "Prediction and target must be 1D arrays."
        raise ValueError(msg)
    if prediction_values.shape != target_values.shape:
        msg = "Prediction and target must have the same shape."
        raise ValueError(msg)
    return prediction_values, target_values


def _prepare_output_path(output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output
