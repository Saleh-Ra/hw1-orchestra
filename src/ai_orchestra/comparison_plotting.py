"""Plotting helpers for FC/RNN/LSTM model comparison."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .comparison import ModelComparison  # noqa: E402


def plot_overall_mse(
    comparison: ModelComparison,
    output_path: str | Path,
) -> Path:
    """Save a bar chart of overall test MSE per model."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model_names = list(comparison.overall_mse.keys())
    mse_values = [comparison.overall_mse[name] for name in model_names]

    figure, axis = plt.subplots(figsize=(7, 4))
    bars = axis.bar(model_names, mse_values, color=["#4C78A8", "#F58518", "#54A24B"])
    for bar, value in zip(bars, mse_values, strict=False):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axis.set_title("Overall Test MSE by Model")
    axis.set_ylabel("MSE (lower is better)")
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)
    return output


def plot_per_frequency_mse(
    comparison: ModelComparison,
    output_path: str | Path,
) -> Path:
    """Save a grouped bar chart of test MSE per frequency per model."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model_names = list(comparison.per_frequency_mse.keys())
    frequencies = sorted({
        frequency
        for by_frequency in comparison.per_frequency_mse.values()
        for frequency in by_frequency
    })
    x_positions = np.arange(len(frequencies))
    bar_width = 0.8 / max(len(model_names), 1)
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    figure, axis = plt.subplots(figsize=(9, 4))
    for model_offset, model_name in enumerate(model_names):
        offset = (model_offset - (len(model_names) - 1) / 2) * bar_width
        values = [
            comparison.per_frequency_mse[model_name].get(frequency, 0.0)
            for frequency in frequencies
        ]
        axis.bar(
            x_positions + offset,
            values,
            bar_width,
            label=model_name,
            color=colors[model_offset % len(colors)],
        )
    axis.set_title("Per-Frequency Test MSE by Model")
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("MSE (lower is better)")
    axis.set_xticks(x_positions)
    axis.set_xticklabels([f"{frequency:g}" for frequency in frequencies])
    axis.legend()
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)
    return output
