"""Plotting helpers for raw generated signals."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .signal_sets import SignalSet


def plot_signal_stack(
    time_axis: NDArray[np.float64],
    signals: NDArray[np.float64],
    frequencies_hz: tuple[float, ...],
    output_path: str | Path,
    title: str,
    max_samples: int = 1000,
) -> Path:
    """Save a readable plot for individual clean or noisy signal rows."""
    output = _prepare_output_path(output_path)
    sample_count = min(max_samples, len(time_axis))

    fig, axis = plt.subplots(figsize=(10, 5))
    for index, frequency in enumerate(frequencies_hz):
        axis.plot(
            time_axis[:sample_count],
            signals[index, :sample_count],
            label=f"{frequency:g} Hz",
        )
    axis.set_title(title)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Amplitude")
    axis.legend()
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_mixed_signal(
    time_axis: NDArray[np.float64],
    mixed_signal: NDArray[np.float64],
    output_path: str | Path,
    title: str,
    max_samples: int = 1000,
) -> Path:
    """Save a readable plot for one mixed signal."""
    output = _prepare_output_path(output_path)
    sample_count = min(max_samples, len(time_axis))

    fig, axis = plt.subplots(figsize=(10, 4))
    axis.plot(time_axis[:sample_count], mixed_signal[:sample_count])
    axis.set_title(title)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Amplitude")
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_raw_signal_examples(
    signal_set: SignalSet,
    output_dir: str | Path,
) -> list[Path]:
    """Save clean, noisy, clean-mix, and noisy-mix raw signal examples."""
    output_root = Path(output_dir)
    return [
        plot_signal_stack(
            signal_set.time_axis,
            signal_set.clean_signals,
            signal_set.frequencies_hz,
            output_root / "clean_signals.png",
            "Clean Individual Signals",
        ),
        plot_signal_stack(
            signal_set.time_axis,
            signal_set.noisy_signals,
            signal_set.frequencies_hz,
            output_root / "noisy_signals.png",
            "Noisy Individual Signals",
        ),
        plot_mixed_signal(
            signal_set.time_axis,
            signal_set.clean_mix,
            output_root / "clean_mixed_signal.png",
            "Clean Mixed Signal",
        ),
        plot_mixed_signal(
            signal_set.time_axis,
            signal_set.noisy_mix,
            output_root / "noisy_mixed_signal.png",
            "Noisy Mixed Signal",
        ),
    ]


def _prepare_output_path(output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output
