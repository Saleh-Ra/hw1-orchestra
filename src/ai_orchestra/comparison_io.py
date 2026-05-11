"""Serialization helpers for model comparison results."""

import csv
import json
from pathlib import Path
from typing import Any

from .comparison import ModelComparison


def save_comparison_json(
    comparison: ModelComparison,
    path: str | Path,
) -> Path:
    """Save a model comparison as a human-readable JSON file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "settings": comparison.settings,
        "overall_mse": comparison.overall_mse,
        "per_frequency_mse": {
            model_name: {
                f"{frequency_hz:g}": mse
                for frequency_hz, mse in by_frequency.items()
            }
            for model_name, by_frequency in comparison.per_frequency_mse.items()
        },
        "train_losses": comparison.train_losses,
        "test_losses": comparison.test_losses,
    }
    with output.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, sort_keys=True)
    return output


def save_comparison_csv(
    comparison: ModelComparison,
    path: str | Path,
) -> Path:
    """Save model overall and per-frequency MSE as a flat CSV."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frequencies = sorted({
        frequency
        for by_frequency in comparison.per_frequency_mse.values()
        for frequency in by_frequency
    })
    header = ["model", "overall_mse", *[f"mse_at_{freq:g}hz" for freq in frequencies]]
    with output.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(header)
        for model_name, overall in comparison.overall_mse.items():
            row: list[Any] = [model_name, overall]
            for frequency in frequencies:
                row.append(comparison.per_frequency_mse[model_name].get(frequency, ""))
            writer.writerow(row)
    return output
