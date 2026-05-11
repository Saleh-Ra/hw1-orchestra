"""Evaluation metrics for trained reconstruction models."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .defaults import DEFAULTS, PipelineDefaults


def compute_mse(
    predictions: NDArray[np.float64],
    targets: NDArray[np.float64],
) -> float:
    """Return mean squared error between predictions and targets."""
    prediction_array = np.asarray(predictions, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64)
    if prediction_array.shape != target_array.shape:
        msg = "Predictions and targets must have the same shape."
        raise ValueError(msg)
    if prediction_array.size == 0:
        msg = "Predictions and targets must contain at least one value."
        raise ValueError(msg)
    difference = prediction_array - target_array
    return float(np.mean(difference * difference))


def compute_mae(
    predictions: NDArray[np.float64],
    targets: NDArray[np.float64],
) -> float:
    """Return mean absolute error between predictions and targets."""
    prediction_array = np.asarray(predictions, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64)
    if prediction_array.shape != target_array.shape:
        msg = "Predictions and targets must have the same shape."
        raise ValueError(msg)
    if prediction_array.size == 0:
        msg = "Predictions and targets must contain at least one value."
        raise ValueError(msg)
    return float(np.mean(np.abs(prediction_array - target_array)))


def evaluate_per_class_mse(
    model: Any,
    dataset: Any,
    test_indices: NDArray[np.int64],
    defaults: PipelineDefaults = DEFAULTS,
    device: str | None = None,
) -> tuple[float, dict[int, float]]:
    """Compute overall test MSE and per-class test MSE for a trained model.

    Class index is recovered from ``index % class_count`` because the dataset
    cycles through classes in its inner loop. This avoids reading dataset
    items just to learn their class.
    """
    from torch.utils.data import DataLoader, Subset

    from .training import evaluate_one_epoch

    class_count = dataset.class_count
    per_class_mse: dict[int, float] = {}
    weighted_squared_sum = 0.0
    total_examples_seen = 0

    for class_index in range(class_count):
        class_test_indices = [
            int(idx) for idx in test_indices if int(idx) % class_count == class_index
        ]
        if not class_test_indices:
            continue
        loader = DataLoader(
            Subset(dataset, class_test_indices),
            batch_size=defaults.batch_size,
            shuffle=False,
        )
        class_mse = evaluate_one_epoch(model, loader, device)
        per_class_mse[class_index] = class_mse
        weighted_squared_sum += class_mse * len(class_test_indices)
        total_examples_seen += len(class_test_indices)

    if total_examples_seen == 0:
        msg = "No test indices were provided."
        raise ValueError(msg)
    overall_mse = weighted_squared_sum / total_examples_seen
    return overall_mse, per_class_mse
