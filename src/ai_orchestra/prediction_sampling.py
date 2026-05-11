"""Shared helpers for device resolution and prediction sampling."""

from typing import Any

import numpy as np
from numpy.typing import NDArray


def resolve_device(device: str | None) -> str:
    """Pick a runtime device, defaulting to CUDA when available."""
    if device is not None:
        return device

    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def sample_prediction_indices(
    indices: NDArray[np.int64],
    class_count: int,
    prediction_count: int,
) -> NDArray[np.int64]:
    """Pick test indices that cover each class before adding extras."""
    selected: list[int] = []
    seen_classes: set[int] = set()
    for index in indices:
        class_index = int(index % class_count)
        if class_index not in seen_classes:
            selected.append(int(index))
            seen_classes.add(class_index)
        if len(selected) == min(prediction_count, class_count):
            break

    if len(selected) < prediction_count:
        selected_set = set(selected)
        for index in indices:
            if int(index) not in selected_set:
                selected.append(int(index))
            if len(selected) == prediction_count:
                break

    return np.asarray(selected, dtype=np.int64)


def sample_predictions(
    model: Any,
    dataset: Any,
    indices: NDArray[np.int64],
    device: str | None,
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]], list[int]]:
    """Run the model on selected dataset items and collect numpy predictions."""
    import torch

    prediction_device = resolve_device(device)
    model.to(prediction_device)
    model.eval()
    predictions: list[NDArray[np.float64]] = []
    targets: list[NDArray[np.float64]] = []
    class_indices: list[int] = []

    with torch.no_grad():
        for index in indices:
            model_input, target = dataset[int(index)]
            prediction = model(
                model_input.unsqueeze(0).to(prediction_device),
            ).squeeze(0)
            predictions.append(prediction.cpu().numpy().astype(np.float64))
            targets.append(target.numpy().astype(np.float64))
            class_indices.append(int(index % dataset.class_count))

    return predictions, targets, class_indices
