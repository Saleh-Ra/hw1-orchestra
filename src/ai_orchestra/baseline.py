"""First fully connected baseline experiment."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .dataloaders import create_data_loaders
from .dataset import FullyConnectedSignalDataset, train_test_split_indices
from .defaults import DEFAULTS, PipelineDefaults
from .signal_sets import generate_signal_sets


@dataclass(frozen=True)
class FcBaselineResult:
    """In-memory result from the first FC baseline run."""

    train_losses: list[float]
    test_losses: list[float]
    predictions: list[NDArray[np.float64]]
    targets: list[NDArray[np.float64]]
    class_indices: list[int]


def run_fc_baseline(
    defaults: PipelineDefaults = DEFAULTS,
    signal_set_count: int | None = None,
    epochs: int = 3,
    learning_rate: float = 1e-3,
    prediction_count: int = 4,
    seed: int = DEFAULTS.random_seed,
    device: str | None = None,
) -> FcBaselineResult:
    """Train the first FC baseline and return losses plus sample predictions."""
    defaults.validate()
    if epochs <= 0:
        msg = "Epoch count must be positive."
        raise ValueError(msg)
    if prediction_count <= 0:
        msg = "Prediction count must be positive."
        raise ValueError(msg)

    import torch

    from .models import FullyConnectedSignalNet
    from .training import create_adam_optimizer, evaluate_one_epoch, train_one_epoch

    torch.manual_seed(seed)
    signal_sets = generate_signal_sets(defaults, count=signal_set_count, seed=seed)
    dataset = FullyConnectedSignalDataset(signal_sets, defaults)
    train_indices, test_indices = train_test_split_indices(
        len(dataset),
        train_ratio=defaults.train_ratio,
        seed=seed,
    )
    train_loader, test_loader = create_data_loaders(
        dataset,
        train_indices,
        test_indices,
        defaults,
    )
    model = FullyConnectedSignalNet()
    optimizer = create_adam_optimizer(model, learning_rate)

    train_losses: list[float] = []
    test_losses: list[float] = []
    for _ in range(epochs):
        train_losses.append(train_one_epoch(model, train_loader, optimizer, device))
        test_losses.append(evaluate_one_epoch(model, test_loader, device))

    prediction_indices = _sample_prediction_indices(
        test_indices,
        dataset.class_count,
        prediction_count,
    )
    predictions, targets, class_indices = _sample_predictions(
        model,
        dataset,
        prediction_indices,
        device,
    )
    return FcBaselineResult(
        train_losses=train_losses,
        test_losses=test_losses,
        predictions=predictions,
        targets=targets,
        class_indices=class_indices,
    )


def _sample_predictions(
    model,
    dataset: FullyConnectedSignalDataset,
    indices: NDArray[np.int64],
    device: str | None,
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]], list[int]]:
    import torch

    prediction_device = _resolve_device(device)
    model.to(prediction_device)
    model.eval()
    predictions: list[NDArray[np.float64]] = []
    targets: list[NDArray[np.float64]] = []
    class_indices: list[int] = []

    with torch.no_grad():
        for index in indices:
            fc_input, target = dataset[int(index)]
            prediction = model(fc_input.unsqueeze(0).to(prediction_device)).squeeze(0)
            predictions.append(prediction.cpu().numpy().astype(np.float64))
            targets.append(target.numpy().astype(np.float64))
            class_indices.append(int(index % dataset.class_count))

    return predictions, targets, class_indices


def _sample_prediction_indices(
    indices: NDArray[np.int64],
    class_count: int,
    prediction_count: int,
) -> NDArray[np.int64]:
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


def _resolve_device(device: str | None) -> str:
    if device is not None:
        return device

    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
