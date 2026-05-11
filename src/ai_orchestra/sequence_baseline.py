"""Shared runner for sequence-format baseline experiments (RNN, LSTM, ...)."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .dataloaders import create_data_loaders
from .dataset import train_test_split_indices
from .defaults import DEFAULTS, PipelineDefaults
from .prediction_sampling import sample_prediction_indices, sample_predictions
from .sequence_dataset import SequenceSignalDataset
from .signal_sets import generate_signal_sets


@dataclass(frozen=True)
class SequenceBaselineResult:
    """In-memory result from one sequence-format baseline run."""

    train_losses: list[float]
    test_losses: list[float]
    predictions: list[NDArray[np.float64]]
    targets: list[NDArray[np.float64]]
    class_indices: list[int]


def run_sequence_baseline(
    model_factory: Callable[[int], Any],
    defaults: PipelineDefaults = DEFAULTS,
    signal_set_count: int | None = None,
    epochs: int = 3,
    learning_rate: float = 1e-3,
    prediction_count: int = 4,
    seed: int = DEFAULTS.random_seed,
    device: str | None = None,
) -> SequenceBaselineResult:
    """Train any sequence-format model on the sequence dataset."""
    defaults.validate()
    if epochs <= 0:
        msg = "Epoch count must be positive."
        raise ValueError(msg)
    if prediction_count <= 0:
        msg = "Prediction count must be positive."
        raise ValueError(msg)

    import torch

    from .training import create_adam_optimizer, evaluate_one_epoch, train_one_epoch

    torch.manual_seed(seed)
    signal_sets = generate_signal_sets(defaults, count=signal_set_count, seed=seed)
    dataset = SequenceSignalDataset(signal_sets, defaults)
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
    model = model_factory(dataset.feature_size)
    optimizer = create_adam_optimizer(model, learning_rate)

    train_losses: list[float] = []
    test_losses: list[float] = []
    for _ in range(epochs):
        train_losses.append(train_one_epoch(model, train_loader, optimizer, device))
        test_losses.append(evaluate_one_epoch(model, test_loader, device))

    prediction_indices = sample_prediction_indices(
        test_indices,
        dataset.class_count,
        prediction_count,
    )
    predictions, targets, class_indices = sample_predictions(
        model,
        dataset,
        prediction_indices,
        device,
    )
    return SequenceBaselineResult(
        train_losses=train_losses,
        test_losses=test_losses,
        predictions=predictions,
        targets=targets,
        class_indices=class_indices,
    )
