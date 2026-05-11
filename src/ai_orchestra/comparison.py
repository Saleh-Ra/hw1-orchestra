"""Train FC, RNN, and LSTM on the same data and compare their test metrics."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .comparison_internals import by_frequency, train_with_loss_history
from .dataloaders import create_data_loaders
from .dataset import FullyConnectedSignalDataset, train_test_split_indices
from .defaults import DEFAULTS, PipelineDefaults
from .evaluation import evaluate_per_class_mse
from .sequence_dataset import SequenceSignalDataset
from .signal_sets import generate_signal_sets

SplitIndices = tuple[NDArray[np.int64], NDArray[np.int64]]


@dataclass(frozen=True)
class ModelComparison:
    """Side-by-side test metrics for FC, RNN, and LSTM models."""

    settings: dict[str, Any]
    overall_mse: dict[str, float]
    per_frequency_mse: dict[str, dict[float, float]]
    train_losses: dict[str, list[float]] = field(default_factory=dict)
    test_losses: dict[str, list[float]] = field(default_factory=dict)


def compare_models(
    defaults: PipelineDefaults = DEFAULTS,
    signal_set_count: int | None = None,
    epochs: int = 3,
    learning_rate: float = 1e-3,
    hidden_size: int = 32,
    num_layers: int = 1,
    seed: int = DEFAULTS.random_seed,
    device: str | None = None,
    split_indices: SplitIndices | None = None,
    split_name: str = "random",
) -> ModelComparison:
    """Train FC, RNN, and LSTM on shared data and report side-by-side metrics."""
    defaults.validate()
    if epochs <= 0:
        msg = "Epoch count must be positive."
        raise ValueError(msg)
    if hidden_size <= 0:
        msg = "Hidden size must be positive."
        raise ValueError(msg)
    if num_layers <= 0:
        msg = "Layer count must be positive."
        raise ValueError(msg)

    import torch

    from .models import FullyConnectedSignalNet, LstmSignalNet, RnnSignalNet

    torch.manual_seed(seed)
    signal_sets = generate_signal_sets(defaults, count=signal_set_count, seed=seed)
    fc_dataset = FullyConnectedSignalDataset(signal_sets, defaults)
    seq_dataset = SequenceSignalDataset(signal_sets, defaults)
    if split_indices is None:
        train_indices, test_indices = train_test_split_indices(
            len(fc_dataset),
            train_ratio=defaults.train_ratio,
            seed=seed,
        )
    else:
        train_indices, test_indices = split_indices
    fc_train, fc_test = create_data_loaders(
        fc_dataset, train_indices, test_indices, defaults,
    )
    seq_train, seq_test = create_data_loaders(
        seq_dataset, train_indices, test_indices, defaults,
    )

    fc_model = FullyConnectedSignalNet()
    rnn_model = RnnSignalNet(
        input_size=seq_dataset.feature_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )
    lstm_model = LstmSignalNet(
        input_size=seq_dataset.feature_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )

    fc_train_losses, fc_test_losses = train_with_loss_history(
        fc_model, fc_train, fc_test, epochs, learning_rate, device,
    )
    rnn_train_losses, rnn_test_losses = train_with_loss_history(
        rnn_model, seq_train, seq_test, epochs, learning_rate, device,
    )
    lstm_train_losses, lstm_test_losses = train_with_loss_history(
        lstm_model, seq_train, seq_test, epochs, learning_rate, device,
    )

    fc_overall, fc_per_class = evaluate_per_class_mse(
        fc_model, fc_dataset, test_indices, defaults, device,
    )
    rnn_overall, rnn_per_class = evaluate_per_class_mse(
        rnn_model, seq_dataset, test_indices, defaults, device,
    )
    lstm_overall, lstm_per_class = evaluate_per_class_mse(
        lstm_model, seq_dataset, test_indices, defaults, device,
    )

    frequencies = defaults.frequencies_hz
    return ModelComparison(
        settings={
            "signal_set_count": len(signal_sets),
            "epochs": epochs,
            "learning_rate": learning_rate,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "batch_size": defaults.batch_size,
            "window_size": defaults.window_size,
            "train_ratio": defaults.train_ratio,
            "amplitude_noise_std": defaults.amplitude_noise_std,
            "phase_noise_std": defaults.phase_noise_std,
            "additive_noise_std": defaults.additive_noise_std,
            "seed": seed,
            "split": split_name,
        },
        overall_mse={
            "FC": fc_overall, "RNN": rnn_overall, "LSTM": lstm_overall,
        },
        per_frequency_mse={
            "FC": by_frequency(frequencies, fc_per_class),
            "RNN": by_frequency(frequencies, rnn_per_class),
            "LSTM": by_frequency(frequencies, lstm_per_class),
        },
        train_losses={
            "FC": fc_train_losses,
            "RNN": rnn_train_losses,
            "LSTM": lstm_train_losses,
        },
        test_losses={
            "FC": fc_test_losses,
            "RNN": rnn_test_losses,
            "LSTM": lstm_test_losses,
        },
    )
