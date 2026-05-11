"""Dataset helpers for conditioned signal reconstruction."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .defaults import DEFAULTS, PipelineDefaults
from .signal_sets import SignalSet


def make_one_hot(
    class_index: int,
    defaults: PipelineDefaults = DEFAULTS,
) -> NDArray[np.float64]:
    """Create a one-hot condition vector for a requested signal class."""
    defaults.validate()
    class_count = len(defaults.frequencies_hz)
    if class_index < 0 or class_index >= class_count:
        msg = f"Class index must be between 0 and {class_count - 1}."
        raise ValueError(msg)

    vector = np.zeros(class_count, dtype=np.float64)
    vector[class_index] = 1.0
    return vector


def valid_window_starts(defaults: PipelineDefaults = DEFAULTS) -> NDArray[np.int64]:
    """Return every valid inclusive start index for a sliding window."""
    defaults.validate()
    last_start = defaults.num_samples - defaults.window_size
    return np.arange(last_start + 1, dtype=np.int64)


def extract_window(
    signal: NDArray[np.float64],
    start: int,
    defaults: PipelineDefaults = DEFAULTS,
) -> NDArray[np.float64]:
    """Extract one fixed-size window from a 1D signal."""
    defaults.validate()
    signal_values = np.asarray(signal, dtype=np.float64)
    if signal_values.ndim != 1:
        msg = "Signal must be a 1D array."
        raise ValueError(msg)
    if len(signal_values) < defaults.window_size:
        msg = "Signal length must be at least the configured window size."
        raise ValueError(msg)

    last_start = len(signal_values) - defaults.window_size
    if start < 0 or start > last_start:
        msg = f"Window start must be between 0 and {last_start}."
        raise ValueError(msg)
    return signal_values[start : start + defaults.window_size]


def examples_per_signal_set(defaults: PipelineDefaults = DEFAULTS) -> int:
    """Return the number of FC examples produced by one signal set."""
    return len(valid_window_starts(defaults)) * len(defaults.frequencies_hz)


def total_examples(
    signal_set_count: int,
    defaults: PipelineDefaults = DEFAULTS,
) -> int:
    """Return the number of FC examples for multiple signal sets."""
    if signal_set_count <= 0:
        msg = "Signal set count must be positive."
        raise ValueError(msg)
    return signal_set_count * examples_per_signal_set(defaults)


def train_test_split_indices(
    item_count: int,
    train_ratio: float = DEFAULTS.train_ratio,
    seed: int = DEFAULTS.random_seed,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Create deterministic random train/test index splits."""
    if item_count <= 1:
        msg = "Item count must be greater than 1."
        raise ValueError(msg)
    if not 0.0 < train_ratio < 1.0:
        msg = "Train ratio must be between 0 and 1."
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(item_count).astype(np.int64)
    train_count = int(item_count * train_ratio)
    return indices[:train_count], indices[train_count:]


class FullyConnectedSignalDataset:
    """Lazy FC dataset returning conditioned noisy windows and clean targets."""

    def __init__(
        self,
        signal_sets: Sequence[SignalSet],
        defaults: PipelineDefaults = DEFAULTS,
    ) -> None:
        defaults.validate()
        if not signal_sets:
            msg = "At least one signal set is required."
            raise ValueError(msg)

        self.signal_sets = list(signal_sets)
        self.defaults = defaults
        self.window_count = len(valid_window_starts(defaults))
        self.class_count = len(defaults.frequencies_hz)
        self.examples_per_set = examples_per_signal_set(defaults)

    def __len__(self) -> int:
        return total_examples(len(self.signal_sets), self.defaults)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        if index < 0 or index >= len(self):
            msg = f"Dataset index must be between 0 and {len(self) - 1}."
            raise IndexError(msg)

        set_index, remainder = divmod(index, self.examples_per_set)
        window_index, class_index = divmod(remainder, self.class_count)
        signal_set = self.signal_sets[set_index]
        noisy_window = extract_window(signal_set.noisy_mix, window_index, self.defaults)
        target_window = extract_window(
            signal_set.clean_signals[class_index],
            window_index,
            self.defaults,
        )
        fc_input = np.concatenate(
            [noisy_window, make_one_hot(class_index, self.defaults)],
        )

        import torch

        return (
            torch.as_tensor(fc_input, dtype=torch.float32),
            torch.as_tensor(target_window, dtype=torch.float32),
        )
