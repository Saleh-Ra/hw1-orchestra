"""Sequence-format dataset wrapper for time-step based models."""

from collections.abc import Sequence
from typing import Any

from .dataset import FullyConnectedSignalDataset
from .defaults import DEFAULTS, PipelineDefaults
from .signal_sets import SignalSet


class SequenceSignalDataset:
    """Lazy sequence dataset returning per-time-step noisy plus condition features."""

    def __init__(
        self,
        signal_sets: Sequence[SignalSet],
        defaults: PipelineDefaults = DEFAULTS,
    ) -> None:
        self._fc_dataset = FullyConnectedSignalDataset(signal_sets, defaults)
        self.defaults = defaults
        self.window_count = self._fc_dataset.window_count
        self.class_count = self._fc_dataset.class_count
        self.examples_per_set = self._fc_dataset.examples_per_set
        self.feature_size = 1 + self.class_count
        self.signal_sets = self._fc_dataset.signal_sets

    def __len__(self) -> int:
        return len(self._fc_dataset)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        fc_input, target = self._fc_dataset[index]

        import torch

        window_size = self.defaults.window_size
        noisy_window = fc_input[:window_size]
        condition = fc_input[window_size:]
        condition_per_step = condition.unsqueeze(0).expand(
            window_size,
            self.class_count,
        )
        sequence_input = torch.cat(
            [noisy_window.unsqueeze(1), condition_per_step],
            dim=1,
        )
        return sequence_input, target
