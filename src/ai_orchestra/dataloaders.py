"""PyTorch DataLoader helpers for reconstruction datasets."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .defaults import DEFAULTS, PipelineDefaults


def create_data_loaders(
    dataset: Any,
    train_indices: NDArray[np.int64],
    test_indices: NDArray[np.int64],
    defaults: PipelineDefaults = DEFAULTS,
    batch_size: int | None = None,
) -> tuple[Any, Any]:
    """Create train and test DataLoaders from index splits."""
    defaults.validate()
    loader_batch_size = defaults.batch_size if batch_size is None else batch_size
    if loader_batch_size <= 0:
        msg = "Batch size must be positive."
        raise ValueError(msg)

    from torch.utils.data import DataLoader, Subset

    train_subset = Subset(dataset, train_indices.tolist())
    test_subset = Subset(dataset, test_indices.tolist())
    return (
        DataLoader(train_subset, batch_size=loader_batch_size, shuffle=True),
        DataLoader(test_subset, batch_size=loader_batch_size, shuffle=False),
    )
