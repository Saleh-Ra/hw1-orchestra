"""Dataset helpers for conditioned signal reconstruction."""

import numpy as np
from numpy.typing import NDArray

from .defaults import DEFAULTS, PipelineDefaults


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
