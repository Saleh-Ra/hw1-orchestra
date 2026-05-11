"""Signal-generation helpers for the reconstruction pipeline."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .defaults import DEFAULTS, PipelineDefaults


def make_time_axis(defaults: PipelineDefaults = DEFAULTS) -> NDArray[np.float64]:
    """Create sample times from the sampling frequency and sample count."""
    defaults.validate()
    return np.arange(defaults.num_samples, dtype=np.float64) / defaults.sampling_frequency_hz


def generate_clean_signal(
    frequency_hz: float,
    time_axis: Sequence[float],
    amplitude: float,
    phase: float,
) -> NDArray[np.float64]:
    """Generate one clean sine wave for a frequency, amplitude, and phase."""
    if frequency_hz <= 0:
        msg = "Frequency must be positive."
        raise ValueError(msg)
    if amplitude <= 0:
        msg = "Amplitude must be positive."
        raise ValueError(msg)

    time_values = np.asarray(time_axis, dtype=np.float64)
    return amplitude * np.sin((2.0 * np.pi * frequency_hz * time_values) + phase)
