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


def sample_signal_parameters(
    defaults: PipelineDefaults = DEFAULTS,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sample one amplitude and phase for each configured signal."""
    defaults.validate()
    rng = np.random.default_rng(seed)
    signal_count = len(defaults.frequencies_hz)
    amplitudes = rng.uniform(*defaults.amplitude_range, size=signal_count)
    phases = rng.uniform(*defaults.phase_range, size=signal_count)
    return amplitudes, phases


def generate_clean_signals(
    defaults: PipelineDefaults = DEFAULTS,
    amplitudes: Sequence[float] | None = None,
    phases: Sequence[float] | None = None,
) -> NDArray[np.float64]:
    """Generate the clean signal stack in the configured frequency order."""
    defaults.validate()
    time_axis = make_time_axis(defaults)
    frequencies = defaults.frequencies_hz
    signal_count = len(frequencies)
    signal_amplitudes = amplitudes if amplitudes is not None else (1.0,) * signal_count
    signal_phases = phases if phases is not None else (0.0,) * signal_count

    if len(signal_amplitudes) != signal_count or len(signal_phases) != signal_count:
        msg = "Amplitudes and phases must match the number of frequencies."
        raise ValueError(msg)

    signals = [
        generate_clean_signal(frequency, time_axis, amplitude, phase)
        for frequency, amplitude, phase in zip(
            frequencies,
            signal_amplitudes,
            signal_phases,
            strict=True,
        )
    ]
    return np.stack(signals)
