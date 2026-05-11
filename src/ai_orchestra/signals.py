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


def generate_noisy_signals(
    defaults: PipelineDefaults = DEFAULTS,
    amplitudes: Sequence[float] | None = None,
    phases: Sequence[float] | None = None,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Generate noisy signal versions from the same base frequencies."""
    defaults.validate()
    rng = np.random.default_rng(seed)
    signal_count = len(defaults.frequencies_hz)
    signal_amplitudes = amplitudes
    signal_phases = phases

    if signal_amplitudes is None or signal_phases is None:
        signal_amplitudes, signal_phases = sample_signal_parameters(defaults, seed)

    if len(signal_amplitudes) != signal_count or len(signal_phases) != signal_count:
        msg = "Amplitudes and phases must match the number of frequencies."
        raise ValueError(msg)

    amplitude_noise = rng.normal(0.0, defaults.amplitude_noise_std, size=signal_count)
    phase_noise = rng.normal(0.0, defaults.phase_noise_std, size=signal_count)
    noisy_amplitudes = np.maximum(
        np.asarray(signal_amplitudes, dtype=np.float64) + amplitude_noise,
        np.finfo(np.float64).eps,
    )
    noisy_phases = np.asarray(signal_phases, dtype=np.float64) + phase_noise
    noisy_signals = generate_clean_signals(
        defaults=defaults,
        amplitudes=noisy_amplitudes,
        phases=noisy_phases,
    )

    if defaults.additive_noise_std > 0:
        additive_noise = rng.normal(
            0.0,
            defaults.additive_noise_std,
            size=noisy_signals.shape,
        )
        noisy_signals = noisy_signals + additive_noise

    return noisy_signals
