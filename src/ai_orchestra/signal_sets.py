"""Complete signal-set representation and generation."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .defaults import DEFAULTS, PipelineDefaults
from .signals import (
    combine_signals,
    generate_clean_signals,
    generate_noisy_signals,
    make_time_axis,
    sample_signal_parameters,
)


@dataclass(frozen=True)
class SignalSet:
    """Clean/noisy signals and mixes for one generated training source."""

    time_axis: NDArray[np.float64]
    frequencies_hz: tuple[float, ...]
    amplitudes: NDArray[np.float64]
    phases: NDArray[np.float64]
    clean_signals: NDArray[np.float64]
    noisy_signals: NDArray[np.float64]
    clean_mix: NDArray[np.float64]
    noisy_mix: NDArray[np.float64]


def generate_signal_set(
    defaults: PipelineDefaults = DEFAULTS,
    seed: int | None = None,
) -> SignalSet:
    """Generate one complete clean/noisy signal set."""
    defaults.validate()
    amplitudes, phases = sample_signal_parameters(defaults, seed)
    clean_signals = generate_clean_signals(
        defaults=defaults,
        amplitudes=amplitudes,
        phases=phases,
    )
    noisy_signals = generate_noisy_signals(
        defaults=defaults,
        amplitudes=amplitudes,
        phases=phases,
        seed=seed,
    )

    return SignalSet(
        time_axis=make_time_axis(defaults),
        frequencies_hz=defaults.frequencies_hz,
        amplitudes=amplitudes,
        phases=phases,
        clean_signals=clean_signals,
        noisy_signals=noisy_signals,
        clean_mix=combine_signals(clean_signals),
        noisy_mix=combine_signals(noisy_signals),
    )
