import numpy as np

from ai_orchestra import DEFAULTS, PipelineDefaults
from ai_orchestra.signals import (
    generate_clean_signals,
    generate_noisy_signals,
    sample_signal_parameters,
)


def test_noisy_signal_shape_matches_clean_signal_shape() -> None:
    amplitudes, phases = sample_signal_parameters(seed=DEFAULTS.random_seed)
    clean_signals = generate_clean_signals(amplitudes=amplitudes, phases=phases)
    noisy_signals = generate_noisy_signals(
        amplitudes=amplitudes,
        phases=phases,
        seed=DEFAULTS.random_seed,
    )

    assert noisy_signals.shape == clean_signals.shape


def test_noisy_signal_shape_matches_default_stack() -> None:
    noisy_signals = generate_noisy_signals(seed=DEFAULTS.random_seed)

    assert noisy_signals.shape == (len(DEFAULTS.frequencies_hz), DEFAULTS.num_samples)


def test_noisy_signal_values_are_finite() -> None:
    noisy_signals = generate_noisy_signals(seed=DEFAULTS.random_seed)

    assert np.isfinite(noisy_signals).all()


def test_noisy_signals_differ_from_clean_signals() -> None:
    amplitudes, phases = sample_signal_parameters(seed=DEFAULTS.random_seed)
    clean_signals = generate_clean_signals(amplitudes=amplitudes, phases=phases)
    noisy_signals = generate_noisy_signals(
        amplitudes=amplitudes,
        phases=phases,
        seed=DEFAULTS.random_seed,
    )

    assert not np.allclose(clean_signals, noisy_signals)


def test_noisy_generation_is_reproducible_with_seed() -> None:
    first = generate_noisy_signals(seed=DEFAULTS.random_seed)
    second = generate_noisy_signals(seed=DEFAULTS.random_seed)

    assert np.allclose(first, second)


def test_stronger_noise_moves_farther_from_clean_signal() -> None:
    amplitudes, phases = sample_signal_parameters(seed=DEFAULTS.random_seed)
    clean_signals = generate_clean_signals(amplitudes=amplitudes, phases=phases)
    low_noise = PipelineDefaults(
        amplitude_noise_std=0.01,
        phase_noise_std=0.01,
        additive_noise_std=0.0,
    )
    high_noise = PipelineDefaults(
        amplitude_noise_std=0.50,
        phase_noise_std=0.50,
        additive_noise_std=0.0,
    )

    low_noisy = generate_noisy_signals(low_noise, amplitudes, phases, seed=1)
    high_noisy = generate_noisy_signals(high_noise, amplitudes, phases, seed=1)
    low_difference = np.mean(np.abs(low_noisy - clean_signals))
    high_difference = np.mean(np.abs(high_noisy - clean_signals))

    assert high_difference > low_difference
