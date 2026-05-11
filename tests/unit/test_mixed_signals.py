import numpy as np

from ai_orchestra import DEFAULTS
from ai_orchestra.signals import (
    combine_signals,
    generate_clean_signals,
    generate_noisy_signals,
    sample_signal_parameters,
)


def test_clean_mixed_signal_shape_matches_sample_count() -> None:
    clean_signals = generate_clean_signals()
    clean_mix = combine_signals(clean_signals)

    assert clean_mix.shape == (DEFAULTS.num_samples,)


def test_noisy_mixed_signal_shape_matches_sample_count() -> None:
    noisy_signals = generate_noisy_signals(seed=DEFAULTS.random_seed)
    noisy_mix = combine_signals(noisy_signals)

    assert noisy_mix.shape == (DEFAULTS.num_samples,)


def test_clean_mixed_signal_equals_sum_of_clean_signals() -> None:
    clean_signals = generate_clean_signals()
    clean_mix = combine_signals(clean_signals)

    assert np.allclose(clean_mix, np.sum(clean_signals, axis=0))


def test_noisy_mixed_signal_equals_sum_of_noisy_signals() -> None:
    noisy_signals = generate_noisy_signals(seed=DEFAULTS.random_seed)
    noisy_mix = combine_signals(noisy_signals)

    assert np.allclose(noisy_mix, np.sum(noisy_signals, axis=0))


def test_mixed_signals_contain_finite_values() -> None:
    clean_mix = combine_signals(generate_clean_signals())
    noisy_mix = combine_signals(generate_noisy_signals(seed=DEFAULTS.random_seed))

    assert np.isfinite(clean_mix).all()
    assert np.isfinite(noisy_mix).all()


def test_clean_and_noisy_mixed_signals_differ_when_noise_is_enabled() -> None:
    amplitudes, phases = sample_signal_parameters(seed=DEFAULTS.random_seed)
    clean_mix = combine_signals(generate_clean_signals(amplitudes=amplitudes, phases=phases))
    noisy_mix = combine_signals(
        generate_noisy_signals(
            amplitudes=amplitudes,
            phases=phases,
            seed=DEFAULTS.random_seed,
        ),
    )

    assert not np.allclose(clean_mix, noisy_mix)


def test_combine_signals_rejects_non_stack_inputs() -> None:
    invalid_signal = np.zeros(DEFAULTS.num_samples)

    try:
        combine_signals(invalid_signal)
    except ValueError as error:
        assert "2D array" in str(error)
    else:
        raise AssertionError("Expected combine_signals to reject 1D inputs.")
