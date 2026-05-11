import numpy as np

from ai_orchestra import DEFAULTS
from ai_orchestra.signals import generate_clean_signals, sample_signal_parameters


def test_same_seed_gives_same_amplitudes() -> None:
    first_amplitudes, _ = sample_signal_parameters(seed=DEFAULTS.random_seed)
    second_amplitudes, _ = sample_signal_parameters(seed=DEFAULTS.random_seed)

    assert np.allclose(first_amplitudes, second_amplitudes)


def test_same_seed_gives_same_phases() -> None:
    _, first_phases = sample_signal_parameters(seed=DEFAULTS.random_seed)
    _, second_phases = sample_signal_parameters(seed=DEFAULTS.random_seed)

    assert np.allclose(first_phases, second_phases)


def test_different_seed_changes_at_least_one_parameter() -> None:
    first_amplitudes, first_phases = sample_signal_parameters(seed=1)
    second_amplitudes, second_phases = sample_signal_parameters(seed=2)

    amplitudes_changed = not np.allclose(first_amplitudes, second_amplitudes)
    phases_changed = not np.allclose(first_phases, second_phases)
    assert amplitudes_changed or phases_changed


def test_sampled_parameters_have_one_value_per_frequency() -> None:
    amplitudes, phases = sample_signal_parameters(seed=DEFAULTS.random_seed)

    assert amplitudes.shape == (len(DEFAULTS.frequencies_hz),)
    assert phases.shape == (len(DEFAULTS.frequencies_hz),)


def test_sampled_phases_stay_between_zero_and_two_pi() -> None:
    _, phases = sample_signal_parameters(seed=DEFAULTS.random_seed)

    assert np.all(phases >= 0.0)
    assert np.all(phases <= np.pi * 2)


def test_random_parameters_generate_reproducible_clean_signals() -> None:
    first_amplitudes, first_phases = sample_signal_parameters(seed=DEFAULTS.random_seed)
    second_amplitudes, second_phases = sample_signal_parameters(seed=DEFAULTS.random_seed)

    first_signals = generate_clean_signals(amplitudes=first_amplitudes, phases=first_phases)
    second_signals = generate_clean_signals(
        amplitudes=second_amplitudes,
        phases=second_phases,
    )

    assert np.allclose(first_signals, second_signals)
