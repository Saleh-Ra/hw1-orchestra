import numpy as np

from ai_orchestra import DEFAULTS, SignalSet, generate_signal_set


def test_signal_set_can_be_generated_with_one_call() -> None:
    signal_set = generate_signal_set(seed=DEFAULTS.random_seed)

    assert isinstance(signal_set, SignalSet)


def test_signal_set_stores_expected_shapes() -> None:
    signal_set = generate_signal_set(seed=DEFAULTS.random_seed)
    signal_count = len(DEFAULTS.frequencies_hz)

    assert signal_set.time_axis.shape == (DEFAULTS.num_samples,)
    assert signal_set.amplitudes.shape == (signal_count,)
    assert signal_set.phases.shape == (signal_count,)
    assert signal_set.clean_signals.shape == (signal_count, DEFAULTS.num_samples)
    assert signal_set.noisy_signals.shape == (signal_count, DEFAULTS.num_samples)
    assert signal_set.clean_mix.shape == (DEFAULTS.num_samples,)
    assert signal_set.noisy_mix.shape == (DEFAULTS.num_samples,)


def test_signal_set_stores_frequencies_in_default_order() -> None:
    signal_set = generate_signal_set(seed=DEFAULTS.random_seed)

    assert signal_set.frequencies_hz == DEFAULTS.frequencies_hz


def test_signal_set_mixes_match_signal_sums() -> None:
    signal_set = generate_signal_set(seed=DEFAULTS.random_seed)

    assert np.allclose(signal_set.clean_mix, np.sum(signal_set.clean_signals, axis=0))
    assert np.allclose(signal_set.noisy_mix, np.sum(signal_set.noisy_signals, axis=0))


def test_signal_set_clean_and_noisy_outputs_differ() -> None:
    signal_set = generate_signal_set(seed=DEFAULTS.random_seed)

    assert not np.allclose(signal_set.clean_signals, signal_set.noisy_signals)
    assert not np.allclose(signal_set.clean_mix, signal_set.noisy_mix)


def test_signal_set_generation_is_reproducible_with_seed() -> None:
    first = generate_signal_set(seed=DEFAULTS.random_seed)
    second = generate_signal_set(seed=DEFAULTS.random_seed)

    assert np.allclose(first.amplitudes, second.amplitudes)
    assert np.allclose(first.phases, second.phases)
    assert np.allclose(first.clean_signals, second.clean_signals)
    assert np.allclose(first.noisy_signals, second.noisy_signals)
    assert np.allclose(first.clean_mix, second.clean_mix)
    assert np.allclose(first.noisy_mix, second.noisy_mix)
