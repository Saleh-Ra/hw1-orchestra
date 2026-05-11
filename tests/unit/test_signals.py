import numpy as np

from ai_orchestra import DEFAULTS
from ai_orchestra.signals import (
    generate_clean_signal,
    generate_clean_signals,
    make_time_axis,
)


def test_time_axis_length_matches_default_sample_count() -> None:
    time_axis = make_time_axis()

    assert time_axis.shape == (DEFAULTS.num_samples,)


def test_time_axis_starts_at_zero() -> None:
    time_axis = make_time_axis()

    assert time_axis[0] == 0.0


def test_time_axis_step_matches_sampling_frequency() -> None:
    time_axis = make_time_axis()

    assert np.isclose(time_axis[1] - time_axis[0], 1 / DEFAULTS.sampling_frequency_hz)


def test_clean_signal_shape_matches_time_axis() -> None:
    time_axis = make_time_axis()
    signal = generate_clean_signal(
        frequency_hz=1.0,
        time_axis=time_axis,
        amplitude=1.0,
        phase=0.0,
    )

    assert signal.shape == (DEFAULTS.num_samples,)


def test_clean_signal_contains_finite_values() -> None:
    signal = generate_clean_signal(
        frequency_hz=1.0,
        time_axis=make_time_axis(),
        amplitude=1.0,
        phase=0.0,
    )

    assert np.isfinite(signal).all()


def test_changing_frequency_changes_signal() -> None:
    time_axis = make_time_axis()
    low_frequency = generate_clean_signal(1.0, time_axis, amplitude=1.0, phase=0.0)
    high_frequency = generate_clean_signal(3.0, time_axis, amplitude=1.0, phase=0.0)

    assert not np.allclose(low_frequency, high_frequency)


def test_changing_phase_changes_signal() -> None:
    time_axis = make_time_axis()
    zero_phase = generate_clean_signal(1.0, time_axis, amplitude=1.0, phase=0.0)
    shifted_phase = generate_clean_signal(
        1.0,
        time_axis,
        amplitude=1.0,
        phase=np.pi / 2,
    )

    assert not np.allclose(zero_phase, shifted_phase)


def test_changing_amplitude_changes_signal_scale() -> None:
    time_axis = make_time_axis()
    base_signal = generate_clean_signal(1.0, time_axis, amplitude=1.0, phase=0.0)
    larger_signal = generate_clean_signal(1.0, time_axis, amplitude=2.0, phase=0.0)

    assert np.allclose(larger_signal, base_signal * 2.0)


def test_clean_signal_stack_shape_matches_default_frequencies() -> None:
    signals = generate_clean_signals()

    assert signals.shape == (len(DEFAULTS.frequencies_hz), DEFAULTS.num_samples)


def test_clean_signal_stack_uses_frequency_order() -> None:
    signals = generate_clean_signals()
    time_axis = make_time_axis()

    for index, frequency in enumerate(DEFAULTS.frequencies_hz):
        expected_signal = generate_clean_signal(
            frequency_hz=frequency,
            time_axis=time_axis,
            amplitude=1.0,
            phase=0.0,
        )
        assert np.allclose(signals[index], expected_signal)


def test_clean_signal_stack_contains_distinct_signals() -> None:
    signals = generate_clean_signals()

    assert not np.allclose(signals[0], signals[1])
    assert not np.allclose(signals[1], signals[2])
    assert not np.allclose(signals[2], signals[3])


def test_clean_signal_stack_is_deterministic() -> None:
    first = generate_clean_signals()
    second = generate_clean_signals()

    assert np.allclose(first, second)
