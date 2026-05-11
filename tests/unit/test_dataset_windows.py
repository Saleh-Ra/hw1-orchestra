import numpy as np

from ai_orchestra import DEFAULTS, PipelineDefaults, extract_window, valid_window_starts


def test_valid_window_starts_use_default_window_size() -> None:
    starts = valid_window_starts()

    assert starts[0] == 0
    assert starts[-1] == 9990


def test_valid_window_starts_length_matches_default_signal() -> None:
    starts = valid_window_starts()

    assert len(starts) == 9991


def test_valid_window_starts_use_custom_defaults() -> None:
    starts = valid_window_starts(PipelineDefaults(num_samples=20, window_size=5))

    assert np.allclose(starts, np.arange(16))


def test_extract_first_window() -> None:
    signal = np.arange(DEFAULTS.num_samples, dtype=np.float64)
    window = extract_window(signal, start=0)

    assert np.allclose(window, np.arange(10, dtype=np.float64))


def test_extract_middle_window() -> None:
    signal = np.arange(DEFAULTS.num_samples, dtype=np.float64)
    window = extract_window(signal, start=6203)

    assert np.allclose(window, np.arange(6203, 6213, dtype=np.float64))


def test_extract_last_valid_window_without_off_by_one_error() -> None:
    signal = np.arange(DEFAULTS.num_samples, dtype=np.float64)
    window = extract_window(signal, start=9990)

    assert np.allclose(window, np.arange(9990, 10000, dtype=np.float64))


def test_extract_window_rejects_negative_start() -> None:
    signal = np.arange(DEFAULTS.num_samples, dtype=np.float64)

    try:
        extract_window(signal, start=-1)
    except ValueError as error:
        assert "between 0 and 9990" in str(error)
    else:
        raise AssertionError("Expected negative window start to fail.")


def test_extract_window_rejects_start_after_last_valid_start() -> None:
    signal = np.arange(DEFAULTS.num_samples, dtype=np.float64)

    try:
        extract_window(signal, start=9991)
    except ValueError as error:
        assert "between 0 and 9990" in str(error)
    else:
        raise AssertionError("Expected too-large window start to fail.")


def test_extract_window_rejects_non_1d_signal() -> None:
    signal = np.zeros((2, DEFAULTS.num_samples), dtype=np.float64)

    try:
        extract_window(signal, start=0)
    except ValueError as error:
        assert "1D array" in str(error)
    else:
        raise AssertionError("Expected non-1D signal to fail.")
