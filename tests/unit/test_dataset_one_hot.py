import numpy as np

from ai_orchestra import DEFAULTS, PipelineDefaults, make_one_hot


def test_class_zero_maps_to_first_frequency() -> None:
    assert np.allclose(make_one_hot(0), np.array([1.0, 0.0, 0.0, 0.0]))


def test_class_one_maps_to_second_frequency() -> None:
    assert np.allclose(make_one_hot(1), np.array([0.0, 1.0, 0.0, 0.0]))


def test_class_two_maps_to_third_frequency() -> None:
    assert np.allclose(make_one_hot(2), np.array([0.0, 0.0, 1.0, 0.0]))


def test_class_three_maps_to_fourth_frequency() -> None:
    assert np.allclose(make_one_hot(3), np.array([0.0, 0.0, 0.0, 1.0]))


def test_one_hot_length_matches_frequency_count() -> None:
    vector = make_one_hot(0)

    assert vector.shape == (len(DEFAULTS.frequencies_hz),)


def test_one_hot_length_uses_custom_frequency_count() -> None:
    defaults = PipelineDefaults(frequencies_hz=(2.0, 4.0, 6.0))

    assert np.allclose(make_one_hot(2, defaults), np.array([0.0, 0.0, 1.0]))


def test_negative_class_index_fails_clearly() -> None:
    try:
        make_one_hot(-1)
    except ValueError as error:
        assert "between 0 and 3" in str(error)
    else:
        raise AssertionError("Expected negative class index to fail.")


def test_too_large_class_index_fails_clearly() -> None:
    try:
        make_one_hot(4)
    except ValueError as error:
        assert "between 0 and 3" in str(error)
    else:
        raise AssertionError("Expected out-of-range class index to fail.")
