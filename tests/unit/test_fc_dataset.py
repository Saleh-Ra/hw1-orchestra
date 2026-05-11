import numpy as np
import pytest

from ai_orchestra import DEFAULTS, FullyConnectedSignalDataset, generate_signal_set


def _torch():
    return pytest.importorskip("torch")


def test_fc_dataset_length_matches_windows_times_classes() -> None:
    signal_set = generate_signal_set(seed=DEFAULTS.random_seed)
    dataset = FullyConnectedSignalDataset([signal_set])

    assert len(dataset) == 9991 * len(DEFAULTS.frequencies_hz)


def test_first_fc_dataset_item_shapes_and_tensor_types() -> None:
    torch = _torch()
    signal_set = generate_signal_set(seed=DEFAULTS.random_seed)
    dataset = FullyConnectedSignalDataset([signal_set])
    fc_input, target = dataset[0]

    assert isinstance(fc_input, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert fc_input.shape == (14,)
    assert target.shape == (10,)
    assert fc_input.dtype == torch.float32
    assert target.dtype == torch.float32


def test_first_fc_dataset_item_contains_noisy_window_plus_condition() -> None:
    _torch()
    signal_set = generate_signal_set(seed=DEFAULTS.random_seed)
    dataset = FullyConnectedSignalDataset([signal_set])
    fc_input, _ = dataset[0]

    assert np.allclose(fc_input[:10].numpy(), signal_set.noisy_mix[:10])
    assert np.allclose(fc_input[10:].numpy(), np.array([1.0, 0.0, 0.0, 0.0]))


def test_condition_zero_targets_first_clean_signal() -> None:
    _torch()
    signal_set = generate_signal_set(seed=DEFAULTS.random_seed)
    dataset = FullyConnectedSignalDataset([signal_set])
    fc_input, target = dataset[0]

    assert np.allclose(fc_input[10:].numpy(), np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(target.numpy(), signal_set.clean_signals[0, :10])


def test_condition_one_targets_second_clean_signal() -> None:
    _torch()
    signal_set = generate_signal_set(seed=DEFAULTS.random_seed)
    dataset = FullyConnectedSignalDataset([signal_set])
    fc_input, target = dataset[1]

    assert np.allclose(fc_input[10:].numpy(), np.array([0.0, 1.0, 0.0, 0.0]))
    assert np.allclose(target.numpy(), signal_set.clean_signals[1, :10])


def test_condition_two_targets_third_clean_signal() -> None:
    _torch()
    signal_set = generate_signal_set(seed=DEFAULTS.random_seed)
    dataset = FullyConnectedSignalDataset([signal_set])
    fc_input, target = dataset[2]

    assert np.allclose(fc_input[10:].numpy(), np.array([0.0, 0.0, 1.0, 0.0]))
    assert np.allclose(target.numpy(), signal_set.clean_signals[2, :10])


def test_condition_three_targets_fourth_clean_signal() -> None:
    _torch()
    signal_set = generate_signal_set(seed=DEFAULTS.random_seed)
    dataset = FullyConnectedSignalDataset([signal_set])
    fc_input, target = dataset[3]

    assert np.allclose(fc_input[10:].numpy(), np.array([0.0, 0.0, 0.0, 1.0]))
    assert np.allclose(target.numpy(), signal_set.clean_signals[3, :10])


def test_middle_window_uses_matching_time_position() -> None:
    _torch()
    signal_set = generate_signal_set(seed=DEFAULTS.random_seed)
    dataset = FullyConnectedSignalDataset([signal_set])
    index = (6203 * len(DEFAULTS.frequencies_hz)) + 1
    fc_input, target = dataset[index]

    assert np.allclose(fc_input[:10].numpy(), signal_set.noisy_mix[6203:6213])
    assert np.allclose(target.numpy(), signal_set.clean_signals[1, 6203:6213])


def test_fc_dataset_rejects_invalid_index() -> None:
    _torch()
    signal_set = generate_signal_set(seed=DEFAULTS.random_seed)
    dataset = FullyConnectedSignalDataset([signal_set])

    try:
        dataset[len(dataset)]
    except IndexError as error:
        assert "Dataset index" in str(error)
    else:
        raise AssertionError("Expected out-of-range dataset index to fail.")
