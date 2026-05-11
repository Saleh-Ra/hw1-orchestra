import numpy as np
import pytest

from ai_orchestra import DEFAULTS, SequenceSignalDataset, generate_signal_set


def _torch():
    return pytest.importorskip("torch")


def test_sequence_dataset_length_matches_fc_dataset_length() -> None:
    signal_set = generate_signal_set(seed=DEFAULTS.random_seed)
    dataset = SequenceSignalDataset([signal_set])

    assert len(dataset) == 9991 * len(DEFAULTS.frequencies_hz)


def test_sequence_item_shapes_and_tensor_types() -> None:
    torch = _torch()
    signal_set = generate_signal_set(seed=DEFAULTS.random_seed)
    dataset = SequenceSignalDataset([signal_set])
    sequence_input, target = dataset[0]

    assert isinstance(sequence_input, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert sequence_input.shape == (10, 5)
    assert target.shape == (10,)
    assert sequence_input.dtype == torch.float32
    assert target.dtype == torch.float32


def test_sequence_item_carries_noisy_window_in_first_channel() -> None:
    _torch()
    signal_set = generate_signal_set(seed=DEFAULTS.random_seed)
    dataset = SequenceSignalDataset([signal_set])
    sequence_input, _ = dataset[0]

    assert np.allclose(sequence_input[:, 0].numpy(), signal_set.noisy_mix[:10])


def test_sequence_item_repeats_condition_across_time_steps() -> None:
    _torch()
    signal_set = generate_signal_set(seed=DEFAULTS.random_seed)
    dataset = SequenceSignalDataset([signal_set])
    sequence_input, _ = dataset[1]

    expected_condition = np.array([0.0, 1.0, 0.0, 0.0])
    for time_step in range(10):
        assert np.allclose(sequence_input[time_step, 1:].numpy(), expected_condition)


def test_sequence_target_mapping_for_each_class() -> None:
    _torch()
    signal_set = generate_signal_set(seed=DEFAULTS.random_seed)
    dataset = SequenceSignalDataset([signal_set])

    for class_index in range(len(DEFAULTS.frequencies_hz)):
        _, target = dataset[class_index]
        assert np.allclose(target.numpy(), signal_set.clean_signals[class_index, :10])


def test_sequence_batch_shape_via_dataloader() -> None:
    torch = _torch()
    from torch.utils.data import DataLoader

    signal_set = generate_signal_set(seed=DEFAULTS.random_seed)
    dataset = SequenceSignalDataset([signal_set])
    loader = DataLoader(dataset, batch_size=8)
    batch_inputs, batch_targets = next(iter(loader))

    assert batch_inputs.shape == (8, 10, 5)
    assert batch_targets.shape == (8, 10)
    assert torch.isfinite(batch_inputs).all()
    assert torch.isfinite(batch_targets).all()


def test_sequence_middle_window_matches_fc_indexing() -> None:
    _torch()
    signal_set = generate_signal_set(seed=DEFAULTS.random_seed)
    dataset = SequenceSignalDataset([signal_set])
    index = (6203 * len(DEFAULTS.frequencies_hz)) + 1
    sequence_input, target = dataset[index]

    assert np.allclose(sequence_input[:, 0].numpy(), signal_set.noisy_mix[6203:6213])
    assert np.allclose(target.numpy(), signal_set.clean_signals[1, 6203:6213])


def test_sequence_dataset_rejects_invalid_index() -> None:
    _torch()
    signal_set = generate_signal_set(seed=DEFAULTS.random_seed)
    dataset = SequenceSignalDataset([signal_set])

    try:
        dataset[len(dataset)]
    except IndexError as error:
        assert "Dataset index" in str(error)
    else:
        raise AssertionError("Expected out-of-range dataset index to fail.")
