import numpy as np
import pytest

from ai_orchestra import (
    DEFAULTS,
    FullyConnectedSignalDataset,
    PipelineDefaults,
    create_data_loaders,
    generate_signal_set,
    train_test_split_indices,
)


def test_create_data_loaders_batches_have_expected_shapes() -> None:
    torch = pytest.importorskip("torch")
    defaults = PipelineDefaults(num_samples=100, batch_size=8)
    dataset = FullyConnectedSignalDataset(
        [generate_signal_set(defaults, seed=DEFAULTS.random_seed)],
        defaults,
    )
    train_indices, test_indices = train_test_split_indices(
        len(dataset),
        seed=DEFAULTS.random_seed,
    )

    train_loader, test_loader = create_data_loaders(
        dataset,
        train_indices,
        test_indices,
        defaults,
    )
    train_inputs, train_targets = next(iter(train_loader))
    test_inputs, test_targets = next(iter(test_loader))

    assert train_inputs.shape == (defaults.batch_size, 14)
    assert train_targets.shape == (defaults.batch_size, 10)
    assert test_inputs.shape[1:] == (14,)
    assert test_targets.shape[1:] == (10,)
    assert torch.isfinite(train_inputs).all()
    assert torch.isfinite(train_targets).all()


def test_create_data_loaders_uses_custom_batch_size() -> None:
    pytest.importorskip("torch")
    defaults = PipelineDefaults(num_samples=100, batch_size=8)
    dataset = FullyConnectedSignalDataset(
        [generate_signal_set(defaults, seed=DEFAULTS.random_seed)],
        defaults,
    )
    train_indices, test_indices = train_test_split_indices(len(dataset))

    train_loader, _ = create_data_loaders(
        dataset,
        train_indices,
        test_indices,
        defaults,
        batch_size=4,
    )
    train_inputs, train_targets = next(iter(train_loader))

    assert train_inputs.shape == (4, 14)
    assert train_targets.shape == (4, 10)


def test_create_data_loaders_rejects_invalid_batch_size() -> None:
    dataset = object()
    train_indices = np.array([0], dtype=np.int64)
    test_indices = np.array([1], dtype=np.int64)

    try:
        create_data_loaders(dataset, train_indices, test_indices, batch_size=0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("Expected invalid batch size to fail.")
