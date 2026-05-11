from ai_orchestra import (
    DEFAULTS,
    FullyConnectedSignalDataset,
    examples_per_signal_set,
    generate_signal_set,
    total_examples,
)


def test_examples_per_signal_set_matches_default_window_count() -> None:
    assert examples_per_signal_set() == 9991 * 4
    assert examples_per_signal_set() == 39964


def test_total_examples_for_fifty_sets_matches_plan() -> None:
    assert total_examples(50) == 1_998_200


def test_dataset_length_with_one_signal_set() -> None:
    signal_set = generate_signal_set(seed=DEFAULTS.random_seed)
    dataset = FullyConnectedSignalDataset([signal_set])

    assert len(dataset) == 39_964


def test_dataset_length_with_two_signal_sets() -> None:
    signal_set = generate_signal_set(seed=DEFAULTS.random_seed)
    dataset = FullyConnectedSignalDataset([signal_set, signal_set])

    assert len(dataset) == 79_928


def test_dataset_length_with_fifty_simulated_signal_sets() -> None:
    signal_set = generate_signal_set(seed=DEFAULTS.random_seed)
    dataset = FullyConnectedSignalDataset([signal_set] * 50)

    assert len(dataset) == 1_998_200


def test_dataset_uses_lazy_indexing_metadata() -> None:
    signal_set = generate_signal_set(seed=DEFAULTS.random_seed)
    dataset = FullyConnectedSignalDataset([signal_set] * 50)

    assert len(dataset.signal_sets) == 50
    assert dataset.examples_per_set == 39_964
    assert dataset.window_count == 9_991
    assert dataset.class_count == 4


def test_total_examples_rejects_invalid_signal_set_count() -> None:
    try:
        total_examples(0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("Expected invalid signal set count to fail.")
