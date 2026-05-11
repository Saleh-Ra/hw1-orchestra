import numpy as np

from ai_orchestra import DEFAULTS, train_test_split_indices


def test_train_test_split_uses_default_eighty_twenty_ratio() -> None:
    train_indices, test_indices = train_test_split_indices(100)

    assert len(train_indices) == 80
    assert len(test_indices) == 20


def test_train_test_split_uses_seed_reproducibly() -> None:
    first_train, first_test = train_test_split_indices(100, seed=DEFAULTS.random_seed)
    second_train, second_test = train_test_split_indices(100, seed=DEFAULTS.random_seed)

    assert np.allclose(first_train, second_train)
    assert np.allclose(first_test, second_test)


def test_train_test_split_changes_with_different_seed() -> None:
    first_train, _ = train_test_split_indices(100, seed=1)
    second_train, _ = train_test_split_indices(100, seed=2)

    assert not np.allclose(first_train, second_train)


def test_train_test_split_has_no_duplicate_indices_between_splits() -> None:
    train_indices, test_indices = train_test_split_indices(100)

    assert set(train_indices).isdisjoint(set(test_indices))
    assert set(train_indices) | set(test_indices) == set(range(100))


def test_train_test_split_ratio_can_change() -> None:
    train_indices, test_indices = train_test_split_indices(100, train_ratio=0.7)

    assert len(train_indices) == 70
    assert len(test_indices) == 30


def test_train_test_split_rejects_invalid_ratio() -> None:
    try:
        train_test_split_indices(100, train_ratio=1.0)
    except ValueError as error:
        assert "between 0 and 1" in str(error)
    else:
        raise AssertionError("Expected invalid train ratio to fail.")


def test_train_test_split_rejects_too_few_items() -> None:
    try:
        train_test_split_indices(1)
    except ValueError as error:
        assert "greater than 1" in str(error)
    else:
        raise AssertionError("Expected too-small item count to fail.")
