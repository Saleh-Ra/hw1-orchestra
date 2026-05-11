import numpy as np
import pytest

from ai_orchestra import (
    DEFAULTS,
    PipelineDefaults,
    holdout_signal_set_split_indices,
)
from ai_orchestra.dataset import examples_per_signal_set


def test_holdout_split_partitions_signal_sets_disjointly() -> None:
    defaults = PipelineDefaults(num_samples=100)
    per_set = examples_per_signal_set(defaults)

    train_idx, test_idx = holdout_signal_set_split_indices(
        num_signal_sets=5,
        defaults=defaults,
        train_ratio=0.8,
        seed=DEFAULTS.random_seed,
    )

    train_set_ids = {int(i) // per_set for i in train_idx}
    test_set_ids = {int(i) // per_set for i in test_idx}
    assert train_set_ids.isdisjoint(test_set_ids)
    assert train_set_ids | test_set_ids == set(range(5))
    assert len(train_set_ids) == 4
    assert len(test_set_ids) == 1


def test_holdout_split_returns_full_signal_set_contents() -> None:
    defaults = PipelineDefaults(num_samples=100)
    per_set = examples_per_signal_set(defaults)

    train_idx, test_idx = holdout_signal_set_split_indices(
        num_signal_sets=5,
        defaults=defaults,
        train_ratio=0.8,
        seed=DEFAULTS.random_seed,
    )

    assert len(train_idx) % per_set == 0
    assert len(test_idx) % per_set == 0
    assert len(train_idx) + len(test_idx) == 5 * per_set


def test_holdout_split_is_deterministic_for_a_seed() -> None:
    defaults = PipelineDefaults(num_samples=100)

    first = holdout_signal_set_split_indices(num_signal_sets=10, defaults=defaults)
    second = holdout_signal_set_split_indices(num_signal_sets=10, defaults=defaults)

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_holdout_split_rejects_single_signal_set() -> None:
    with pytest.raises(ValueError, match="at least 2 signal sets"):
        holdout_signal_set_split_indices(num_signal_sets=1)


def test_holdout_split_rejects_ratio_that_empties_a_partition() -> None:
    defaults = PipelineDefaults(num_samples=100)

    with pytest.raises(ValueError, match="at least one signal set"):
        holdout_signal_set_split_indices(
            num_signal_sets=2,
            defaults=defaults,
            train_ratio=0.1,
        )


def test_compare_models_accepts_custom_split_indices() -> None:
    pytest.importorskip("torch")

    from ai_orchestra import compare_models

    defaults = PipelineDefaults(num_samples=100, batch_size=16)
    train_idx, test_idx = holdout_signal_set_split_indices(
        num_signal_sets=3,
        defaults=defaults,
        train_ratio=2 / 3,
        seed=DEFAULTS.random_seed,
    )

    result = compare_models(
        defaults=defaults,
        signal_set_count=3,
        epochs=1,
        hidden_size=8,
        device="cpu",
        split_indices=(train_idx, test_idx),
        split_name="holdout_signal_set",
    )

    assert result.settings["split"] == "holdout_signal_set"
    assert set(result.overall_mse) == {"FC", "RNN", "LSTM"}
