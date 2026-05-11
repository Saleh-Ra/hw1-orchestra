import pytest


def test_evaluate_per_class_mse_matches_dataset_classes() -> None:
    pytest.importorskip("torch")

    from ai_orchestra import (
        DEFAULTS,
        FullyConnectedSignalDataset,
        PipelineDefaults,
        evaluate_per_class_mse,
        generate_signal_sets,
        train_test_split_indices,
    )
    from ai_orchestra.models import FullyConnectedSignalNet

    defaults = PipelineDefaults(num_samples=100, batch_size=16)
    signal_sets = generate_signal_sets(defaults, count=1, seed=DEFAULTS.random_seed)
    dataset = FullyConnectedSignalDataset(signal_sets, defaults)
    _, test_indices = train_test_split_indices(
        len(dataset), train_ratio=defaults.train_ratio, seed=DEFAULTS.random_seed,
    )

    model = FullyConnectedSignalNet()
    overall, per_class = evaluate_per_class_mse(
        model, dataset, test_indices, defaults, device="cpu",
    )

    assert overall > 0.0
    assert set(per_class.keys()) == set(range(dataset.class_count))
    assert all(mse >= 0.0 for mse in per_class.values())
    class_count = dataset.class_count
    weighted_total = 0.0
    for class_index, class_mse in per_class.items():
        count_in_class = sum(
            1 for idx in test_indices if int(idx) % class_count == class_index
        )
        weighted_total += class_mse * count_in_class
    expected_overall = weighted_total / len(test_indices)
    assert overall == pytest.approx(expected_overall, rel=1e-6)
