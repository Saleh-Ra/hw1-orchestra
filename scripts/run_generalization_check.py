"""Compare random-window vs held-out-signal-set generalization.

The random-window split lets every signal set leak across train/test.
The held-out-signal-set split keeps each set in exactly one partition,
so test signal sets are never seen during training. Comparing the two
shows whether each model truly generalizes to new signals or memorizes
per-set quirks.
"""

import json
from pathlib import Path

from ai_orchestra import (
    DEFAULTS,
    FullyConnectedSignalDataset,
    PipelineDefaults,
    compare_models,
    generate_signal_sets,
    holdout_signal_set_split_indices,
    save_comparison_csv,
    save_comparison_json,
)


def main() -> None:
    defaults = PipelineDefaults(num_samples=100, batch_size=32)
    signal_set_count = 5
    epochs = 3
    learning_rate = 1e-3
    hidden_size = 32
    seed = DEFAULTS.random_seed

    output_dir = Path("results/generalization")
    output_dir.mkdir(parents=True, exist_ok=True)

    random_result = compare_models(
        defaults=defaults,
        signal_set_count=signal_set_count,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        seed=seed,
        device="cpu",
        split_name="random",
    )
    save_comparison_json(random_result, output_dir / "random_split_metrics.json")
    save_comparison_csv(random_result, output_dir / "random_split_metrics.csv")

    signal_sets = generate_signal_sets(defaults, count=signal_set_count, seed=seed)
    dataset = FullyConnectedSignalDataset(signal_sets, defaults)
    train_idx, test_idx = holdout_signal_set_split_indices(
        num_signal_sets=signal_set_count,
        defaults=defaults,
        train_ratio=defaults.train_ratio,
        seed=seed,
    )
    holdout_result = compare_models(
        defaults=defaults,
        signal_set_count=signal_set_count,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        seed=seed,
        device="cpu",
        split_indices=(train_idx, test_idx),
        split_name="holdout_signal_set",
    )
    save_comparison_json(holdout_result, output_dir / "holdout_split_metrics.json")
    save_comparison_csv(holdout_result, output_dir / "holdout_split_metrics.csv")

    summary = {
        "signal_set_count": signal_set_count,
        "train_set_count": int(len(train_idx) / dataset.examples_per_set),
        "test_set_count": int(len(test_idx) / dataset.examples_per_set),
        "epochs": epochs,
        "random_split": random_result.overall_mse,
        "holdout_split": holdout_result.overall_mse,
        "delta_random_to_holdout": {
            model: holdout_result.overall_mse[model] - random_result.overall_mse[model]
            for model in random_result.overall_mse
        },
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("Random split overall MSE:")
    for model_name, value in random_result.overall_mse.items():
        print(f"  {model_name}: {value:.4f}")
    print("Held-out-signal-set overall MSE:")
    for model_name, value in holdout_result.overall_mse.items():
        print(f"  {model_name}: {value:.4f}")
    print("Delta (holdout - random, positive means worse generalization):")
    for model_name, delta in summary["delta_random_to_holdout"].items():
        print(f"  {model_name}: {delta:+.4f}")


if __name__ == "__main__":
    main()
