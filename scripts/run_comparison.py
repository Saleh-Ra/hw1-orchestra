"""Run the FC/RNN/LSTM comparison once and write metrics + plots to disk."""

from pathlib import Path

from ai_orchestra import (
    DEFAULTS,
    PipelineDefaults,
    compare_models,
    plot_overall_mse,
    plot_per_frequency_mse,
    save_comparison_csv,
    save_comparison_json,
)


def main() -> None:
    defaults = PipelineDefaults(num_samples=100, batch_size=32)
    comparison = compare_models(
        defaults=defaults,
        signal_set_count=1,
        epochs=5,
        learning_rate=1e-3,
        hidden_size=32,
        seed=DEFAULTS.random_seed,
        device="cpu",
    )

    metrics_dir = Path("results/metrics")
    figures_dir = Path("results/figures/comparison")
    save_comparison_json(comparison, metrics_dir / "model_comparison.json")
    save_comparison_csv(comparison, metrics_dir / "model_comparison.csv")
    plot_overall_mse(comparison, figures_dir / "overall_test_mse.png")
    plot_per_frequency_mse(comparison, figures_dir / "per_frequency_test_mse.png")

    print("Overall test MSE:")
    for model_name, value in comparison.overall_mse.items():
        print(f"  {model_name}: {value:.4f}")
    print("\nPer-frequency test MSE:")
    for model_name, per_freq in comparison.per_frequency_mse.items():
        joined = ", ".join(f"{freq:g}Hz={mse:.4f}" for freq, mse in per_freq.items())
        print(f"  {model_name}: {joined}")


if __name__ == "__main__":
    main()
