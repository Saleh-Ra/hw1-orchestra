def _make_sample_comparison():
    from ai_orchestra import ModelComparison

    return ModelComparison(
        settings={"epochs": 2, "signal_set_count": 1},
        overall_mse={"FC": 0.5, "RNN": 0.3, "LSTM": 0.6},
        per_frequency_mse={
            "FC": {1.0: 0.4, 3.0: 0.5, 5.0: 0.55, 7.0: 0.6},
            "RNN": {1.0: 0.25, 3.0: 0.3, 5.0: 0.32, 7.0: 0.35},
            "LSTM": {1.0: 0.55, 3.0: 0.6, 5.0: 0.62, 7.0: 0.65},
        },
    )


def test_plot_overall_mse_creates_file(tmp_path) -> None:
    from ai_orchestra import plot_overall_mse

    output_path = tmp_path / "overall.png"
    plot_overall_mse(_make_sample_comparison(), output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_per_frequency_mse_creates_file(tmp_path) -> None:
    from ai_orchestra import plot_per_frequency_mse

    output_path = tmp_path / "per_freq.png"
    plot_per_frequency_mse(_make_sample_comparison(), output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
