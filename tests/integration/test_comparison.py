import math

import pytest


def test_compare_models_returns_metrics_for_all_models() -> None:
    pytest.importorskip("torch")

    from ai_orchestra import DEFAULTS, PipelineDefaults, compare_models

    result = compare_models(
        defaults=PipelineDefaults(num_samples=100, batch_size=16),
        signal_set_count=1,
        epochs=2,
        hidden_size=8,
        seed=DEFAULTS.random_seed,
        device="cpu",
    )

    assert set(result.overall_mse.keys()) == {"FC", "RNN", "LSTM"}
    assert set(result.per_frequency_mse.keys()) == {"FC", "RNN", "LSTM"}
    for model_name in ("FC", "RNN", "LSTM"):
        overall = result.overall_mse[model_name]
        per_freq = result.per_frequency_mse[model_name]
        assert math.isfinite(overall) and overall >= 0.0
        assert set(per_freq.keys()) == set(DEFAULTS.frequencies_hz[:4])
        assert all(math.isfinite(mse) and mse >= 0.0 for mse in per_freq.values())
        assert len(result.train_losses[model_name]) == 2
        assert len(result.test_losses[model_name]) == 2
    assert result.settings["epochs"] == 2
    assert result.settings["signal_set_count"] == 1


def test_compare_models_rejects_invalid_epochs() -> None:
    pytest.importorskip("torch")

    from ai_orchestra import compare_models

    with pytest.raises(ValueError, match="positive"):
        compare_models(epochs=0)


def test_compare_models_rejects_invalid_hidden_size() -> None:
    pytest.importorskip("torch")

    from ai_orchestra import compare_models

    with pytest.raises(ValueError, match="positive"):
        compare_models(hidden_size=0)
