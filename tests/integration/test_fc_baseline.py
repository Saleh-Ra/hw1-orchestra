import numpy as np
import pytest


def test_fc_baseline_returns_losses_and_predictions() -> None:
    torch = pytest.importorskip("torch")

    from ai_orchestra import DEFAULTS, PipelineDefaults
    from ai_orchestra.baseline import run_fc_baseline

    result = run_fc_baseline(
        defaults=PipelineDefaults(num_samples=100, batch_size=16),
        signal_set_count=1,
        epochs=2,
        prediction_count=4,
        seed=DEFAULTS.random_seed,
        device="cpu",
    )

    assert len(result.train_losses) == 2
    assert len(result.test_losses) == 2
    assert len(result.predictions) == 4
    assert len(result.targets) == 4
    assert set(result.class_indices) == {0, 1, 2, 3}
    assert all(loss >= 0.0 for loss in result.train_losses)
    assert all(loss >= 0.0 for loss in result.test_losses)
    assert torch.isfinite(torch.tensor(result.train_losses)).all()
    assert torch.isfinite(torch.tensor(result.test_losses)).all()
    assert all(prediction.shape == (10,) for prediction in result.predictions)
    assert all(target.shape == (10,) for target in result.targets)
    assert all(np.isfinite(prediction).all() for prediction in result.predictions)


def test_fc_baseline_rejects_invalid_epoch_count() -> None:
    pytest.importorskip("torch")

    from ai_orchestra.baseline import run_fc_baseline

    try:
        run_fc_baseline(epochs=0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("Expected invalid epoch count to fail.")


def test_fc_baseline_rejects_invalid_prediction_count() -> None:
    pytest.importorskip("torch")

    from ai_orchestra.baseline import run_fc_baseline

    try:
        run_fc_baseline(prediction_count=0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("Expected invalid prediction count to fail.")
