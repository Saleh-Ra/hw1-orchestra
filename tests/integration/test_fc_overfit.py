import numpy as np
import pytest


def test_fc_overfit_check_returns_losses_prediction_and_plot(tmp_path) -> None:
    torch = pytest.importorskip("torch")

    from ai_orchestra import DEFAULTS, PipelineDefaults
    from ai_orchestra.overfit import run_fc_overfit_check

    result = run_fc_overfit_check(
        defaults=PipelineDefaults(num_samples=100, batch_size=16),
        seed=DEFAULTS.random_seed,
        subset_size=16,
        epochs=5,
        output_path=tmp_path / "prediction.png",
    )

    assert len(result.train_losses) == 5
    assert result.train_losses[-1] <= result.train_losses[0]
    assert result.prediction.shape == (10,)
    assert result.target.shape == (10,)
    assert np.isfinite(result.prediction).all()
    assert np.isfinite(result.target).all()
    assert result.plot_path == tmp_path / "prediction.png"
    assert result.plot_path.exists()
    assert torch.isfinite(torch.tensor(result.train_losses)).all()


def test_fc_overfit_check_rejects_invalid_subset_size() -> None:
    pytest.importorskip("torch")

    from ai_orchestra.overfit import run_fc_overfit_check

    try:
        run_fc_overfit_check(subset_size=0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("Expected invalid subset size to fail.")


def test_fc_overfit_check_rejects_invalid_epoch_count() -> None:
    pytest.importorskip("torch")

    from ai_orchestra.overfit import run_fc_overfit_check

    try:
        run_fc_overfit_check(epochs=0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("Expected invalid epoch count to fail.")
