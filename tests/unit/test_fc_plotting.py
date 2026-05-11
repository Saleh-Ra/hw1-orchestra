import numpy as np

from ai_orchestra.baseline import FcBaselineResult
from ai_orchestra.fc_plotting import (
    plot_fc_baseline_result,
    plot_loss_curves,
    plot_prediction_window,
)


def test_plot_prediction_window_saves_file(tmp_path) -> None:
    prediction = np.linspace(0.0, 1.0, 10)
    target = np.linspace(0.1, 1.1, 10)
    output_path = tmp_path / "prediction.png"

    saved_path = plot_prediction_window(prediction, target, 1.0, output_path)

    assert saved_path == output_path
    assert output_path.exists()


def test_plot_loss_curves_saves_file(tmp_path) -> None:
    output_path = tmp_path / "loss.png"

    saved_path = plot_loss_curves([0.9, 0.7], [1.0, 0.8], output_path)

    assert saved_path == output_path
    assert output_path.exists()


def test_plot_fc_baseline_result_saves_predictions_and_loss(tmp_path) -> None:
    result = FcBaselineResult(
        train_losses=[0.9, 0.7],
        test_losses=[1.0, 0.8],
        predictions=[np.zeros(10), np.ones(10)],
        targets=[np.ones(10), np.zeros(10)],
        class_indices=[0, 3],
    )

    saved_paths = plot_fc_baseline_result(result, (1.0, 3.0, 5.0, 7.0), tmp_path)

    assert {path.name for path in saved_paths} == {
        "fc_loss_curve.png",
        "fc_prediction_s1_1hz.png",
        "fc_prediction_s4_7hz.png",
    }
    assert all(path.exists() for path in saved_paths)


def test_plot_prediction_window_rejects_shape_mismatch(tmp_path) -> None:
    try:
        plot_prediction_window(np.zeros(10), np.zeros(9), 1.0, tmp_path / "bad.png")
    except ValueError as error:
        assert "same shape" in str(error)
    else:
        raise AssertionError("Expected shape mismatch to fail.")
