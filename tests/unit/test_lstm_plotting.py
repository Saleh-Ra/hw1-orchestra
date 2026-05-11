import numpy as np

from ai_orchestra.lstm_baseline import LstmBaselineResult
from ai_orchestra.lstm_plotting import plot_lstm_baseline_result


def test_plot_lstm_baseline_result_saves_predictions_and_loss(tmp_path) -> None:
    result = LstmBaselineResult(
        train_losses=[0.9, 0.7],
        test_losses=[1.0, 0.8],
        predictions=[np.zeros(10), np.ones(10)],
        targets=[np.ones(10), np.zeros(10)],
        class_indices=[0, 3],
    )

    saved_paths = plot_lstm_baseline_result(result, (1.0, 3.0, 5.0, 7.0), tmp_path)

    assert {path.name for path in saved_paths} == {
        "lstm_loss_curve.png",
        "lstm_prediction_s1_1hz.png",
        "lstm_prediction_s4_7hz.png",
    }
    assert all(path.exists() for path in saved_paths)
