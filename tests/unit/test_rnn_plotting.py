import numpy as np

from ai_orchestra.rnn_baseline import RnnBaselineResult
from ai_orchestra.rnn_plotting import plot_rnn_baseline_result


def test_plot_rnn_baseline_result_saves_predictions_and_loss(tmp_path) -> None:
    result = RnnBaselineResult(
        train_losses=[0.9, 0.7],
        test_losses=[1.0, 0.8],
        predictions=[np.zeros(10), np.ones(10)],
        targets=[np.ones(10), np.zeros(10)],
        class_indices=[0, 3],
    )

    saved_paths = plot_rnn_baseline_result(result, (1.0, 3.0, 5.0, 7.0), tmp_path)

    assert {path.name for path in saved_paths} == {
        "rnn_loss_curve.png",
        "rnn_prediction_s1_1hz.png",
        "rnn_prediction_s4_7hz.png",
    }
    assert all(path.exists() for path in saved_paths)
