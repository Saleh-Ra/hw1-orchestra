import csv
import json


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
        train_losses={"FC": [0.8, 0.5], "RNN": [0.7, 0.3], "LSTM": [0.9, 0.6]},
        test_losses={"FC": [0.85, 0.5], "RNN": [0.7, 0.3], "LSTM": [0.95, 0.6]},
    )


def test_save_comparison_json_writes_expected_keys(tmp_path) -> None:
    from ai_orchestra import save_comparison_json

    output_path = tmp_path / "metrics.json"
    save_comparison_json(_make_sample_comparison(), output_path)

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert set(payload.keys()) == {
        "settings",
        "overall_mse",
        "per_frequency_mse",
        "train_losses",
        "test_losses",
    }
    assert payload["overall_mse"]["RNN"] == 0.3
    assert payload["per_frequency_mse"]["FC"]["1"] == 0.4


def test_save_comparison_csv_contains_expected_rows(tmp_path) -> None:
    from ai_orchestra import save_comparison_csv

    output_path = tmp_path / "metrics.csv"
    save_comparison_csv(_make_sample_comparison(), output_path)

    assert output_path.exists()
    with output_path.open(encoding="utf-8") as file_handle:
        rows = list(csv.reader(file_handle))
    assert rows[0] == [
        "model",
        "overall_mse",
        "mse_at_1hz",
        "mse_at_3hz",
        "mse_at_5hz",
        "mse_at_7hz",
    ]
    assert rows[1][0] == "FC"
    assert float(rows[2][1]) == 0.3
