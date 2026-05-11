import numpy as np
import pytest


def test_compute_mse_matches_manual_value() -> None:
    from ai_orchestra import compute_mse

    predictions = np.array([1.0, 2.0, 3.0])
    targets = np.array([1.5, 2.0, 2.5])

    result = compute_mse(predictions, targets)

    assert result == pytest.approx((0.25 + 0.0 + 0.25) / 3)


def test_compute_mae_matches_manual_value() -> None:
    from ai_orchestra import compute_mae

    predictions = np.array([1.0, 2.0, 3.0])
    targets = np.array([1.5, 2.0, 2.5])

    result = compute_mae(predictions, targets)

    assert result == pytest.approx((0.5 + 0.0 + 0.5) / 3)


def test_compute_mse_rejects_shape_mismatch() -> None:
    from ai_orchestra import compute_mse

    with pytest.raises(ValueError, match="same shape"):
        compute_mse(np.zeros(3), np.zeros(4))


def test_compute_mae_rejects_empty_input() -> None:
    from ai_orchestra import compute_mae

    with pytest.raises(ValueError, match="at least one"):
        compute_mae(np.array([]), np.array([]))


def test_compute_mse_is_zero_for_identical_inputs() -> None:
    from ai_orchestra import compute_mse

    values = np.array([0.1, -0.2, 0.3])

    assert compute_mse(values, values) == pytest.approx(0.0)
