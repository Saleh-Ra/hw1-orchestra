import pytest


def test_fully_connected_model_forward_shape() -> None:
    torch = pytest.importorskip("torch")
    from ai_orchestra.models import FullyConnectedSignalNet

    model = FullyConnectedSignalNet()
    fake_batch = torch.randn(8, 14)
    output = model(fake_batch)

    assert output.shape == (8, 10)


def test_fully_connected_model_output_is_finite() -> None:
    torch = pytest.importorskip("torch")
    from ai_orchestra.models import FullyConnectedSignalNet

    model = FullyConnectedSignalNet()
    fake_batch = torch.randn(8, 14)
    output = model(fake_batch)

    assert torch.isfinite(output).all()


def test_fully_connected_model_rejects_invalid_sizes() -> None:
    pytest.importorskip("torch")
    from ai_orchestra.models import FullyConnectedSignalNet

    try:
        FullyConnectedSignalNet(input_size=0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("Expected invalid model size to fail.")


def test_rnn_model_forward_shape() -> None:
    torch = pytest.importorskip("torch")
    from ai_orchestra.models import RnnSignalNet

    model = RnnSignalNet()
    fake_batch = torch.randn(8, 10, 5)
    output = model(fake_batch)

    assert output.shape == (8, 10)


def test_rnn_model_output_is_finite() -> None:
    torch = pytest.importorskip("torch")
    from ai_orchestra.models import RnnSignalNet

    model = RnnSignalNet()
    fake_batch = torch.randn(8, 10, 5)
    output = model(fake_batch)

    assert torch.isfinite(output).all()


def test_rnn_model_rejects_invalid_sizes() -> None:
    pytest.importorskip("torch")
    from ai_orchestra.models import RnnSignalNet

    try:
        RnnSignalNet(hidden_size=0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("Expected invalid model size to fail.")


def test_lstm_model_forward_shape() -> None:
    torch = pytest.importorskip("torch")
    from ai_orchestra.models import LstmSignalNet

    model = LstmSignalNet()
    fake_batch = torch.randn(8, 10, 5)
    output = model(fake_batch)

    assert output.shape == (8, 10)


def test_lstm_model_output_is_finite() -> None:
    torch = pytest.importorskip("torch")
    from ai_orchestra.models import LstmSignalNet

    model = LstmSignalNet()
    fake_batch = torch.randn(8, 10, 5)
    output = model(fake_batch)

    assert torch.isfinite(output).all()


def test_lstm_model_rejects_invalid_sizes() -> None:
    pytest.importorskip("torch")
    from ai_orchestra.models import LstmSignalNet

    try:
        LstmSignalNet(hidden_size=0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("Expected invalid model size to fail.")
