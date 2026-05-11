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
