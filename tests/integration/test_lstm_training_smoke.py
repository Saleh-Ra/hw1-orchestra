import pytest


def test_tiny_lstm_training_smoke_run() -> None:
    torch = pytest.importorskip("torch")

    from ai_orchestra.models import LstmSignalNet
    from ai_orchestra.training import create_adam_optimizer, train_one_epoch

    inputs = torch.randn(8, 10, 5)
    targets = torch.randn(8, 10)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inputs, targets),
        batch_size=4,
    )
    model = LstmSignalNet()
    optimizer = create_adam_optimizer(model)

    loss = train_one_epoch(model, loader, optimizer, device="cpu")

    assert loss >= 0.0
    assert torch.isfinite(torch.tensor(loss))
