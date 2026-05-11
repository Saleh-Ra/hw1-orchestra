import pytest


def test_tiny_fc_training_smoke_run() -> None:
    torch = pytest.importorskip("torch")

    from ai_orchestra.models import FullyConnectedSignalNet
    from ai_orchestra.training import create_adam_optimizer, train_one_epoch

    inputs = torch.randn(8, 14)
    targets = torch.randn(8, 10)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inputs, targets),
        batch_size=4,
    )
    model = FullyConnectedSignalNet()
    optimizer = create_adam_optimizer(model)

    loss = train_one_epoch(model, loader, optimizer, device="cpu")

    assert loss >= 0.0
    assert torch.isfinite(torch.tensor(loss))
