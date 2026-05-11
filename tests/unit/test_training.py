import pytest


def test_create_adam_optimizer_rejects_invalid_learning_rate() -> None:
    pytest.importorskip("torch")
    from ai_orchestra.models import FullyConnectedSignalNet
    from ai_orchestra.training import create_adam_optimizer

    model = FullyConnectedSignalNet()

    try:
        create_adam_optimizer(model, learning_rate=0.0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("Expected invalid learning rate to fail.")


def test_train_and_evaluate_one_epoch_return_finite_losses() -> None:
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader, TensorDataset

    from ai_orchestra.models import FullyConnectedSignalNet
    from ai_orchestra.training import (
        create_adam_optimizer,
        evaluate_one_epoch,
        train_one_epoch,
    )

    inputs = torch.randn(16, 14)
    targets = torch.randn(16, 10)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=8)
    model = FullyConnectedSignalNet()
    optimizer = create_adam_optimizer(model)

    train_loss = train_one_epoch(model, loader, optimizer, device="cpu")
    test_loss = evaluate_one_epoch(model, loader, device="cpu")

    assert train_loss >= 0.0
    assert test_loss >= 0.0
    assert torch.isfinite(torch.tensor(train_loss))
    assert torch.isfinite(torch.tensor(test_loss))
