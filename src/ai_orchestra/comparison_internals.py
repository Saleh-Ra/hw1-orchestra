"""Internal helpers shared by :mod:`ai_orchestra.comparison`."""

from typing import Any


def train_with_loss_history(
    model: Any,
    train_loader: Any,
    test_loader: Any,
    epochs: int,
    learning_rate: float,
    device: str | None,
) -> tuple[list[float], list[float]]:
    """Train ``model`` for ``epochs`` and return per-epoch train/test MSE."""
    from .training import create_adam_optimizer, evaluate_one_epoch, train_one_epoch

    optimizer = create_adam_optimizer(model, learning_rate)
    train_losses: list[float] = []
    test_losses: list[float] = []
    for _ in range(epochs):
        train_losses.append(train_one_epoch(model, train_loader, optimizer, device))
        test_losses.append(evaluate_one_epoch(model, test_loader, device))
    return train_losses, test_losses


def by_frequency(
    frequencies: tuple[float, ...],
    per_class: dict[int, float],
) -> dict[float, float]:
    """Re-key a per-class-index MSE map by the actual frequency value."""
    return {frequencies[class_index]: mse for class_index, mse in per_class.items()}
