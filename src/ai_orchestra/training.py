"""Minimal training and evaluation loops."""

from typing import Any


def create_adam_optimizer(
    model: Any,
    learning_rate: float = 1e-3,
) -> Any:
    """Create the default Adam optimizer for a model."""
    if learning_rate <= 0:
        msg = "Learning rate must be positive."
        raise ValueError(msg)

    import torch

    return torch.optim.Adam(model.parameters(), lr=learning_rate)


def train_one_epoch(
    model: Any,
    data_loader: Any,
    optimizer: Any,
    device: str | None = None,
) -> float:
    """Train a model for one epoch and return average MSE loss."""
    import torch

    training_device = _resolve_device(device)
    loss_fn = torch.nn.MSELoss()
    model.to(training_device)
    model.train()
    total_loss = 0.0
    total_items = 0

    for inputs, targets in data_loader:
        inputs = inputs.to(training_device)
        targets = targets.to(training_device)
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()
        batch_size = inputs.shape[0]
        total_loss += loss.item() * batch_size
        total_items += batch_size

    return _average_loss(total_loss, total_items)


def evaluate_one_epoch(
    model: Any,
    data_loader: Any,
    device: str | None = None,
) -> float:
    """Evaluate a model for one epoch and return average MSE loss."""
    import torch

    evaluation_device = _resolve_device(device)
    loss_fn = torch.nn.MSELoss()
    model.to(evaluation_device)
    model.eval()
    total_loss = 0.0
    total_items = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(evaluation_device)
            targets = targets.to(evaluation_device)
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            batch_size = inputs.shape[0]
            total_loss += loss.item() * batch_size
            total_items += batch_size

    return _average_loss(total_loss, total_items)


def _resolve_device(device: str | None) -> str:
    if device is not None:
        return device

    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _average_loss(total_loss: float, total_items: int) -> float:
    if total_items <= 0:
        msg = "Data loader must provide at least one item."
        raise ValueError(msg)
    return total_loss / total_items
