"""Small overfit check for the first fully connected baseline."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .dataset import FullyConnectedSignalDataset
from .defaults import DEFAULTS, PipelineDefaults
from .signal_sets import generate_signal_set


@dataclass(frozen=True)
class OverfitResult:
    """Result summary from a tiny overfit run."""

    train_losses: list[float]
    prediction: NDArray[np.float64]
    target: NDArray[np.float64]
    plot_path: Path | None


def run_fc_overfit_check(
    defaults: PipelineDefaults | None = None,
    seed: int = DEFAULTS.random_seed,
    subset_size: int = 32,
    epochs: int = 50,
    learning_rate: float = 1e-2,
    output_path: str | Path | None = "results/figures/fc_overfit_prediction.png",
    device: str = "cpu",
) -> OverfitResult:
    """Train the FC model on a tiny subset to verify it can overfit."""
    run_defaults = defaults or PipelineDefaults(num_samples=100, batch_size=16)
    if subset_size <= 0:
        msg = "Subset size must be positive."
        raise ValueError(msg)
    if epochs <= 0:
        msg = "Epoch count must be positive."
        raise ValueError(msg)

    import torch
    from torch.utils.data import DataLoader, Subset

    from .models import FullyConnectedSignalNet
    from .training import create_adam_optimizer, train_one_epoch

    torch.manual_seed(seed)
    dataset = FullyConnectedSignalDataset(
        [generate_signal_set(run_defaults, seed=seed)],
        run_defaults,
    )
    subset_indices = list(range(min(subset_size, len(dataset))))
    loader = DataLoader(
        Subset(dataset, subset_indices),
        batch_size=run_defaults.batch_size,
        shuffle=True,
    )
    model = FullyConnectedSignalNet()
    optimizer = create_adam_optimizer(model, learning_rate)
    train_losses = [
        train_one_epoch(model, loader, optimizer, device=device) for _ in range(epochs)
    ]
    first_input, first_target = dataset[subset_indices[0]]

    model.eval()
    with torch.no_grad():
        prediction_tensor = model(first_input.unsqueeze(0).to(device)).squeeze(0).cpu()

    prediction = prediction_tensor.numpy().astype(np.float64)
    target = first_target.numpy().astype(np.float64)
    plot_path = _plot_prediction(prediction, target, output_path)
    return OverfitResult(train_losses, prediction, target, plot_path)


def _plot_prediction(
    prediction: NDArray[np.float64],
    target: NDArray[np.float64],
    output_path: str | Path | None,
) -> Path | None:
    if output_path is None:
        return None

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 4))
    sample_indices = np.arange(len(target))
    axis.plot(sample_indices, target, label="Target")
    axis.plot(sample_indices, prediction, label="Prediction")
    axis.set_title("FC Overfit Check Prediction")
    axis.set_xlabel("Window sample")
    axis.set_ylabel("Amplitude")
    axis.legend()
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output
