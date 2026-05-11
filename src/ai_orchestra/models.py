"""Neural network models for signal reconstruction."""

import torch
from torch import nn


class FullyConnectedSignalNet(nn.Module):
    """Simple baseline network for flat conditioned signal windows."""

    def __init__(
        self,
        input_size: int = 14,
        hidden_size: int = 64,
        output_size: int = 10,
    ) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0 or output_size <= 0:
            msg = "Model layer sizes must be positive."
            raise ValueError(msg)

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict one clean 10-sample target window."""
        return self.network(inputs)
