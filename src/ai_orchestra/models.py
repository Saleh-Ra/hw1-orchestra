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


class _SequenceSignalNet(nn.Module):
    """Shared base for sequence-format nets with a per-time-step linear head."""

    def __init__(self, recurrent_cell: nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.recurrent_cell = recurrent_cell
        self.output_head = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict one clean target window from a sequence input batch."""
        outputs, _ = self.recurrent_cell(inputs)
        return self.output_head(outputs).squeeze(-1)


def _validate_sequence_sizes(
    input_size: int,
    hidden_size: int,
    num_layers: int,
) -> None:
    if input_size <= 0 or hidden_size <= 0 or num_layers <= 0:
        msg = "Model layer sizes must be positive."
        raise ValueError(msg)


class RnnSignalNet(_SequenceSignalNet):
    """Simple recurrent network for sequence-format conditioned windows."""

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 32,
        num_layers: int = 1,
    ) -> None:
        _validate_sequence_sizes(input_size, hidden_size, num_layers)
        super().__init__(
            recurrent_cell=nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            ),
            hidden_size=hidden_size,
        )


class LstmSignalNet(_SequenceSignalNet):
    """LSTM network for sequence-format conditioned windows."""

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 32,
        num_layers: int = 1,
    ) -> None:
        _validate_sequence_sizes(input_size, hidden_size, num_layers)
        super().__init__(
            recurrent_cell=nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            ),
            hidden_size=hidden_size,
        )
