# AI Orchestra

This project trains neural networks to reconstruct a requested clean sine signal
from a noisy mixed signal.

The first implementation will stay intentionally small: generate signals, build
the fully connected dataset, train one simple fully connected model, verify that
it reconstructs signal windows, and plot predictions. RNN, LSTM, config polish,
and broader experiments come after that baseline works.

## Development

Planned workflow:

```powershell
uv run pytest
uv run ruff check .
```

`uv` is required for the project workflow.
