"""LSTM baseline experiment over the sequence-format dataset."""

from .defaults import DEFAULTS, PipelineDefaults
from .sequence_baseline import SequenceBaselineResult, run_sequence_baseline

LstmBaselineResult = SequenceBaselineResult


def run_lstm_baseline(
    defaults: PipelineDefaults = DEFAULTS,
    signal_set_count: int | None = None,
    epochs: int = 3,
    learning_rate: float = 1e-3,
    hidden_size: int = 32,
    prediction_count: int = 4,
    seed: int = DEFAULTS.random_seed,
    device: str | None = None,
) -> SequenceBaselineResult:
    """Train the LSTM baseline and return losses plus sample predictions."""
    from .models import LstmSignalNet

    def make_lstm(feature_size: int) -> LstmSignalNet:
        return LstmSignalNet(input_size=feature_size, hidden_size=hidden_size)

    return run_sequence_baseline(
        model_factory=make_lstm,
        defaults=defaults,
        signal_set_count=signal_set_count,
        epochs=epochs,
        learning_rate=learning_rate,
        prediction_count=prediction_count,
        seed=seed,
        device=device,
    )
