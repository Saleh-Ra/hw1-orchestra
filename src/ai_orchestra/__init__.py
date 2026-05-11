"""Tools for conditioned sine-signal reconstruction experiments."""

from .dataloaders import create_data_loaders
from .dataset import (
    FullyConnectedSignalDataset,
    examples_per_signal_set,
    extract_window,
    make_one_hot,
    total_examples,
    train_test_split_indices,
    valid_window_starts,
)
from .defaults import DEFAULTS, PipelineDefaults
from .sdk import (
    MinimalFcPipelineResult,
    build_fc_dataset,
    plot_fc_predictions,
    plot_lstm_predictions,
    plot_rnn_predictions,
    run_minimal_fc_pipeline,
    run_signal_generation,
    summarize_minimal_result,
    train_fc_baseline,
    train_lstm_baseline,
    train_rnn_baseline,
)
from .sequence_dataset import SequenceSignalDataset
from .signal_sets import SignalSet, generate_signal_set, generate_signal_sets

__version__ = "0.1.0"

__all__ = [
    "DEFAULTS",
    "FullyConnectedSignalDataset",
    "MinimalFcPipelineResult",
    "PipelineDefaults",
    "SequenceSignalDataset",
    "SignalSet",
    "__version__",
    "build_fc_dataset",
    "create_data_loaders",
    "examples_per_signal_set",
    "generate_signal_set",
    "generate_signal_sets",
    "extract_window",
    "make_one_hot",
    "plot_fc_predictions",
    "plot_lstm_predictions",
    "plot_rnn_predictions",
    "run_minimal_fc_pipeline",
    "run_signal_generation",
    "summarize_minimal_result",
    "total_examples",
    "train_fc_baseline",
    "train_lstm_baseline",
    "train_rnn_baseline",
    "train_test_split_indices",
    "valid_window_starts",
]

