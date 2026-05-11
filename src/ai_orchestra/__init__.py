"""Tools for conditioned sine-signal reconstruction experiments."""

from .comparison import ModelComparison, compare_models
from .comparison_io import save_comparison_csv, save_comparison_json
from .comparison_plotting import plot_overall_mse, plot_per_frequency_mse
from .config import DEFAULT_CONFIG_PATH, load_pipeline_defaults
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
from .evaluation import compute_mae, compute_mse, evaluate_per_class_mse
from .experiment_runner import (
    DEFAULT_EXPERIMENT_RESULTS_DIR,
    list_known_experiments,
    run_experiment,
)
from .experiments import (
    DEFAULT_EXPERIMENTS_PATH,
    ExperimentSpec,
    apply_overrides,
    load_experiments,
)
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
from .splits import holdout_signal_set_split_indices

__version__ = "0.1.0"

__all__ = [
    "DEFAULTS",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_EXPERIMENTS_PATH",
    "DEFAULT_EXPERIMENT_RESULTS_DIR",
    "ExperimentSpec",
    "FullyConnectedSignalDataset",
    "MinimalFcPipelineResult",
    "ModelComparison",
    "PipelineDefaults",
    "SequenceSignalDataset",
    "SignalSet",
    "__version__",
    "apply_overrides",
    "build_fc_dataset",
    "compare_models",
    "compute_mae",
    "compute_mse",
    "create_data_loaders",
    "evaluate_per_class_mse",
    "examples_per_signal_set",
    "extract_window",
    "generate_signal_set",
    "generate_signal_sets",
    "holdout_signal_set_split_indices",
    "list_known_experiments",
    "load_experiments",
    "load_pipeline_defaults",
    "make_one_hot",
    "plot_fc_predictions",
    "plot_lstm_predictions",
    "plot_overall_mse",
    "plot_per_frequency_mse",
    "plot_rnn_predictions",
    "run_experiment",
    "run_minimal_fc_pipeline",
    "run_signal_generation",
    "save_comparison_csv",
    "save_comparison_json",
    "summarize_minimal_result",
    "total_examples",
    "train_fc_baseline",
    "train_lstm_baseline",
    "train_rnn_baseline",
    "train_test_split_indices",
    "valid_window_starts",
]
