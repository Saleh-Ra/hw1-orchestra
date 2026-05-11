"""Alternative train/test splits for stricter generalization checks.

The default :func:`ai_orchestra.dataset.train_test_split_indices` permutes
over the full ``(signal_set, window, class)`` index space, so train and
test share every signal set. The helper below partitions whole signal
sets instead, so a test signal set is never seen during training.
"""

import numpy as np
from numpy.typing import NDArray

from .dataset import examples_per_signal_set
from .defaults import DEFAULTS, PipelineDefaults


def holdout_signal_set_split_indices(
    num_signal_sets: int,
    defaults: PipelineDefaults = DEFAULTS,
    train_ratio: float = DEFAULTS.train_ratio,
    seed: int = DEFAULTS.random_seed,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Split example indices so each signal set lives in only one partition.

    With ``num_signal_sets=10`` and ``train_ratio=0.8`` this gives eight
    signal sets to the training partition and two to the test partition.
    Every window and class for a held-out set goes to the test partition,
    so the test partition contains examples whose underlying clean signals
    are unseen during training.
    """
    if num_signal_sets <= 1:
        msg = "Holdout split requires at least 2 signal sets."
        raise ValueError(msg)
    if not 0.0 < train_ratio < 1.0:
        msg = "Train ratio must be between 0 and 1."
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    set_order = rng.permutation(num_signal_sets)
    train_set_count = int(num_signal_sets * train_ratio)
    if train_set_count == 0 or train_set_count == num_signal_sets:
        msg = (
            "Train ratio must keep at least one signal set in each split. "
            f"Got train_ratio={train_ratio} for num_signal_sets={num_signal_sets}."
        )
        raise ValueError(msg)
    train_signal_sets = set(int(idx) for idx in set_order[:train_set_count])

    per_set = examples_per_signal_set(defaults)
    train_indices: list[int] = []
    test_indices: list[int] = []
    for set_index in range(num_signal_sets):
        start = set_index * per_set
        target = train_indices if set_index in train_signal_sets else test_indices
        target.extend(range(start, start + per_set))

    return (
        np.asarray(train_indices, dtype=np.int64),
        np.asarray(test_indices, dtype=np.int64),
    )
