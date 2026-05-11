from math import tau

from ai_orchestra import DEFAULTS, PipelineDefaults


def test_default_signal_values_match_plan() -> None:
    assert DEFAULTS.frequencies_hz == (1.0, 3.0, 5.0, 7.0)
    assert DEFAULTS.sampling_frequency_hz == 1000
    assert DEFAULTS.num_samples == 10_000
    assert DEFAULTS.duration_seconds == 10.0
    assert DEFAULTS.window_size == 10


def test_default_training_values_match_plan() -> None:
    assert DEFAULTS.num_signal_sets == 50
    assert DEFAULTS.train_ratio == 0.8
    assert DEFAULTS.batch_size == 64
    assert isinstance(DEFAULTS.random_seed, int)
    assert DEFAULTS.epochs == 3
    assert DEFAULTS.learning_rate == 1e-3
    assert DEFAULTS.hidden_size == 32
    assert DEFAULTS.num_layers == 1


def test_default_randomization_ranges_are_valid() -> None:
    assert 0 < DEFAULTS.amplitude_range[0] < DEFAULTS.amplitude_range[1]
    assert DEFAULTS.phase_range == (0.0, tau)
    assert DEFAULTS.amplitude_noise_std >= 0.0
    assert DEFAULTS.phase_noise_std >= 0.0
    assert DEFAULTS.additive_noise_std >= 0.0


def test_defaults_validate() -> None:
    DEFAULTS.validate()


def test_defaults_can_be_overridden_for_small_runs() -> None:
    defaults = PipelineDefaults(num_signal_sets=2, num_samples=100, window_size=5)

    assert defaults.num_signal_sets == 2
    assert defaults.num_samples == 100
    assert defaults.window_size == 5
    defaults.validate()
