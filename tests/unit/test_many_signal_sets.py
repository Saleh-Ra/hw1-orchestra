import numpy as np

from ai_orchestra import DEFAULTS, PipelineDefaults, generate_signal_sets


def test_generate_signal_sets_uses_default_count() -> None:
    defaults = PipelineDefaults(num_signal_sets=3, num_samples=100)
    signal_sets = generate_signal_sets(defaults=defaults, seed=DEFAULTS.random_seed)

    assert len(signal_sets) == defaults.num_signal_sets


def test_generate_signal_sets_accepts_count_override() -> None:
    defaults = PipelineDefaults(num_signal_sets=50, num_samples=100)
    signal_sets = generate_signal_sets(
        defaults=defaults,
        count=2,
        seed=DEFAULTS.random_seed,
    )

    assert len(signal_sets) == 2


def test_generated_signal_sets_keep_frequency_order() -> None:
    defaults = PipelineDefaults(num_signal_sets=2, num_samples=100)
    signal_sets = generate_signal_sets(defaults=defaults, seed=DEFAULTS.random_seed)

    assert all(signal_set.frequencies_hz == defaults.frequencies_hz for signal_set in signal_sets)


def test_generated_signal_sets_are_not_identical() -> None:
    defaults = PipelineDefaults(num_signal_sets=2, num_samples=100)
    first, second = generate_signal_sets(defaults=defaults, seed=DEFAULTS.random_seed)

    assert not np.allclose(first.amplitudes, second.amplitudes)
    assert not np.allclose(first.phases, second.phases)
    assert not np.allclose(first.noisy_mix, second.noisy_mix)


def test_generate_signal_sets_is_reproducible_from_base_seed() -> None:
    defaults = PipelineDefaults(num_signal_sets=2, num_samples=100)
    first_run = generate_signal_sets(defaults=defaults, seed=DEFAULTS.random_seed)
    second_run = generate_signal_sets(defaults=defaults, seed=DEFAULTS.random_seed)

    for first, second in zip(first_run, second_run, strict=True):
        assert np.allclose(first.amplitudes, second.amplitudes)
        assert np.allclose(first.phases, second.phases)
        assert np.allclose(first.clean_mix, second.clean_mix)
        assert np.allclose(first.noisy_mix, second.noisy_mix)


def test_small_signal_set_setting_keeps_shapes_small() -> None:
    defaults = PipelineDefaults(num_signal_sets=2, num_samples=100)
    signal_sets = generate_signal_sets(defaults=defaults, seed=DEFAULTS.random_seed)

    assert signal_sets[0].clean_signals.shape == (len(defaults.frequencies_hz), 100)
    assert signal_sets[0].noisy_mix.shape == (100,)
