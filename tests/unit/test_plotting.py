from ai_orchestra import DEFAULTS, PipelineDefaults, generate_signal_set
from ai_orchestra.plotting import plot_mixed_signal, plot_raw_signal_examples, plot_signal_stack


def test_plot_signal_stack_saves_file(tmp_path) -> None:
    signal_set = generate_signal_set(
        PipelineDefaults(num_samples=100),
        seed=DEFAULTS.random_seed,
    )
    output_path = tmp_path / "clean.png"

    saved_path = plot_signal_stack(
        signal_set.time_axis,
        signal_set.clean_signals,
        signal_set.frequencies_hz,
        output_path,
        "Clean Signals",
        max_samples=50,
    )

    assert saved_path == output_path
    assert output_path.exists()


def test_plot_mixed_signal_saves_file(tmp_path) -> None:
    signal_set = generate_signal_set(
        PipelineDefaults(num_samples=100),
        seed=DEFAULTS.random_seed,
    )
    output_path = tmp_path / "mixed.png"

    saved_path = plot_mixed_signal(
        signal_set.time_axis,
        signal_set.noisy_mix,
        output_path,
        "Noisy Mix",
        max_samples=50,
    )

    assert saved_path == output_path
    assert output_path.exists()


def test_plot_raw_signal_examples_saves_expected_files(tmp_path) -> None:
    signal_set = generate_signal_set(
        PipelineDefaults(num_samples=100),
        seed=DEFAULTS.random_seed,
    )

    saved_paths = plot_raw_signal_examples(signal_set, tmp_path)

    assert len(saved_paths) == 4
    assert {path.name for path in saved_paths} == {
        "clean_mixed_signal.png",
        "clean_signals.png",
        "noisy_mixed_signal.png",
        "noisy_signals.png",
    }
    assert all(path.exists() for path in saved_paths)
