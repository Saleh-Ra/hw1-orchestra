import pytest


def test_run_signal_generation_returns_requested_sets() -> None:
    from ai_orchestra import DEFAULTS, PipelineDefaults, run_signal_generation

    defaults = PipelineDefaults(num_signal_sets=3, num_samples=100)

    signal_sets = run_signal_generation(
        defaults=defaults,
        signal_set_count=2,
        seed=DEFAULTS.random_seed,
    )

    assert len(signal_sets) == 2
    assert signal_sets[0].clean_mix.shape == (100,)


def test_build_fc_dataset_returns_lazy_dataset() -> None:
    from ai_orchestra import (
        DEFAULTS,
        PipelineDefaults,
        build_fc_dataset,
        run_signal_generation,
    )

    defaults = PipelineDefaults(num_samples=100)
    signal_sets = run_signal_generation(defaults, signal_set_count=1)

    dataset = build_fc_dataset(signal_sets, defaults)

    assert len(dataset) == 364
    assert dataset.class_count == len(DEFAULTS.frequencies_hz)


def test_run_minimal_fc_pipeline_returns_metrics_and_plots(tmp_path) -> None:
    pytest.importorskip("torch")
    from ai_orchestra import run_minimal_fc_pipeline, summarize_minimal_result

    result = run_minimal_fc_pipeline(output_dir=tmp_path)
    summary = summarize_minimal_result(result)

    assert len(summary["train_losses"]) == 5
    assert len(summary["test_losses"]) == 5
    assert set(summary["class_indices"]) == {0, 1, 2, 3}
    assert len(summary["plot_paths"]) == 5
    assert all(path.exists() for path in result.plot_paths)
