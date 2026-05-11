"""Run one named experiment variant and write its artifacts to disk."""

import argparse

from ai_orchestra import DEFAULTS, PipelineDefaults
from ai_orchestra.experiment_runner import (
    DEFAULT_EXPERIMENT_RESULTS_DIR,
    list_known_experiments,
    run_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one experiment variant from config/experiments.json.",
    )
    parser.add_argument("name", help="Experiment name (see --list).")
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the known experiment names and exit.",
    )
    parser.add_argument(
        "--signal-set-count",
        type=int,
        default=1,
        help="Override the number of signal sets used for this run.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_EXPERIMENT_RESULTS_DIR),
        help="Where to write the experiment artifacts.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Optionally override num_samples on the base config.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optionally override batch_size on the base config.",
    )
    parser.add_argument(
        "--device", default="cpu", help="Torch device to use (default: cpu).",
    )
    args = parser.parse_args()

    if args.list:
        for name in list_known_experiments():
            print(name)
        return

    base = _build_base_defaults(args.num_samples, args.batch_size)
    summary = run_experiment(
        args.name,
        base_defaults=base,
        results_dir=args.results_dir,
        signal_set_count=args.signal_set_count,
        device=args.device,
    )

    print(f"Experiment '{summary['name']}' wrote artifacts to {summary['output_dir']}")
    print("Overall test MSE:")
    for model_name, value in summary["overall_mse"].items():
        print(f"  {model_name}: {value:.4f}")


def _build_base_defaults(
    num_samples: int | None,
    batch_size: int | None,
) -> PipelineDefaults:
    base = DEFAULTS
    overrides: dict[str, int] = {}
    if num_samples is not None:
        overrides["num_samples"] = num_samples
    if batch_size is not None:
        overrides["batch_size"] = batch_size
    if not overrides:
        return base
    from dataclasses import replace

    merged = replace(base, **overrides)
    merged.validate()
    return merged


if __name__ == "__main__":
    main()
