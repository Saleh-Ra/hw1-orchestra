"""Run the current minimal FC pipeline with `python -m ai_orchestra`."""

from .sdk import run_minimal_fc_pipeline, summarize_minimal_result


def main() -> None:
    """Execute the small runnable baseline and print its result summary."""
    result = run_minimal_fc_pipeline()
    summary = summarize_minimal_result(result)
    print("Minimal FC pipeline complete.")
    print(f"Train losses: {summary['train_losses']}")
    print(f"Test losses: {summary['test_losses']}")
    print(f"Class indices: {summary['class_indices']}")
    print("Plots:")
    for path in summary["plot_paths"]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
