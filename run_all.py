"""
Reproduce all Part 1 experiments and data characterization figures.

Usage:
    python run_all.py

Alternatively:
    python run_all.py --data-path data/ml-1m
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"


def generate_data_figures(output_dir):
    """Generate and save the existing dataset characterization figures."""

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    sys.path.insert(0, str(SRC))

    from data_analysis import load_data_with_charasteristics

    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data_with_charasteristics()
    data.print_summary()

    figures = [
        (
            "interactions_distribution.png",
            data.plot_interaction_distributions,
        ),
        (
            "rating_distribution.png",
            data.plot_rating_distribution,
        ),
        (
            "time_distribution.png",
            data.plot_time_distribution,
        ),
    ]

    original_show = plt.show

    try:
        plt.show = lambda *args, **kwargs: None

        for filename, plot_function in figures:
            plt.close("all")

            plot_function()

            figure = plt.gcf()
            destination = output_dir / filename

            figure.savefig(
                destination,
                dpi=150,
                bbox_inches="tight",
            )

            print(f"Saved figure: {destination}", flush=True)

            plt.close("all")

    finally:
        plt.show = original_show


def run_script(script_name):
    """Execute an existing project script and stop if it fails."""

    script_path = SRC / script_name

    print(f"\n{'=' * 60}", flush=True)
    print(f"Running: {script_name}", flush=True)
    print(f"{'=' * 60}", flush=True)

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=ROOT,
        env=os.environ.copy(),
        check=True,
    )


def main():

    parser = argparse.ArgumentParser(
        description="Reproduce all Part 1 experiments."
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        help="Path to the extracted MovieLens 1M dataset.",
    )

    args = parser.parse_args()

    os.chdir(ROOT)

    # Configure dataset path
    if args.data_path is not None:
        os.environ["DATA_PATH"] = str(
            args.data_path.expanduser().resolve()
        )

    data_path = os.environ.get("DATA_PATH")

    if not data_path:
        parser.error(
            "DATA_PATH is not configured. "
            "Use --data-path to specify the dataset location."
        )

    dataset = Path(data_path)

    required_files = [
        "ratings.dat",
        "movies.dat",
        "users.dat",
    ]

    missing_files = [
        filename
        for filename in required_files
        if not (dataset / filename).is_file()
    ]

    if missing_files:
        parser.error(
            f"Missing MovieLens files in {dataset}: "
            f"{', '.join(missing_files)}"
        )

    print("\nRECOMMENDER SYSTEMS - PART 1")
    print("Reproducing experiments and results")
    print(f"Dataset: {dataset}")

    # --------------------------------------------------
    # STEP 1: DATA CHARACTERIZATION
    # --------------------------------------------------

    print("\nSTEP 1: DATA CHARACTERIZATION", flush=True)

    generate_data_figures(ROOT / "figures")

    # --------------------------------------------------
    # STEP 2: ORIGINAL ITEMKNN VALIDATION
    # --------------------------------------------------

    print("\nSTEP 2: ORIGINAL ITEMKNN VALIDATION", flush=True)

    run_script("tune_item_knn.py")

    # --------------------------------------------------
    # STEP 3: PEARSON ADJUSTED VALIDATION
    # --------------------------------------------------

    print("\nSTEP 3: PEARSON ADJUSTED VALIDATION", flush=True)

    run_script("tune_item_knn_adjusted.py")

    # --------------------------------------------------
    # STEP 4: COSINE ADJUSTED VALIDATION
    # --------------------------------------------------

    print("\nSTEP 4: COSINE ADJUSTED VALIDATION", flush=True)

    run_script("tune_cosine_adjusted.py")

    # --------------------------------------------------
    # STEP 5: ITEMKNN RECOMMENDATION CHECKS
    # --------------------------------------------------

    print("\nSTEP 5: ITEMKNN RECOMMENDATION CHECKS", flush=True)

    run_script("check_item_knn.py")

    # --------------------------------------------------
    # STEP 6: SELECTED ITEMKNN TEST EVALUATION
    # --------------------------------------------------

    print("\nSTEP 6: SELECTED ITEMKNN TEST EVALUATION", flush=True)

    run_script("test_item_knn_final.py")

    # --------------------------------------------------
    # STEP 7: FINAL COMPARISON OF SIX MODELS
    # --------------------------------------------------

    print("\nSTEP 7: FINAL MODEL COMPARISON", flush=True)

    run_script("compare_models.py")

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("ALL STEPS COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nGenerated CSV files:")

    print("  - item_knn_validation_results.csv")
    print("  - item_knn_adjusted_validation_results.csv")
    print("  - cosine_adjusted_validation_results.csv")
    print("  - item_knn_test_results.csv")
    print("  - model_comparison_test.csv")

    print("\nGenerated figures:")

    print("  - figures/interactions_distribution.png")
    print("  - figures/rating_distribution.png")
    print("  - figures/time_distribution.png")


if __name__ == "__main__":
    main()