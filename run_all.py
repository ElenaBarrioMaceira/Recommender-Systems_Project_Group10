"""
Reproduce all Part 1, Part 2 and Part 3 experiments and data characterization figures.

Usage:
    python run_all.py

Alternatively:
    python run_all.py --data-path data/ml-1m

Part 3 only:
    python run_all.py --part3-only --data-path data/ml-1m
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
PART2 = SRC / "part_2"


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


def run_script(script_name, *script_args):
    """Execute an existing project script and stop if it fails."""

    script_path = SRC / script_name

    print(f"\n{'=' * 60}", flush=True)
    print(f"Running: {script_name}", flush=True)
    print(f"{'=' * 60}", flush=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SRC)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    subprocess.run(
        [sys.executable, str(script_path), *map(str, script_args)],
        cwd=ROOT,
        env=env,
        check=True,
    )


def run_part2_script(script_name):
    """Execute Part 2 script using the existing runner."""
    run_script(f"part_2/{script_name}")


def run_part3_script(script_name, *script_args):
    """Execute Part 3 script with its command-line arguments."""
    run_script(f"part_3/{script_name}", *script_args)


def run_part3(dataset):
    """Reproduce Track A: MF-ALS baseline, MMR sweep and evaluation."""
    baseline_dir = ROOT / "results" / "part_3" / "mf_als_baseline"
    rankings_path = ROOT / "part3_mmr_results" / "mmr_rankings.csv"
    output_dir = ROOT / "results" / "part_3"

    print("\nPART 3 : MF-ALS BASELINE", flush=True)
    run_part3_script(
        "evaluate_mf_als_baseline.py",
        "--data-path",
        dataset,
        "--output-dir",
        baseline_dir,
    )

    print("\nPART 3 : MMR LAMBDA SWEEP", flush=True)
    run_part3_script("mmr_experiments.py")

    print("\nPART 3 : RANKING EVALUATION AND FIGURES", flush=True)
    run_part3_script(
        "evaluate_rankings.py",
        "--rankings",
        rankings_path,
        "--data-path",
        dataset,
        "--output-dir",
        output_dir,
    )


def print_part3_summary():
    print("\nPart 3 - Generated CSV files:")
    print("  - results/part_3/mf_als_baseline/rankings.csv")
    print("  - results/part_3/mf_als_baseline/track_a_metrics.csv")
    print("  - part3_mmr_results/mmr_rankings.csv")
    print("  - results/part_3/track_a_metrics.csv")

    print("\nPart 3 - Generated figures:")
    print("  - results/part_3/accuracy_diversity.png")
    print("  - results/part_3/metrics_by_lambda.png")


def main():

    parser = argparse.ArgumentParser(
        description="Reproduce all Part 1, Part 2 and Part 3 experiments."
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        help="Path to the extracted MovieLens 1M dataset.",
    )

    parser.add_argument(
        "--part3-only",
        action="store_true",
        help="Run only Part 3 Track A experiments and evaluation.",
    )

    args = parser.parse_args()

    # Resolve an explicit CLI path relative to the caller before changing cwd.
    if args.data_path is not None:
        args.data_path = args.data_path.expanduser().resolve()

    os.chdir(ROOT)

    # Configure dataset path
    if args.data_path is not None:
        os.environ["DATA_PATH"] = str(args.data_path.expanduser().resolve())

    data_path = os.environ.get("DATA_PATH")

    if not data_path:
        parser.error(
            "DATA_PATH is not configured. "
            "Use --data-path to specify the dataset location."
        )

    dataset = Path(data_path).expanduser().resolve()
    os.environ["DATA_PATH"] = str(dataset)

    required_files = [
        "ratings.dat",
        "movies.dat",
        "users.dat",
    ]

    missing_files = [
        filename for filename in required_files if not (dataset / filename).is_file()
    ]

    if missing_files:
        parser.error(
            f"Missing MovieLens files in {dataset}: {', '.join(missing_files)}"
        )

    print(
        "\nRECOMMENDER SYSTEMS - "
        + ("PART 3" if args.part3_only else "PARTS 1, 2 and 3")
    )
    print("Reproducing experiments and results")
    print(f"Dataset: {dataset}")

    if args.part3_only:
        run_part3(dataset)
        print_part3_summary()
        print("\nPART 3 COMPLETED SUCCESSFULLY")
        return

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
    # PART 2: HYPERPARAMETER TUNING
    # --------------------------------------------------

    print("\nPART 2 : HYPERPARAMETER TUNING", flush=True)

    run_part2_script("tune_content_based.py")
    run_part2_script("tune_mf_als.py")
    run_part2_script("run_user_knn_experiments.py")

    # --------------------------------------------------
    # PART 2: FINAL TEST EVALUATION
    # --------------------------------------------------

    print("\nPART 2 : FINAL TEST EVALUATION", flush=True)

    run_part2_script("evaluate_content_based_test.py")
    run_part2_script("evaluate_mf_als_test.py")
    run_part2_script("test_user_knn_final.py")

    # --------------------------------------------------
    # PART 2: ADDITIONAL ANALYSIS
    # --------------------------------------------------

    print("\nPART 2 : ADDITIONAL ANALYSES", flush=True)

    run_part2_script("part2_content_analysis.py")
    run_part2_script("part2_mf_als_analysis.py")
    run_part2_script("part2_user_knn_analysis.py")

    # --------------------------------------------------
    # PART 2: COMPARISON BETWEEN MODELS
    # --------------------------------------------------

    print("\nPART 2 : COMPARISON BETWEEN MODELS", flush=True)

    run_part2_script("statistical_tests.py")
    run_part2_script("disaggregated.py")
    run_part2_script("sparsity_comparison.py")

    # --------------------------------------------------
    # PART 3: BEYOND ACCURACY (TRACK A)
    # --------------------------------------------------

    run_part3(dataset)

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

    # --------------------------------------------------
    # PART 2 SUMMARY
    # --------------------------------------------------

    print("\nPart 2 - Generated CSV files:")

    print("\nHyperparameter tuning:")
    print("  - content_based_validation_results.csv")
    print("  - mf_als_tuning_results.csv")
    print("  - results/part_2/user_knn_validation_492_users.csv")

    print("\nFinal test evaluation:")
    print("  - content_based_test_results.csv")
    print("  - mf_als_test_results.csv")
    print("  - results/part_2/user_knn_test_results.csv")

    print("\nAdditional model analyses:")
    print("  - part2_content_results/")
    print("  - part2_mf_als_results/")
    print("  - part2_user_knn_results/")

    print("\nCross-model comparisons:")
    print("  - part2_statistical_results/paired_bootstrap_test.csv")
    print("  - part2_disaggregated_results/disaggregated_comparison_test.csv")
    print("  - part2_sparsity_results/sparsity_comparison_test.csv")

    print_part3_summary()


if __name__ == "__main__":
    main()
