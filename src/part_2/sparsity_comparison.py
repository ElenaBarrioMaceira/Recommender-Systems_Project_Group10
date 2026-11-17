"""
Part 2 - Sparsity ablation comparison.

Compare Content-based, MF-ALS and UserKNN
under three training-data fractions:

- 100%
- 50%
- 20%

The script combines existing experiment results.
It does not retrain the models.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

OUTPUT_DIR = ROOT_DIR / "part2_sparsity_results"


MODELS = {
    "Content-based": (
        ROOT_DIR
        / "part2_content_results"
        / "content_sparsity_test.csv"
    ),
    "MF-ALS": (
        ROOT_DIR
        / "part2_mf_als_results"
        / "mf_als_sparsity_test.csv"
    ),
    "UserKNN": (
        ROOT_DIR
        / "part2_user_knn_results"
        / "user_knn_sparsity_test.csv"
    ),
}


EXPECTED_FRACTIONS = {1.0, 0.5, 0.2}


# --------------------------------------------------
# Load sparsity results
# --------------------------------------------------

def load_sparsity_results(model_name, path):

    results = pd.read_csv(path)

    required_columns = {
        "Train fraction",
        "Training interactions",
        "Precision@10",
        "Recall@10",
        "NDCG@10",
        "Catalogue Coverage",
    }

    missing = required_columns - set(results.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {path}: {missing}"
        )

    if results["Train fraction"].duplicated().any():
        raise ValueError(
            f"Duplicate training fractions in {path}"
        )

    fractions = set(
        results["Train fraction"].to_numpy(dtype=float)
    )

    if fractions != EXPECTED_FRACTIONS:
        raise ValueError(
            f"Unexpected training fractions in {path}: "
            f"{fractions}"
        )

    numeric_columns = [
        "Train fraction",
        "Training interactions",
        "Precision@10",
        "Recall@10",
        "NDCG@10",
        "Catalogue Coverage",
    ]

    if not np.isfinite(
        results[numeric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            f"Invalid numeric values in {path}"
        )

    results["Model"] = model_name

    return results


# --------------------------------------------------
# Validate training subsets
# --------------------------------------------------

def validate_training_sizes(all_results):

    reference = (
        all_results[0]
        .set_index("Train fraction")
        ["Training interactions"]
        .sort_index()
    )

    for results in all_results[1:]:

        current = (
            results
            .set_index("Train fraction")
            ["Training interactions"]
            .sort_index()
        )

        pd.testing.assert_series_equal(
            reference,
            current,
            check_names=False,
            check_dtype=False,
        )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = []

    print("\nLOADING SPARSITY RESULTS")

    for model_name, path in MODELS.items():

        results = load_sparsity_results(
            model_name,
            path,
        )

        all_results.append(results)

        print(
            f"{model_name}: "
            f"{len(results)} training fractions"
        )

    # Verify identical training fractions and sizes.
    validate_training_sizes(all_results)

    print(
        "\nAll models use the same training "
        "fractions and interaction counts."
    )

    combined = pd.concat(
        all_results,
        ignore_index=True,
    )

    columns = [
        "Model",
        "Train fraction",
        "Training interactions",
        "Precision@10",
        "Recall@10",
        "NDCG@10",
        "Catalogue Coverage",
    ]

    combined = combined[columns]

    # Preserve model order.
    model_order = list(MODELS.keys())

    combined["Model"] = pd.Categorical(
        combined["Model"],
        categories=model_order,
        ordered=True,
    )

    combined = combined.sort_values(
        by=["Train fraction", "Model"],
        ascending=[False, True],
    )

    combined["Model"] = combined["Model"].astype(str)

    print("\nSPARSITY ABLATION COMPARISON")
    print(combined.to_string(index=False))

    output_path = (
        OUTPUT_DIR
        / "sparsity_comparison_test.csv"
    )

    combined.to_csv(
        output_path,
        index=False,
    )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()