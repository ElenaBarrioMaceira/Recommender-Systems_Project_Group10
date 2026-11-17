"""
Part 2 - Disaggregated analysis.

Combine the disaggregated evaluation results of:

- Content-based
- MF-ALS
- UserKNN

Dimensions:
1. User activity
2. Item popularity

All models must use the same segments,
user counts and thresholds.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

OUTPUT_DIR = ROOT_DIR / "part2_disaggregated_results"


MODELS = {
    "Content-based": (
        ROOT_DIR
        / "part2_content_results"
        / "content_disaggregated_test.csv"
    ),
    "MF-ALS": (
        ROOT_DIR
        / "part2_mf_als_results"
        / "mf_als_disaggregated_test.csv"
    ),
    "UserKNN": (
        ROOT_DIR
        / "part2_user_knn_results"
        / "user_knn_disaggregated_test.csv"
    ),
}


# --------------------------------------------------
# Load disaggregated results
# --------------------------------------------------

def load_disaggregated_results(model_name, path):

    results = pd.read_csv(path)

    required_columns = {
        "Dimension",
        "Segment",
        "Users",
        "Threshold",
        "Precision@10",
        "Recall@10",
        "NDCG@10",
    }

    missing = required_columns - set(results.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {path}: {missing}"
        )

    if results.duplicated(
        subset=["Dimension", "Segment"]
    ).any():
        raise ValueError(
            f"Duplicate segments in {path}"
        )

    if results["Users"].isna().any():
        raise ValueError(
            f"Missing user counts in {path}"
        )

    numeric_columns = [
        "Users",
        "Threshold",
        "Precision@10",
        "Recall@10",
        "NDCG@10",
    ]

    if not np.isfinite(
        results[numeric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            f"Invalid numeric values in {path}"
        )

    results.insert(
        0,
        "Model",
        model_name,
    )

    return results


# --------------------------------------------------
# Validate segments
# --------------------------------------------------

def validate_segments(all_results):

    reference = (
        all_results[0]
        .set_index(["Dimension", "Segment"])
        [["Users", "Threshold"]]
        .sort_index()
    )

    for results in all_results[1:]:

        current = (
            results
            .set_index(["Dimension", "Segment"])
            [["Users", "Threshold"]]
            .sort_index()
        )

        if not reference.index.equals(current.index):
            raise ValueError(
                "Models have different disaggregated segments."
            )

        pd.testing.assert_frame_equal(
            reference,
            current,
            check_dtype=False,
            check_exact=False,
            rtol=1e-10,
            atol=1e-10,
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

    print("\nLOADING DISAGGREGATED RESULTS")

    for model_name, path in MODELS.items():

        results = load_disaggregated_results(
            model_name,
            path,
        )

        all_results.append(results)

        print(
            f"{model_name}: {len(results)} segments"
        )

    # Check identical segmentation.
    validate_segments(all_results)

    print(
        "\nAll models use the same segments, "
        "user counts and thresholds."
    )

    combined = pd.concat(
        all_results,
        ignore_index=True,
    )

    columns = [
        "Model",
        "Dimension",
        "Segment",
        "Users",
        "Threshold",
        "Precision@10",
        "Recall@10",
        "NDCG@10",
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
        by=["Dimension", "Segment", "Model"]
    )

    combined["Model"] = combined["Model"].astype(str)

    print("\nDISAGGREGATED ANALYSIS")
    print(combined.to_string(index=False))

    output_path = (
        OUTPUT_DIR
        / "disaggregated_comparison_test.csv"
    )

    combined.to_csv(
        output_path,
        index=False,
    )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()