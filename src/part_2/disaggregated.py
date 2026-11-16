from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "part2_disaggregated_results"


def load_disaggregated_results(model_name, path):
    results = pd.read_csv(path)

    required_columns = {
        "Dimension",
        "Segment",
        "Users",
        "NDCG@10",
        "Precision@10",
        "Recall@10",
    }

    missing = required_columns - set(results.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {path}: {missing}"
        )

    results.insert(0, "Model", model_name)

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    models = {
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
    }

    all_results = []

    for model_name, path in models.items():

        results = load_disaggregated_results(
            model_name,
            path,
        )

        all_results.append(results)

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

    print("\nDISAGGREGATED ANALYSIS")
    print(combined.to_string(index=False))

    output_path = (
        OUTPUT_DIR / "disaggregated_comparison_test.csv"
    )

    combined.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()