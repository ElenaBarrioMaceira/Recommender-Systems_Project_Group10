from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "part2_sparsity_results"


def load_sparsity_results(model_name, path):
    results = pd.read_csv(path)

    required_columns = {
        "Train fraction",
        "Training interactions",
        "NDCG@10",
        "Precision@10",
        "Recall@10",
        "Catalogue Coverage",
    }

    missing = required_columns - set(results.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {path}: {missing}"
        )

    results["Model"] = model_name

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    models = {
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
    }

    all_results = []

    for model_name, path in models.items():
        results = load_sparsity_results(
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
        "Train fraction",
        "Training interactions",
        "Precision@10",
        "Recall@10",
        "NDCG@10",
        "Catalogue Coverage",
    ]

    combined = combined[columns]

    combined = combined.sort_values(
        by=["Train fraction", "Model"],
        ascending=[False, True],
    )

    print("\nSPARSITY ABLATION COMPARISON")
    print(combined.to_string(index=False))

    output_path = (
        OUTPUT_DIR / "sparsity_comparison_test.csv"
    )

    combined.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()