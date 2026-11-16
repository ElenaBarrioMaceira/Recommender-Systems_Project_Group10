from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "part2_statistical_results"

N_BOOTSTRAP = 2000
SEED = 42


def load_user_scores(path):
    """Load per-user NDCG@10 scores."""

    scores = pd.read_csv(path)

    required_columns = {"UserID", "NDCG@10"}

    if not required_columns.issubset(scores.columns):
        raise ValueError(f"Missing columns in {path}")

    if scores["UserID"].duplicated().any():
        raise ValueError(f"Duplicate UserIDs in {path}")

    return scores[["UserID", "NDCG@10"]]


def paired_bootstrap(scores_a, scores_b):
    """Compare two models using paired bootstrap over users."""

    paired = scores_a.merge(
        scores_b,
        on="UserID",
        how="inner",
        suffixes=("_A", "_B"),
        validate="one_to_one",
    )

    if len(paired) < 2:
        raise ValueError("Not enough common users for comparison")

    differences = (
        paired["NDCG@10_A"] - paired["NDCG@10_B"]
    ).to_numpy(dtype=float)

    if not np.isfinite(differences).all():
        raise ValueError("NDCG scores must be finite")

    mean_difference = differences.mean()

    rng = np.random.default_rng(SEED)
    bootstrap_means = np.empty(N_BOOTSTRAP)

    for i in range(N_BOOTSTRAP):
        sample = rng.choice(
            differences,
            size=len(differences),
            replace=True,
        )

        bootstrap_means[i] = sample.mean()

    ci_lower, ci_upper = np.quantile(
        bootstrap_means,
        [0.025, 0.975],
    )

    return {
        "Common users": len(paired),
        "Mean NDCG difference": mean_difference,
        "CI lower 95%": ci_lower,
        "CI upper 95%": ci_upper,
        "CI excludes zero": bool(ci_lower > 0 or ci_upper < 0),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    models = {
        "Content-based": (
            ROOT_DIR
            / "part2_content_results"
            / "content_per_user_test.csv"
        ),
        "MF-ALS": (
            ROOT_DIR
            / "part2_mf_als_results"
            / "mf_als_per_user_test.csv"
        ),
    }

    scores = {
        name: load_user_scores(path)
        for name, path in models.items()
    }

    results = []

    model_names = list(scores)

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):

            name_a = model_names[i]
            name_b = model_names[j]

            comparison = paired_bootstrap(
                scores[name_a],
                scores[name_b],
            )

            results.append({
                "Model A": name_a,
                "Model B": name_b,
                **comparison,
            })

    results_df = pd.DataFrame(results)

    print("\nPAIRED BOOTSTRAP RESULTS")
    print(results_df.to_string(index=False))

    output_path = OUTPUT_DIR / "paired_bootstrap_test.csv"

    results_df.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()