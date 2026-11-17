"""
Part 2 - Statistical testing.

Compare Content-based, MF-ALS and UserKNN using
paired bootstrap confidence intervals for NDCG@10.

All models must be evaluated on the same test users.
"""

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

OUTPUT_DIR = ROOT_DIR / "part2_statistical_results"

N_BOOTSTRAP = 2000
SEED = 42


MODELS = {
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
    "UserKNN": (
        ROOT_DIR
        / "part2_user_knn_results"
        / "user_knn_per_user_test.csv"
    ),
}


# --------------------------------------------------
# Load per-user results
# --------------------------------------------------

def load_user_scores(path):

    scores = pd.read_csv(path)

    required_columns = {
        "UserID",
        "NDCG@10",
    }

    missing = required_columns - set(scores.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {path}: {missing}"
        )

    if scores["UserID"].isna().any():
        raise ValueError(
            f"Missing UserIDs in {path}"
        )

    if scores["UserID"].duplicated().any():
        raise ValueError(
            f"Duplicate UserIDs in {path}"
        )

    if not np.isfinite(
        scores["NDCG@10"].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            f"Invalid NDCG@10 values in {path}"
        )

    return scores[
        ["UserID", "NDCG@10"]
    ].copy()


# --------------------------------------------------
# Paired bootstrap
# --------------------------------------------------

def paired_bootstrap(scores_a, scores_b):

    paired = scores_a.merge(
        scores_b,
        on="UserID",
        how="inner",
        suffixes=("_A", "_B"),
        validate="one_to_one",
    )

    # All users must be present in both models.
    if (
        len(paired) != len(scores_a)
        or len(paired) != len(scores_b)
    ):
        raise ValueError(
            "Models were not evaluated on identical test users."
        )

    if len(paired) < 2:
        raise ValueError(
            "Not enough users for paired bootstrap."
        )

    differences = (
        paired["NDCG@10_A"]
        - paired["NDCG@10_B"]
    ).to_numpy(dtype=float)

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
        "CI excludes zero": bool(
            ci_lower > 0 or ci_upper < 0
        ),
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    scores = {}

    print("\nLOADING PER-USER TEST RESULTS")

    for model_name, path in MODELS.items():

        model_scores = load_user_scores(path)

        scores[model_name] = model_scores

        print(
            f"{model_name}: "
            f"{len(model_scores)} users, "
            f"mean NDCG@10 = "
            f"{model_scores['NDCG@10'].mean():.6f}"
        )

    # Verify identical user cohorts.
    reference_users = set(
        scores["Content-based"]["UserID"]
    )

    for model_name, model_scores in scores.items():

        current_users = set(
            model_scores["UserID"]
        )

        if current_users != reference_users:
            raise ValueError(
                f"{model_name} has a different test cohort."
            )

    print("\nAll models use the same test users.")

    # Pairwise comparisons.
    results = []

    for name_a, name_b in combinations(
        MODELS.keys(),
        2,
    ):

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

    output_path = (
        OUTPUT_DIR
        / "paired_bootstrap_test.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()