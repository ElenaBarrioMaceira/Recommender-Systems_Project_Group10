"""
Part 2 - Content-based additional analyses.

Includes:
1. User-level evaluation
2. Bootstrap confidence interval for NDCG@10
3. Disaggregated analysis
4. Sparsity ablation

Evaluation metrics are imported from Part 1.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# --------------------------------------------------
# Imports
# --------------------------------------------------

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent

sys.path.insert(0, str(SRC_DIR))

from data_analysis import load_data_with_splits

from evaluation import (
    prepare_evaluation_data,
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    catalogue_coverage,
)

from part_2.content_based import ContentBased


# --------------------------------------------------
# Configuration
# --------------------------------------------------

K = 10
SEED = 42
N_BOOTSTRAP = 2000

TRAIN_FRACTIONS = [1.0, 0.5, 0.2]

OUTPUT_DIR = ROOT_DIR / "part2_content_results"


# --------------------------------------------------
# Model training
# --------------------------------------------------

def train_model(train, movies):

    model = ContentBased(
        weighting="user_mean",
        year_weight=2.0,
    )

    model.fit(train, movies)

    return model


# --------------------------------------------------
# Fixed evaluation cohort
# --------------------------------------------------

def prepare_fixed_cohort(train, test):

    test_filtered, catalogue = prepare_evaluation_data(
        train,
        test,
    )

    relevant = (
        test_filtered[test_filtered["Rating"] >= 4]
        .groupby("UserID")["MovieID"]
        .apply(set)
        .to_dict()
    )

    return relevant, set(catalogue)


# --------------------------------------------------
# Per-user evaluation
# --------------------------------------------------

def evaluate_users(model, relevant, catalogue):

    rows = []
    recommendations = {}

    for user_id, relevant_movies in relevant.items():

        recs = model.recommend(
            user_id=user_id,
            top_n=K,
        )

        if len(recs) != len(set(recs)):
            raise ValueError(
                f"Duplicate recommendations for user {user_id}"
            )

        if not set(recs).issubset(catalogue):
            raise ValueError(
                f"Recommendations outside catalogue for user {user_id}"
            )

        recommendations[user_id] = recs

        rows.append({
            "UserID": user_id,
            "Precision@10": precision_at_k(
                recs,
                relevant_movies,
                K,
            ),
            "Recall@10": recall_at_k(
                recs,
                relevant_movies,
                K,
            ),
            "NDCG@10": ndcg_at_k(
                recs,
                relevant_movies,
                K,
            ),
        })

    scores = pd.DataFrame(rows)

    return scores, recommendations


# --------------------------------------------------
# Overall metrics
# --------------------------------------------------

def calculate_overall(scores, recommendations, catalogue):

    return {
        "Precision@10": scores["Precision@10"].mean(),
        "Recall@10": scores["Recall@10"].mean(),
        "NDCG@10": scores["NDCG@10"].mean(),
        "Catalogue Coverage": catalogue_coverage(
            recommendations,
            catalogue,
        ),
        "Evaluated users": len(scores),
    }


# --------------------------------------------------
# Bootstrap confidence interval
# --------------------------------------------------

def bootstrap_confidence_interval(
    scores,
    metric="NDCG@10",
    n_bootstrap=N_BOOTSTRAP,
    seed=SEED,
):

    values = scores[metric].to_numpy(dtype=float)

    if len(values) < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    bootstrap_means = np.empty(n_bootstrap)

    for i in range(n_bootstrap):

        sample = rng.choice(
            values,
            size=len(values),
            replace=True,
        )

        bootstrap_means[i] = sample.mean()

    lower, upper = np.quantile(
        bootstrap_means,
        [0.025, 0.975],
    )

    return lower, upper


# --------------------------------------------------
# Disaggregation: user activity
# --------------------------------------------------

def analyse_user_activity(scores, train):

    activity = train.groupby("UserID").size()

    counts = scores["UserID"].map(activity).fillna(0)

    threshold = counts.median()

    labelled = scores.copy()

    labelled["Segment"] = np.where(
        counts <= threshold,
        "Low activity",
        "High activity",
    )

    rows = []

    for segment, group in labelled.groupby("Segment"):

        rows.append({
            "Dimension": "User activity",
            "Segment": segment,
            "Users": len(group),
            "Threshold": threshold,
            "Precision@10": group["Precision@10"].mean(),
            "Recall@10": group["Recall@10"].mean(),
            "NDCG@10": group["NDCG@10"].mean(),
        })

    return rows


# --------------------------------------------------
# Disaggregation: item popularity
# --------------------------------------------------

def analyse_item_popularity(
    recommendations,
    relevant,
    train,
):

    popularity = train.groupby("MovieID").size()

    threshold = popularity.median()

    popular_items = set(
        popularity[popularity > threshold].index
    )

    long_tail_items = set(
        popularity[popularity <= threshold].index
    )

    groups = {
        "Popular": popular_items,
        "Long tail": long_tail_items,
    }

    rows = []

    for segment, items in groups.items():

        user_rows = []

        for user_id, relevant_movies in relevant.items():

            segment_relevant = relevant_movies & items

            if not segment_relevant:
                continue

            recs = recommendations[user_id]

            user_rows.append({
                "Precision@10": precision_at_k(
                    recs,
                    segment_relevant,
                    K,
                ),
                "Recall@10": recall_at_k(
                    recs,
                    segment_relevant,
                    K,
                ),
                "NDCG@10": ndcg_at_k(
                    recs,
                    segment_relevant,
                    K,
                ),
            })

        segment_scores = pd.DataFrame(user_rows)

        row = {
            "Dimension": "Item popularity",
            "Segment": segment,
            "Users": len(user_rows),
            "Threshold": threshold,
        }

        for metric in [
            "Precision@10",
            "Recall@10",
            "NDCG@10",
        ]:

            if len(segment_scores) > 0:
                row[metric] = segment_scores[metric].mean()
            else:
                row[metric] = np.nan

        rows.append(row)

    return rows


# --------------------------------------------------
# Sparsity: training subsets
# --------------------------------------------------

def create_train_subsets(train):

    rng = np.random.default_rng(SEED)

    shuffled_indices = rng.permutation(len(train))

    subsets = {}

    for fraction in TRAIN_FRACTIONS:

        if fraction == 1.0:

            subsets[fraction] = train.copy()

        else:

            sample_size = round(
                len(train) * fraction
            )

            selected_indices = shuffled_indices[
                :sample_size
            ]

            subsets[fraction] = train.iloc[
                selected_indices
            ].copy()

    return subsets


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading MovieLens 1M data...")

    data = load_data_with_splits()

    train = data.train
    test = data.test
    movies = data.movie_data

    relevant, catalogue = prepare_fixed_cohort(
        train,
        test,
    )

    print("Training interactions:", len(train))
    print("Evaluated users:", len(relevant))
    print("Catalogue size:", len(catalogue))

    # --------------------------------------------------
    # 1. Full-data evaluation
    # --------------------------------------------------

    print("\nTraining full-data Content-based model...")

    model = train_model(train, movies)

    scores, recommendations = evaluate_users(
        model,
        relevant,
        catalogue,
    )

    scores.to_csv(
        OUTPUT_DIR / "content_per_user_test.csv",
        index=False,
    )

    overall = calculate_overall(
        scores,
        recommendations,
        catalogue,
    )

    ci_lower, ci_upper = bootstrap_confidence_interval(
        scores
    )

    overall.update({
        "Model": "Content-based",
        "Train fraction": 1.0,
        "Training interactions": len(train),
        "NDCG@10 CI lower": ci_lower,
        "NDCG@10 CI upper": ci_upper,
    })

    print("\nOVERALL TEST RESULTS")

    for key, value in overall.items():
        print(f"{key}: {value}")

    pd.DataFrame([overall]).to_csv(
        OUTPUT_DIR / "content_overall_with_ci_test.csv",
        index=False,
    )

    # --------------------------------------------------
    # 2. Disaggregated analysis
    # --------------------------------------------------

    print("\nCalculating disaggregated results...")

    disaggregated = (
        analyse_user_activity(scores, train)
        +
        analyse_item_popularity(
            recommendations,
            relevant,
            train,
        )
    )

    disaggregated_df = pd.DataFrame(disaggregated)

    disaggregated_df.to_csv(
        OUTPUT_DIR / "content_disaggregated_test.csv",
        index=False,
    )

    print(disaggregated_df.to_string(index=False))

    # --------------------------------------------------
    # 3. Sparsity ablation
    # --------------------------------------------------

    print("\nRunning sparsity ablation...")

    subsets = create_train_subsets(train)

    sparsity_results = []

    for fraction in TRAIN_FRACTIONS:

        print(f"\nTraining fraction: {fraction}")

        reduced_train = subsets[fraction]

        reduced_model = train_model(
            reduced_train,
            movies,
        )

        reduced_scores, reduced_recommendations = (
            evaluate_users(
                reduced_model,
                relevant,
                catalogue,
            )
        )

        metrics = calculate_overall(
            reduced_scores,
            reduced_recommendations,
            catalogue,
        )

        lower, upper = bootstrap_confidence_interval(
            reduced_scores
        )

        metrics.update({
            "Model": "Content-based",
            "Train fraction": fraction,
            "Training interactions": len(reduced_train),
            "NDCG@10 CI lower": lower,
            "NDCG@10 CI upper": upper,
        })

        sparsity_results.append(metrics)

        print(
            f"NDCG@10: {metrics['NDCG@10']:.4f}"
        )

    sparsity_df = pd.DataFrame(sparsity_results)

    sparsity_df.to_csv(
        OUTPUT_DIR / "content_sparsity_test.csv",
        index=False,
    )

    print("\nSPARSITY RESULTS")
    print(sparsity_df.to_string(index=False))

    print("\nAll Content-based analyses completed.")
    print("Results saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()