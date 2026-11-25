import argparse
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from evaluation import (
    catalogue_coverage,
    get_relevant_items,
    ndcg_at_k,
    precision_at_k,
    prepare_evaluation_data,
    recall_at_k,
)

TOP_K = 10
RANKING_COLUMNS = ["variant", "lambda", "UserID", "MovieID", "rank"]
METRIC_LABELS = {
    "ndcg_at_10": "NDCG@10",
    "ild": "Intra-list diversity",
    "coverage": "Catalogue coverage",
    "gini": "Gini index",
    "calibration_error": "Calibration error",
}


def read_rankings(path):
    """Read one CSV row per recommended movie and validate each top-10 list."""
    rows = pd.read_csv(path)
    missing_columns = set(RANKING_COLUMNS) - set(rows.columns)
    if missing_columns:
        raise ValueError(f"Missing ranking columns: {sorted(missing_columns)}")

    rows = rows[RANKING_COLUMNS].copy()
    if rows.empty or not rows["variant"].isin(["baseline", "mmr"]).all():
        raise ValueError("Expected rankings with 'baseline' or 'mmr' variants.")

    for column in ("UserID", "MovieID", "rank"):
        values = pd.to_numeric(rows[column], errors="raise")
        if not np.isfinite(values).all() or not values.mod(1).eq(0).all():
            raise ValueError(f"{column} must contain integers.")
        rows[column] = values.astype(int)

    rows["lambda"] = pd.to_numeric(rows["lambda"], errors="raise")
    is_baseline = rows["variant"].eq("baseline")
    if not is_baseline.any() or rows.loc[is_baseline, "lambda"].notna().any():
        raise ValueError("The original ranking must use variant=baseline and blank lambda.")
    if not rows.loc[~is_baseline, "lambda"].between(0, 1).all():
        raise ValueError("Every MMR row must have a lambda between 0 and 1.")

    configurations = {}
    expected_ranks = list(range(1, TOP_K + 1))

    for (variant, weight), group in rows.groupby(["variant", "lambda"], dropna=False):
        rankings = {}
        for user_id, user_rows in group.groupby("UserID"):
            ordered = user_rows.sort_values("rank")
            movie_ids = ordered["MovieID"].tolist()
            if ordered["rank"].tolist() != expected_ranks or len(set(movie_ids)) != TOP_K:
                raise ValueError(
                    f"Expected {TOP_K} distinct movies with ranks 1 to {TOP_K} "
                    f"for user {user_id}, variant={variant}, lambda={weight}."
                )
            rankings[user_id] = movie_ids

        configurations[(variant, None if pd.isna(weight) else float(weight))] = rankings

    return configurations


def build_genre_vectors(movies, catalogue):
    """Build unit vectors for cosine distance and genre shares for calibration."""
    genres = movies.set_index("MovieID")["Genres"]
    genres = genres.reindex(sorted(catalogue)).fillna("")
    encoded = genres.str.get_dummies(sep="|")
    vectors = encoded.to_numpy(dtype=float)

    genre_counts = vectors.sum(axis=1)
    if (genre_counts == 0).any():
        missing_ids = encoded.index[genre_counts == 0].tolist()
        raise ValueError(f"Missing genres for catalogue movies: {missing_ids[:10]}")

    cosine_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    genre_shares = vectors / genre_counts[:, None]

    return (
        dict(zip(encoded.index, cosine_vectors)),
        dict(zip(encoded.index, genre_shares)),
    )


def intra_list_diversity(movie_ids, cosine_vectors):
    """Return the mean cosine distance across pairs in a ranking."""
    vectors = np.stack([cosine_vectors[movie_id] for movie_id in movie_ids])
    similarities = vectors @ vectors.T
    pairs = np.triu_indices(len(movie_ids), k=1)
    return float(np.clip(1 - similarities[pairs], 0, 1).mean())


def gini_index(rankings, catalogue):
    """Compute Gini over recommendation counts, including zero-count movies."""
    counts = pd.Series(
        [movie_id for movie_ids in rankings.values() for movie_id in movie_ids]
    ).value_counts()
    frequencies = np.sort(counts.reindex(sorted(catalogue), fill_value=0).to_numpy())
    size = len(frequencies)
    positions = np.arange(1, size + 1)
    weights = 2 * positions - size - 1
    return float(np.dot(weights, frequencies) / (size * frequencies.sum()))


def genre_distribution(movie_ids, genre_shares):
    return np.mean([genre_shares[movie_id] for movie_id in movie_ids], axis=0)


def calibration_error(movie_ids, profile, genre_shares):
    """Return the total variation distance between the two genre distributions."""
    recommended_profile = genre_distribution(movie_ids, genre_shares)
    return float(0.5 * np.abs(profile - recommended_profile).sum())


def evaluate_rankings(configurations, train, test, movies):
    """Evaluate every configuration on the same Part 2 test cohort."""
    test_filtered, catalogue = prepare_evaluation_data(train, test)
    relevant_by_user = get_relevant_items(test_filtered)
    if not relevant_by_user:
        raise ValueError("No eligible users in the test split.")

    cosine_vectors, genre_shares = build_genre_vectors(movies, catalogue)
    history = (
        train.loc[train["UserID"].isin(relevant_by_user)]
        .groupby("UserID")["MovieID"]
        .agg(list)
        .to_dict()
    )
    seen = {user_id: set(movie_ids) for user_id, movie_ids in history.items()}
    profiles = {
        user_id: genre_distribution(movie_ids, genre_shares)
        for user_id, movie_ids in history.items()
    }

    baseline = configurations[("baseline", None)]
    relevance_only = configurations.get(("mmr", 1.0))
    if relevance_only is not None and relevance_only != baseline:
        raise ValueError("MMR with lambda=1 must match the original rankings in order.")

    results = []
    for (variant, weight), rankings in configurations.items():
        if rankings.keys() != relevant_by_user.keys():
            missing = len(relevant_by_user.keys() - rankings.keys())
            extra = len(rankings.keys() - relevant_by_user.keys())
            raise ValueError(
                f"{variant}, lambda={weight}: {missing} missing users, "
                f"{extra} extra users compared with the Part 2 test cohort."
            )

        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        diversity_scores = []
        calibration_scores = []

        for user_id, movie_ids in rankings.items():
            movie_set = set(movie_ids)
            if not movie_set.issubset(catalogue):
                raise ValueError(f"Ranking contains an out-of-catalogue movie for {user_id}.")
            if movie_set.intersection(seen[user_id]):
                raise ValueError(f"Ranking contains a previously rated movie for {user_id}.")

            relevant_items = relevant_by_user[user_id]
            precision_scores.append(precision_at_k(movie_ids, relevant_items, TOP_K))
            recall_scores.append(recall_at_k(movie_ids, relevant_items, TOP_K))
            ndcg_scores.append(ndcg_at_k(movie_ids, relevant_items, TOP_K))
            diversity_scores.append(intra_list_diversity(movie_ids, cosine_vectors))
            calibration_scores.append(
                calibration_error(movie_ids, profiles[user_id], genre_shares)
            )

        results.append(
            {
                "variant": variant,
                "lambda": weight,
                "evaluated_users": len(rankings),
                "precision_at_10": np.mean(precision_scores),
                "recall_at_10": np.mean(recall_scores),
                "ndcg_at_10": np.mean(ndcg_scores),
                "ild": np.mean(diversity_scores),
                "coverage": catalogue_coverage(rankings, catalogue),
                "gini": gini_index(rankings, catalogue),
                "calibration_error": np.mean(calibration_scores),
            }
        )

    return pd.DataFrame(results).sort_values(["variant", "lambda"])


def save_plots(results, output_dir):
    baseline = results.loc[results["variant"].eq("baseline")].iloc[0]
    mmr = results.loc[results["variant"].eq("mmr")].sort_values("lambda")

    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    if not mmr.empty:
        ax.plot(mmr["ild"], mmr["ndcg_at_10"], "o-", label="MMR, ordered by λ")
        for weight, diversity, ndcg in mmr[["lambda", "ild", "ndcg_at_10"]].itertuples(
            index=False, name=None
        ):
            ax.annotate(
                f"λ={weight:g}",
                (diversity, ndcg),
                xytext=(6, 6),
                textcoords="offset points",
            )

    ax.scatter(
        baseline["ild"],
        baseline["ndcg_at_10"],
        marker="x",
        color="black",
        s=90,
        zorder=3,
        label="Original MF-ALS",
    )
    ax.set(xlabel="Intra-list diversity", ylabel="NDCG@10")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output_dir / "accuracy_diversity.png", dpi=180)
    plt.close(fig)

    if mmr.empty:
        return

    fig, axes = plt.subplots(3, 2, figsize=(9, 9), layout="constrained")
    for ax, (metric, label) in zip(axes.flat, METRIC_LABELS.items()):
        ax.plot(mmr["lambda"], mmr[metric], "o-", label="MMR")
        ax.axhline(baseline[metric], color="black", linestyle="--", label="Original")
        ax.set(xlabel="λ", ylabel=label)
        ax.grid(alpha=0.25)

    axes.flat[0].legend()
    axes.flat[-1].axis("off")
    fig.savefig(output_dir / "metrics_by_lambda.png", dpi=180)
    plt.close(fig)


def make_demo(output_dir):
    """Create synthetic rankings and train/test data for a quick check."""
    movies = pd.DataFrame(
        {
            "MovieID": range(1, 13),
            "Genres": ["Action"] * 6 + ["Comedy"] * 5 + ["Action|Comedy"],
        }
    )
    rating_columns = ["UserID", "MovieID", "Rating"]
    train = pd.DataFrame(
        [(1, 1, 4), (2, 12, 4)] + [(3, movie_id, 4) for movie_id in range(1, 13)],
        columns=rating_columns,
    )
    test = pd.DataFrame([(1, 11, 5), (2, 1, 5)], columns=rating_columns)

    baseline = {1: list(range(2, 12)), 2: list(range(1, 11))}
    reranked = {
        1: [12, 11, *range(2, 10)],
        2: [7, 1, 8, 2, 9, 3, 10, 4, 11, 5],
    }
    configurations = [
        ("baseline", None, baseline),
        ("mmr", 0.5, reranked),
        ("mmr", 1.0, baseline),
    ]
    rows = [
        (variant, weight, user_id, movie_id, rank)
        for variant, weight, rankings in configurations
        for user_id, movie_ids in rankings.items()
        for rank, movie_id in enumerate(movie_ids, start=1)
    ]
    path = output_dir / "demo_rankings.csv"
    pd.DataFrame(rows, columns=RANKING_COLUMNS).to_csv(path, index=False)
    return path, train, test, movies


def main():
    parser = argparse.ArgumentParser(description="Evaluate Part 3 Track A rankings.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true", help="Use a tiny synthetic dataset.")
    mode.add_argument("--rankings", type=Path, help="CSV with baseline and MMR rankings.")
    parser.add_argument("--data-path", type=Path, help="Path to the MovieLens 1M folder.")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if args.rankings is not None and args.data_path is None:
        parser.error("--data-path is required with --rankings.")

    output_dir = args.output_dir or Path(
        "results/part_3/demo" if args.demo else "results/part_3"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        rankings_path, train, test, movies = make_demo(output_dir)
    else:
        from data_analysis import load_data_with_splits
        from data_loader import load_movies

        data_path = args.data_path.resolve()
        os.environ["DATA_PATH"] = str(data_path)
        data = load_data_with_splits()
        train, test = data.train, data.test
        movies = load_movies(data_path)
        rankings_path = args.rankings

    configurations = read_rankings(rankings_path)
    results = evaluate_rankings(configurations, train, test, movies)
    results.to_csv(output_dir / "track_a_metrics.csv", index=False)
    save_plots(results, output_dir)

    print(results.to_string(index=False))
    print(f"\nSaved results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()