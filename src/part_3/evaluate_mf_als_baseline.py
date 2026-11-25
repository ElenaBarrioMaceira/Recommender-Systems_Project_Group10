import argparse
import os
from pathlib import Path

import pandas as pd

from evaluate_rankings import (
    RANKING_COLUMNS,
    TOP_K,
    evaluate_rankings,
    read_rankings,
)

from data_analysis import load_data_with_splits
from data_loader import load_movies
from evaluation import get_relevant_items, prepare_evaluation_data
from part_2.mf_als import MatrixFactorizationALS


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the original Part 2 MF-ALS top-10 rankings."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/ml-1m"),
        help="Path to the MovieLens 1M folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/part_3/mf_als_baseline"),
        help="Directory for the rankings and metrics CSV files.",
    )
    args = parser.parse_args()

    data_path = args.data_path.resolve()
    for filename in ("ratings.dat", "movies.dat", "users.dat"):
        if not (data_path / filename).is_file():
            parser.error(f"Missing MovieLens file: {data_path / filename}")

    os.environ["DATA_PATH"] = str(data_path)

    print("Loading the Part 2 data split...", flush=True)
    data = load_data_with_splits()
    train, test = data.train, data.test
    movies = load_movies(data_path)

    test_filtered, _ = prepare_evaluation_data(train, test)
    user_ids = sorted(get_relevant_items(test_filtered))
    print(f"Eligible test users: {len(user_ids)}", flush=True)

    model = MatrixFactorizationALS(
        n_factors=80,
        n_iterations=20,
        regularization=0.1,
        seed=42,
    )

    print("Training MF-ALS with the Part 2 settings...", flush=True)
    model.fit(train)

    print("Generating the original top-10 rankings...", flush=True)
    rows = []
    for user_id in user_ids:
        movie_ids = model.recommend(user_id, top_n=TOP_K)
        for rank, movie_id in enumerate(movie_ids, start=1):
            rows.append(("baseline", None, user_id, movie_id, rank))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rankings_path = args.output_dir / "rankings.csv"
    pd.DataFrame(rows, columns=RANKING_COLUMNS).to_csv(rankings_path, index=False)

    print("Evaluating the baseline with the Part 3 evaluator...", flush=True)
    rankings = read_rankings(rankings_path)
    metrics = evaluate_rankings(rankings, train, test, movies)

    metrics_path = args.output_dir / "track_a_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    print(metrics.to_string(index=False))
    print(f"\nRankings ({len(rows)} rows): {rankings_path.resolve()}")
    print(f"Metrics: {metrics_path.resolve()}")


if __name__ == "__main__":
    main()