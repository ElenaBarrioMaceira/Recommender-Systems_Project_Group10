import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from data_analysis import load_data_with_splits
from evaluation import get_relevant_items, prepare_evaluation_data
from part_2.mf_als import MatrixFactorizationALS
from part_3.genre_vectors import build_genre_vectors
from part_3.mmr import mmr_rerank

TOP_CANDIDATES = 100
TOP_N = 10
LAMBDA_VALUES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
OUTPUT_FILE = ROOT_DIR / "part3_mmr_results" / "mmr_rankings.csv"
RANKING_COLUMNS = ["variant", "lambda", "UserID", "MovieID", "rank"]


def add_ranking_rows(rows, variant, lambda_value, user_id, recommendations):
    for rank, movie_id in enumerate(recommendations, start=1):
        rows.append(
            {
                "variant": variant,
                "lambda": lambda_value,
                "UserID": int(user_id),
                "MovieID": int(movie_id),
                "rank": rank,
            }
        )


def main():
    print("Loading MovieLens 1M data...", flush=True)
    data = load_data_with_splits()
    train, test = data.train, data.test

    test_filtered, catalogue = prepare_evaluation_data(train, test)
    evaluation_users = sorted(get_relevant_items(test_filtered))
    if len(evaluation_users) != 992:
        raise ValueError(f"Expected 992 test users, found {len(evaluation_users)}.")

    print(f"Evaluation users: {len(evaluation_users)}", flush=True)
    print(f"Catalogue size: {len(catalogue)}", flush=True)
    genre_vectors, _ = build_genre_vectors(data.movie_data)

    print("Training MF-ALS...", flush=True)
    model = MatrixFactorizationALS(
        n_factors=80,
        n_iterations=20,
        regularization=0.1,
        seed=42,
    )
    model.fit(train)

    print("Generating baseline and MMR rankings...", flush=True)
    rows = []

    for user_number, user_id in enumerate(evaluation_users, start=1):
        candidates = model.get_candidates(user_id, top_n=TOP_CANDIDATES)

        if len(candidates) != TOP_CANDIDATES:
            raise ValueError(f"Expected {TOP_CANDIDATES} candidates for user {user_id}.")
        if not candidates["MovieID"].is_unique:
            raise ValueError(f"Duplicate candidates for user {user_id}.")
        if not np.isfinite(candidates["Score"]).all():
            raise ValueError(f"Invalid candidate scores for user {user_id}.")

        candidate_ids = candidates["MovieID"].astype(int).tolist()
        if not set(candidate_ids).issubset(catalogue):
            raise ValueError(f"Candidate outside the training catalogue for user {user_id}.")

        candidate_scores = dict(zip(candidate_ids, candidates["Score"].astype(float)))
        original_top10 = candidate_ids[:TOP_N]

        add_ranking_rows(rows, "baseline", None, user_id, original_top10)

        for lambda_value in LAMBDA_VALUES:
            reranked = mmr_rerank(
                candidate_scores=candidate_scores,
                genre_vectors=genre_vectors,
                lambda_value=lambda_value,
                top_n=TOP_N,
            )
            if len(reranked) != TOP_N or len(set(reranked)) != TOP_N:
                raise ValueError(f"Invalid top 10 for user {user_id}, lambda={lambda_value}.")
            if not set(reranked).issubset(candidate_scores):
                raise ValueError(f"MMR selected a movie outside the pool for user {user_id}.")
            if lambda_value == 1.0 and reranked != original_top10:
                raise ValueError(f"Lambda=1 did not match the baseline for user {user_id}.")

            add_ranking_rows(rows, "mmr", lambda_value, user_id, reranked)

        if user_number % 100 == 0:
            print(f"Finished {user_number}/{len(evaluation_users)} users", flush=True)

    results = pd.DataFrame(rows, columns=RANKING_COLUMNS)
    rows_per_configuration = len(evaluation_users) * TOP_N
    expected_counts = {weight: rows_per_configuration for weight in LAMBDA_VALUES}

    baseline = results.loc[results["variant"].eq("baseline")]
    mmr = results.loc[results["variant"].eq("mmr")]
    if len(baseline) != rows_per_configuration or baseline["lambda"].notna().any():
        raise ValueError("Incorrect baseline export.")
    if mmr["lambda"].isna().any() or mmr["lambda"].value_counts().to_dict() != expected_counts:
        raise ValueError("Incorrect MMR export. Expected a complete ranking for every lambda.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved {len(results)} rows to {OUTPUT_FILE}", flush=True)
    print("All lambda = 1 checks passed.", flush=True)


if __name__ == "__main__":
    main()