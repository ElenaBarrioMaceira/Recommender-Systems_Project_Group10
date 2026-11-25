from pathlib import Path
import sys
import numpy as np
import pandas as pd

#Add src to the pythonh import path
SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from data_analysis import load_data_with_splits
from evaluation import prepare_evaluation_data
from part_2.mf_als import MatrixFactorizationALS
from part_3.genre_vectors import build_genre_vectors
from part_3.mmr import mmr_rerank

#----------------------------------------
#Configuration
#----------------------------------------
TOP_CANDIDATES = 100
TOP_N = 10
LAMBDA_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

OUTPUT_DIR = ROOT_DIR / "part3_mmr_results"
OUTPUT_FILE = OUTPUT_DIR / "mmr_rankings.csv"

#----------------------------------------
#Prepare evaluation users
#----------------------------------------

#Select the same warm-start evaluation users as Part 2
def get_evaluation_users(train, test):
    test_filtered, catalogue = prepare_evaluation_data(train, test)
    relevant = (test_filtered[test_filtered["Rating"] >= 4].groupby("UserID")["MovieID"].apply(set).to_dict())
    return sorted(relevant.keys()), set(catalogue)

#----------------------------------------
#Save ranking rows
#----------------------------------------

#Add one csv row per recommended movie
def add_ranking_rows(rows, variant, lambda_value, user_id, recommendations):
    for rank, movie_id in enumerate(recommendations, start = 1):
        rows.append({
            "variant": variant,
            "lambda": lambda_value,
            "UserID": int(user_id),
            "MovieID": int(movie_id),
            "rank": rank
        })

#----------------------------------------
#Main experiment
#----------------------------------------
def main():
    print("Loading MovieLens 1M data...", flush = True)

    data = load_data_with_splits()
    train = data.train
    test = data.test
    movies = data.movie_data

    #Use the same evaluation cohort as Part 2
    evaluation_users, catalogue = get_evaluation_users(train, test)

    print("Training ratings:", len(train))
    print("Evaluation users:", len(evaluation_users))
    print("Catalogue size:", len(catalogue))

    assert len(evaluation_users) == 992, ("Evaluation cohort differs from the expected 992 users")

    #Build genre-only representations
    print("\nBuilding genre vectors...", flush = True)

    genre_vectors, all_genres = build_genre_vectors(movies)

    print("Genres:", len(all_genres))
    print("Movies with genre vectors:", len(genre_vectors))

    #Train the selected MF-ALS model
    print("\nTraining MF-ALS...", flush = True)

    model = MatrixFactorizationALS(n_factors = 80, n_iterations = 20, regularization = 0.1, seed = 42)
    model.fit(train)

    print("MF-ALS training completed!", flush = True)

    #Store all recommendations
    results = []

    print("\nGenerating original and MMR rankings...", flush = True)

    for user_number, user_id in enumerate(evaluation_users, start = 1):
        user_id = int(user_id)
        #Retreive the top q00 candidates only once per user
        candidates = model.get_candidates(user_id, top_n = TOP_CANDIDATES)

        if len(candidates) != TOP_CANDIDATES:
            raise ValueError(f"Unexpected candidate count per user {user_id}")
        if not candidates["MovieID"].is_unique:
            raise ValueError(f"Duplicate candidate movies for user {user_id}")
        if not np.isfinite(candidates["Score"]).all():
            raise ValueError(f"Invalid MF-ALS scores for user {user_id}")

        candidate_ids = [int(movie_id) for movie_id in candidates["MovieID"]]

        if not set(candidate_ids).issubset(catalogue):
            raise ValueError(f"Candidates outside catalogue for user {user_id}")

        #Convert MF-ALS output into the format requires by MMR
        candidate_scores = {int(movie_id): float(score) for movie_id, score in zip(candidates["MovieID"], candidates["Score"])}
        #Original MF-ALS top 10
        original_top10 = candidate_ids[:TOP_N]
        #Store the original ranking as a reference
        add_ranking_rows(results, variant = "original", lambda_value = 1.0, user_id = user_id, recommendations = original_top10)
        
        #Apply MMR to the same candidate pool for every lambda
        for lambda_value in LAMBDA_VALUES:
            reranked = mmr_rerank(candidate_scores = candidate_scores, genre_vectors = genre_vectors, lambda_value = lambda_value, top_n = TOP_N)

            if len(reranked) != TOP_N:
                raise ValueError(f"Incorrect ranking length for user {user_id}")
            if len(set(reranked)) != TOP_N:
                raise ValueError(f"Duplicate recommendations for user {user_id}")
            if not set(reranked).issubset(candidate_scores):
                raise ValueError(f"MMR selected an invalid candidate for user {user_id}")
            if lambda_value == 1.0:
                if reranked != original_top10:
                    raise AssertionError(f"Lambda = 1 does not match MF-ALS for user {user_id}")

            #Save reranked recommendations
            add_ranking_rows(results, variant = "mmr", lambda_value = lambda_value, user_id = user_id, recommendations = reranked)

#----------------------------------------
#Export results
#----------------------------------------  
    OUTPUT_DIR.mkdir(parents = True, exist_ok = True)

    results_df = pd.DataFrame(results, columns = ["variant", "lamda", "UserID", "MovieID", "rank"])
    expected_rows = len(evaluation_users) * TOP_N * (len(LAMBDA_VALUES) + 1)

    assert len(results_df) == expected_rows, ("Incorrect number od output rows")

    results_df.to_csv(OUTPUT_FILE, index = False)

    print("\nFinal results")
    print("Users evaluated:", len(evaluation_users))
    print("Lambda values:", LAMBDA_VALUES)
    print("Total CSV rows:", len(results_df))
    print("Results saved to:", OUTPUT_FILE)
    print("\nAll lambda = 1 checks passed")
    print("MMR experiments completed succesfully!")

if __name__ == "__main__":
    main()

    

