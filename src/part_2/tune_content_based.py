from pathlib import Path
import sys
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from data_analysis import load_data_with_splits

from evaluation import (
    prepare_evaluation_data,
    evaluate_recommender
)

from part_2.content_based import ContentBased


def main():

    print("Loading MovieLens 1M data...")

    data = load_data_with_splits()

    train = data.train
    validation = data.validation
    movies = data.movie_data

    validation_filtered, catalogue = prepare_evaluation_data(
        train,
        validation
    )

    year_weights = [0.5, 1.0, 2.0]

    all_results = []

    for year_weight in year_weights:

        print(f"\nEvaluating year_weight = {year_weight}")

        model = ContentBased(
            weighting="user_mean",
            year_weight=year_weight
        )

        model.fit(train, movies)

        results = evaluate_recommender(
            validation=validation_filtered,
            recommend_function=lambda user_id, n: model.recommend(
                user_id=user_id,
                top_n=n
            ),
            catalogue=catalogue,
            k=10
        )

        all_results.append({
            "weighting": "user_mean",
            "year_weight": year_weight,
            **results
        })

        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")

    results_df = pd.DataFrame(all_results)

    results_df = results_df.sort_values(
        by="NDCG@10",
        ascending=False
    )

    print("\nCONTENT-BASED TUNING RESULTS")
    print(results_df.to_string(index=False))

    results_df.to_csv(
        "content_based_year_tuning_results.csv",
        index=False
    )

    best = results_df.iloc[0]

    print("\nBEST CONFIGURATION")
    print("Weighting:", best["weighting"])
    print("Year weight:", best["year_weight"])
    print("Validation NDCG@10:", best["NDCG@10"])


if __name__ == "__main__":
    main()