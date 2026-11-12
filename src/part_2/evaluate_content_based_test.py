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
    test = data.test
    movies = data.movie_data

    test_filtered, catalogue = prepare_evaluation_data(
        train,
        test
    )

    print("Training best Content-based model...")

    model = ContentBased(
        weighting="user_mean",
        year_weight=2.0
    )

    model.fit(train, movies)

    print("Evaluating on TEST...")

    results = evaluate_recommender(
        validation=test_filtered,
        recommend_function=lambda user_id, n: model.recommend(
            user_id=user_id,
            top_n=n
        ),
        catalogue=catalogue,
        k=10
    )

    print("\nCONTENT-BASED TEST RESULTS")

    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    results_df = pd.DataFrame([{
        "Model": "Content-based",
        "Weighting": "user_mean",
        "Year Weight": 2.0,
        **results
    }])

    results_df.to_csv(
        "content_based_test_results.csv",
        index=False
    )

    print("\nResults saved to content_based_test_results.csv")


if __name__ == "__main__":
    main()