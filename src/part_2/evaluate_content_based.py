from pathlib import Path
import sys

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

    model = ContentBased(
        weighting="user_mean",
        year_weight=2.0
    )

    print("Training Content-based model...")
    model.fit(train, movies)

    print("Evaluating on validation...")

    results = evaluate_recommender(
        validation=validation_filtered,
        recommend_function=lambda user_id, n: model.recommend(
            user_id=user_id,
            top_n=n
        ),
        catalogue=catalogue,
        k=10
    )

    print("\nCONTENT-BASED VALIDATION RESULTS")
    print("Weighting: user_mean")
    print("Year weight: 2.0")

    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()