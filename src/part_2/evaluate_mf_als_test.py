from pathlib import Path
import sys
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from data_analysis import load_data_with_splits
from evaluation import prepare_evaluation_data, evaluate_recommender
from part_2.mf_als import MatrixFactorizationALS


def main():

    print("Loading MovieLens 1M data...")

    data = load_data_with_splits()

    train = data.train
    test = data.test

    test_filtered, catalogue = prepare_evaluation_data(train, test)

    model = MatrixFactorizationALS(
        n_factors=80,
        n_iterations=20,
        regularization=0.1,
        seed=42,
    )

    print("Training Matrix Factorization ALS...")
    model.fit(train)

    print("Evaluating on test...")

    results = evaluate_recommender(
        validation=test_filtered,
        recommend_function=lambda user_id, n: model.recommend(user_id=user_id, top_n=n),
        catalogue=catalogue,
        k=10,
    )

    print("\nMATRIX FACTORIZATION ALS TEST RESULTS")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    results_df = pd.DataFrame(
        [
            {
                "Model": "Matrix Factorization ALS",
                "Factors": 80,
                "Iterations": 20,
                "Regularization": 0.1,
                "Seed": 42,
                **results,
            }
        ]
    )

    results_df.to_csv("mf_als_test_results.csv", index=False)

    print("\nResults saved to mf_als_test_results.csv")


if __name__ == "__main__":
    main()
