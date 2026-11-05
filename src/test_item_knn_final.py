import pandas as pd

from data_analysis import load_data_with_splits
from evaluation import prepare_evaluation_data, evaluate_recommender
from item_knn import ItemKNN


BEST_SIMILARITY = "cosine"
BEST_NEIGHBORS = 10
TOP_N = 10


def main():

    # Load data
    data = load_data_with_splits()

    train = data.train
    test = data.test

    # Prepare test data using the same evaluation protocol
    test_filtered, catalogue = prepare_evaluation_data(
        train, test
    )

    print(f"Training ratings: {len(train)}")
    print(f"Test ratings: {len(test)}")
    print(f"Eligible test users: {test_filtered['UserID'].nunique()}")
    print(f"Catalogue size: {len(catalogue)}")

    # Train selected model
    model = ItemKNN(
        k=BEST_NEIGHBORS,
        similarity=BEST_SIMILARITY
    )

    print("\nTraining ItemKNN...")
    model.fit(train)

    # Evaluate on test
    print("\nEvaluating ItemKNN on test...")

    metrics = evaluate_recommender(
        validation=test_filtered,
        recommend_function=lambda user_id, n: model.recommend(
            user_id,
            top_n=n
        ),
        catalogue=catalogue,
        k=TOP_N
    )

    # Display results
    print("\nFinal ItemKNN test results:")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")

    # Save results
    results = {
        "Model": "ItemKNN",
        "Similarity": BEST_SIMILARITY,
        "Neighbors": BEST_NEIGHBORS,
        **metrics
    }

    pd.DataFrame([results]).to_csv(
        "item_knn_test_results.csv",
        index=False
    )

    print("\nResults saved to item_knn_test_results.csv")


if __name__ == "__main__":
    main()