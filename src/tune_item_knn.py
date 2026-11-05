import pandas as pd

from data_analysis import load_data_with_splits
from item_knn import ItemKNN
from evaluation import prepare_evaluation_data, evaluate_recommender


SIMILARITIES = ["cosine", "pearson"]
K_VALUES = [10, 20, 50]
TOP_N = 10


def main():

    # Load the dataset and temporal splits
    data = load_data_with_splits()

    train = data.train
    validation = data.validation

    # Prepare the same evaluation data used by Most Popular
    validation_filtered, catalogue = prepare_evaluation_data(
        train, validation
    )

    print(f"Training ratings: {len(train)}")
    print(f"Evaluation users: {validation_filtered['UserID'].nunique()}")
    print(f"Catalogue size: {len(catalogue)}")

    results = []

    # Train once for each similarity method
    for similarity in SIMILARITIES:

        print(f"\nTraining ItemKNN with {similarity} similarity...")

        model = ItemKNN(
            k=K_VALUES[0],
            similarity=similarity
        )

        model.fit(train)

        # Evaluate different neighborhood sizes
        for neighbors in K_VALUES:

            model.k = neighbors

            print(f"Evaluating {similarity}, k={neighbors}...")

            metrics = evaluate_recommender(
                validation=validation_filtered,
                recommend_function=lambda user_id, n: model.recommend(
                    user_id,
                    top_n=n
                ),
                catalogue=catalogue,
                k=TOP_N
            )

            results.append({
                "Similarity": similarity,
                "Neighbors": neighbors,
                **metrics
            })

    # Display results
    results_df = pd.DataFrame(results)

    print("\nItemKNN validation results:")
    print(results_df.to_string(index=False))

    # Save results
    results_df.to_csv(
        "item_knn_validation_results.csv",
        index=False
    )

    print("\nResults saved to item_knn_validation_results.csv")


if __name__ == "__main__":
    main()