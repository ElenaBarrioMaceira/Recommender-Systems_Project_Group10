import pandas as pd

from data_analysis import load_data_with_splits

from evaluation import (
    prepare_evaluation_data,
    evaluate_recommender
)

from item_knn_adjusted import ItemKNNAdjusted


# Experimental configurations
MIN_PERIODS_VALUES = [4, 5]
K_VALUES = [10, 20, 50]

TOP_N = 10


def main():

    # 1. Load the same temporal splits
    data = load_data_with_splits()

    train = data.train
    validation = data.validation

    # 2. Prepare evaluation data
    validation_filtered, catalogue = (
        prepare_evaluation_data(
            train,
            validation
        )
    )

    print("\nITEMKNN ADJUSTED - VALIDATION")

    print(f"Training ratings: {len(train)}")

    print(
        "Evaluation users: "
        f"{validation_filtered['UserID'].nunique()}"
    )

    print(f"Catalogue size: {len(catalogue)}")

    results = []

    # 3. Test different Pearson thresholds
    for min_periods in MIN_PERIODS_VALUES:

        print(
            f"\nTraining adjusted Pearson "
            f"with min_periods={min_periods}..."
        )

        model = ItemKNNAdjusted(
            k=K_VALUES[0],
            min_periods=min_periods
        )

        model.fit(train)

        # 4. Test different neighborhood sizes
        for neighbors in K_VALUES:

            model.k = neighbors

            print(
                f"Evaluating adjusted Pearson: "
                f"min_periods={min_periods}, "
                f"k={neighbors}..."
            )

            metrics = evaluate_recommender(
                validation=validation_filtered,
                recommend_function=(
                    lambda user_id, n: model.recommend(
                        user_id,
                        top_n=n
                    )
                ),
                catalogue=catalogue,
                k=TOP_N
            )

            results.append({
                "Model": "ItemKNN Adjusted",
                "Similarity": "pearson",
                "Min Periods": min_periods,
                "Neighbors": neighbors,
                **metrics
            })

    # 5. Create results table
    results_df = pd.DataFrame(results)

    # 6. Sort by NDCG@10
    results_df = results_df.sort_values(
        by="NDCG@10",
        ascending=False
    )

    print("\nAdjusted ItemKNN validation results:")

    print(
        results_df.to_string(index=False)
    )

    # 7. Save results separately
    results_df.to_csv(
        "item_knn_adjusted_validation_results.csv",
        index=False
    )

    # 8. Show best configuration
    best = results_df.iloc[0]

    print("\nBEST ADJUSTED CONFIGURATION")

    print(
        f"min_periods: {best['Min Periods']}"
    )

    print(
        f"k: {best['Neighbors']}"
    )

    print(
        f"NDCG@10: {best['NDCG@10']:.6f}"
    )

    print(
        "\nResults saved to "
        "item_knn_adjusted_validation_results.csv"
    )


if __name__ == "__main__":
    main()