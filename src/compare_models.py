import pandas as pd

from data_analysis import load_data_with_splits
from evaluation import prepare_evaluation_data, evaluate_recommender
from baselines import (
    most_popular_all_ratings,
    most_popular_positive_ratings,
    recommend_top_n,
)
from item_knn import ItemKNN


TOP_N = 10


def main():

    # 1. Load the dataset
    data = load_data_with_splits()

    train = data.train
    test = data.test

    # 2. Prepare test data
    test_filtered, catalogue = prepare_evaluation_data(
        train, test
    )

    print(f"Training ratings: {len(train)}")
    print(f"Eligible test users: {test_filtered['UserID'].nunique()}")
    print(f"Catalogue size: {len(catalogue)}")

    results = []

    # 3. Most Popular: all ratings
    print("\nEvaluating Most Popular (all ratings)...")

    popularity_all = most_popular_all_ratings(train)

    metrics_all = evaluate_recommender(
        validation=test_filtered,
        recommend_function=lambda user_id, n: recommend_top_n(
            popularity_all,
            train,
            user_id,
            n=n
        ),
        catalogue=catalogue,
        k=TOP_N
    )

    results.append({
        "Model": "Most Popular (all ratings)",
        **metrics_all
    })

    # 4. Most Popular: positive ratings
    print("\nEvaluating Most Popular (positive ratings)...")

    popularity_positive = most_popular_positive_ratings(train)

    metrics_positive = evaluate_recommender(
        validation=test_filtered,
        recommend_function=lambda user_id, n: recommend_top_n(
            popularity_positive,
            train,
            user_id,
            n=n
        ),
        catalogue=catalogue,
        k=TOP_N
    )

    results.append({
        "Model": "Most Popular (positive ratings)",
        **metrics_positive
    })

    # 5. Item-based kNN
    print("\nTraining ItemKNN...")

    model = ItemKNN(
        k=10,
        similarity="cosine"
    )

    model.fit(train)

    print("\nEvaluating ItemKNN...")

    metrics_knn = evaluate_recommender(
        validation=test_filtered,
        recommend_function=lambda user_id, n: model.recommend(
            user_id,
            top_n=n
        ),
        catalogue=catalogue,
        k=TOP_N
    )

    results.append({
        "Model": "ItemKNN (cosine, k=10)",
        **metrics_knn
    })

    # 6. Display results
    results_df = pd.DataFrame(results)

    print("\nFINAL MODEL COMPARISON:")
    print(results_df.to_string(index=False))

    # 7. Save results
    results_df.to_csv(
        "model_comparison_test.csv",
        index=False
    )

    print("\nResults saved to model_comparison_test.csv")


if __name__ == "__main__":
    main()