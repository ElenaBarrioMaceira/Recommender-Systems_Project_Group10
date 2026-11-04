import time

from data_analysis import load_data_with_splits
from item_knn import ItemKNN


# Hyperparameters to test
K_VALUES = [10, 20, 50]
SIMILARITIES = ["cosine", "pearson"]

# Number of users for the initial test
N_TEST_USERS = 5


def main():

    # 1. Load the temporal train/validation/test splits
    data = load_data_with_splits()

    train = data.train
    validation = data.validation

    print("Training ratings:", len(train))
    print("Validation ratings:", len(validation))

    # 2. Select validation users who exist in training
    train_users = set(train["UserID"].unique())
    validation_users = set(validation["UserID"].unique())

    known_users = sorted(train_users.intersection(validation_users))

    print("Validation users known in training:", len(known_users))

    # Use a small sample while testing the pipeline
    selected_users = known_users[:N_TEST_USERS]

    # 3. Test each similarity method
    for similarity in SIMILARITIES:

        print(f"\n{'=' * 40}")
        print("Similarity:", similarity)
        print(f"{'=' * 40}")

        # Train once per similarity method
        model = ItemKNN(k=20, similarity=similarity)

        start_time = time.perf_counter()
        model.fit(train)
        training_time = time.perf_counter() - start_time

        print("Training time:", round(training_time, 2), "seconds")

        # 4. Test different k values
        for k in K_VALUES:

            model.k = k

            print(f"\n--- k = {k} ---")

            # 5. Generate recommendations for validation users
            for user_id in selected_users:

                recommendations = model.recommend(
                    user_id,
                    top_n=10
                )

                print(
                    f"User {user_id}: {recommendations}"
                )

                # TODO:
                # Evaluate recommendations using:
                # - Precision@10
                # - Recall@10
                # - NDCG@10
                #
                # These metrics will be added when
                # the evaluation module is available.


if __name__ == "__main__":
    main()