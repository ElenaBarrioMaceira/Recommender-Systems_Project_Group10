import numpy as np

from data_analysis import load_data_with_splits
from evaluation import prepare_evaluation_data, get_relevant_items
from item_knn import ItemKNN


SIMILARITIES = ["cosine", "pearson"]
NEIGHBORS = 10
TOP_N = 10


def main():

    # Load training and validation data
    data = load_data_with_splits()

    train = data.train
    validation = data.validation

    # Use the same users and catalogue as the evaluation
    validation_filtered, catalogue = prepare_evaluation_data(
        train, validation
    )

    relevant_by_user = get_relevant_items(validation_filtered)
    user_ids = list(relevant_by_user.keys())

    print(f"Evaluation users: {len(user_ids)}")
    print(f"Catalogue size: {len(catalogue)}")

    for similarity in SIMILARITIES:

        print(f"\nChecking ItemKNN with {similarity} similarity...")

        model = ItemKNN(
            k=NEIGHBORS,
            similarity=similarity
        )

        model.fit(train)

        recommendation_lengths = []
        users_with_fewer_than_10 = 0
        total_hits = 0

        for user_id in user_ids:

            recommendations = model.recommend(
                user_id,
                top_n=TOP_N
            )

            recommendation_lengths.append(len(recommendations))

            if len(recommendations) < TOP_N:
                users_with_fewer_than_10 += 1

            # Verify that recommended movies belong to the catalogue
            assert set(recommendations).issubset(catalogue)

            # Verify that recommendations do not contain duplicates
            assert len(recommendations) == len(set(recommendations))

            # Verify that already-rated movies are excluded
            user_idx = model.user_to_idx[user_id]

            rated_indices = model.user_item.getrow(user_idx).indices
            rated_movies = set(model.movie_ids[rated_indices])

            assert not set(recommendations).intersection(rated_movies)

            # Count relevant recommendations
            relevant_items = relevant_by_user[user_id]

            hits = sum(
                movie_id in relevant_items
                for movie_id in recommendations
            )

            total_hits += hits

        print(f"Average recommendations per user: {np.mean(recommendation_lengths):.2f}")
        print(f"Minimum recommendations: {min(recommendation_lengths)}")
        print(f"Maximum recommendations: {max(recommendation_lengths)}")
        print(f"Users with fewer than 10: {users_with_fewer_than_10}")
        print(f"Total relevant recommendations: {total_hits}")

        print("All recommendation checks passed!")


if __name__ == "__main__":
    main()