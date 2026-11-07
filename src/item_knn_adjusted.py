import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix

from item_knn import ItemKNN


class ItemKNNAdjusted(ItemKNN):
    """
    Experimental ItemKNN using Pearson correlation and
    item-mean-adjusted rating predictions.

    The original ItemKNN implementation remains unchanged.
    """

    def __init__(self, k=20, min_periods=5):

        if k < 1:
            raise ValueError("k must be positive")

        if min_periods < 2:
            raise ValueError("min_periods must be at least 2")

        super().__init__(
            k=k,
            similarity="pearson"
        )

        self.min_periods = min_periods

    def fit(self, train):

        # 1. Get unique users and movies
        self.user_ids = np.sort(
            train["UserID"].unique()
        )

        self.movie_ids = np.sort(
            train["MovieID"].unique()
        )

        # 2. Create mappings
        self.user_to_idx = {
            user_id: idx
            for idx, user_id in enumerate(self.user_ids)
        }

        self.movie_to_idx = {
            movie_id: idx
            for idx, movie_id in enumerate(self.movie_ids)
        }

        # 3. Convert IDs to matrix indices
        user_indices = train["UserID"].map(
            self.user_to_idx
        ).to_numpy()

        movie_indices = train["MovieID"].map(
            self.movie_to_idx
        ).to_numpy()

        ratings = train["Rating"].to_numpy(
            dtype=float
        )

        # 4. Build sparse user-item matrix
        self.user_item = csr_matrix(
            (
                ratings,
                (user_indices, movie_indices)
            ),
            shape=(
                len(self.user_ids),
                len(self.movie_ids)
            )
        )

        # 5. Calculate each movie's average rating
        rating_sums = np.asarray(
            self.user_item.sum(axis=0)
        ).ravel()

        rating_counts = np.diff(
            self.user_item.tocsc().indptr
        )

        self.item_means = (
            rating_sums / rating_counts
        )

        # 6. Prepare ratings for Pearson
        ratings_matrix = self.user_item.toarray()

        ratings_matrix[
            ratings_matrix == 0
        ] = np.nan

        # 7. Calculate Pearson similarities
        self.sim_matrix = (
            pd.DataFrame(ratings_matrix)
            .corr(
                method="pearson",
                min_periods=self.min_periods
            )
            .fillna(0)
            .to_numpy()
        )

        # 8. Exclude self-similarity
        np.fill_diagonal(
            self.sim_matrix,
            0
        )

        return self

    def score_items(self, user_id):

        # 1. Check if the user exists
        if user_id not in self.user_to_idx:
            raise ValueError("Unknown user")

        # 2. Get the user's ratings
        user_idx = self.user_to_idx[user_id]

        user_ratings = self.user_item.getrow(
            user_idx
        )

        rated_indices = user_ratings.indices
        ratings = user_ratings.data

        # 3. Initialize candidate scores
        scores = np.full(
            len(self.movie_ids),
            -np.inf
        )

        rated_set = set(rated_indices)

        # 4. Evaluate every movie in the training catalogue
        for movie_idx in range(len(self.movie_ids)):

            # Exclude previously rated movies
            if movie_idx in rated_set:
                continue

            # Similarities to movies rated by this user
            similarities = self.sim_matrix[
                movie_idx,
                rated_indices
            ]

            # Keep positive Pearson correlations
            valid_neighbors = np.where(
                similarities > 0
            )[0]

            if len(valid_neighbors) == 0:
                continue

            # Select up to k most similar neighbors
            sorted_neighbors = valid_neighbors[
                np.argsort(
                    similarities[valid_neighbors]
                )
            ]

            neighbor_indices = sorted_neighbors[
                -self.k:
            ]

            neighbor_similarities = similarities[
                neighbor_indices
            ]

            neighbor_ratings = ratings[
                neighbor_indices
            ]

            # Get mean ratings of neighbor movies
            neighbor_movie_indices = rated_indices[
                neighbor_indices
            ]

            neighbor_means = self.item_means[
                neighbor_movie_indices
            ]

            # Calculate deviations from movie means
            rating_deviations = (
                neighbor_ratings - neighbor_means
            )

            # Weighted prediction of deviations
            numerator = np.sum(
                neighbor_similarities
                * rating_deviations
            )

            denominator = np.sum(
                np.abs(neighbor_similarities)
            )

            if denominator > 0:

                # Add the candidate movie's average rating
                prediction = (
                    self.item_means[movie_idx]
                    + numerator / denominator
                )

                # Keep predictions within the rating scale
                scores[movie_idx] = np.clip(
                    prediction,
                    1.0,
                    5.0
                )

        return scores