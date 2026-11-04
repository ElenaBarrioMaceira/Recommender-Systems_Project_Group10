import numpy as np

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


class ItemKNN:

    def __init__(self, k=20, similarity="cosine"):

        if k < 1:
            raise ValueError("k must be positive")

        if similarity not in ["cosine", "pearson"]:
            raise ValueError("Similarity must be cosine or pearson")

        self.k = k
        self.similarity = similarity

    def fit(self, train):

        # 1. Get unique users and movies
        self.user_ids = np.sort(train["UserID"].unique())
        self.movie_ids = np.sort(train["MovieID"].unique())

        # 2. Create mappings from IDs to matrix indices
        self.user_to_idx = {
            user_id: idx
            for idx, user_id in enumerate(self.user_ids)
        }

        self.movie_to_idx = {
            movie_id: idx
            for idx, movie_id in enumerate(self.movie_ids)
        }

        # 3. Convert original IDs to matrix indices
        user_indices = train["UserID"].map(
            self.user_to_idx
        ).to_numpy()

        movie_indices = train["MovieID"].map(
            self.movie_to_idx
        ).to_numpy()

        ratings = train["Rating"].to_numpy(dtype=float)

        # 4. Build sparse user-item matrix
        self.user_item = csr_matrix(
            (ratings, (user_indices, movie_indices)),
            shape=(len(self.user_ids), len(self.movie_ids))
        )

        # 5. Calculate item-item similarities
        if self.similarity == "cosine":

            self.sim_matrix = cosine_similarity(
                self.user_item.T,
                dense_output=True
            )

        elif self.similarity == "pearson":

    # 1. Convert the sparse matrix to a dense matrix
            ratings_matrix = self.user_item.toarray()

    # 2. Replace missing ratings with NaN
            ratings_matrix[ratings_matrix == 0] = np.nan

    # 3. Calculate item-item Pearson correlations
            n_items = len(self.movie_ids)
            self.sim_matrix = np.zeros((n_items, n_items))

            for i in range(n_items):
                for j in range(i + 1, n_items):

            # Users who rated both movies
                    common_users = (
                        ~np.isnan(ratings_matrix[:, i])
                        & ~np.isnan(ratings_matrix[:, j])
                    )

                    if np.sum(common_users) < 2:
                        continue

                    ratings_i = ratings_matrix[common_users, i]
                    ratings_j = ratings_matrix[common_users, j]

            # Pearson correlation
                    if np.std(ratings_i) > 0 and np.std(ratings_j) > 0:
                        similarity = np.corrcoef(ratings_i, ratings_j)[0, 1]

                        self.sim_matrix[i, j] = similarity
                        self.sim_matrix[j, i] = similarity

        # 6. Exclude self-similarity
        np.fill_diagonal(self.sim_matrix, 0)

        return self

    def score_items(self, user_id):

        # 1. Check if the user exists
        if user_id not in self.user_to_idx:
            raise ValueError("Unknown user")

        # 2. Get the user's ratings
        user_idx = self.user_to_idx[user_id]
        user_ratings = self.user_item.getrow(user_idx)

        # 3. Get the rated movies and their ratings
        rated_indices = user_ratings.indices
        ratings = user_ratings.data

        # 4. Initialize scores
        scores = np.full(len(self.movie_ids), -np.inf)

        # 5. Loop through candidate movies
        for movie_idx in range(len(self.movie_ids)):

            # Do not recommend already rated movies
            if movie_idx in rated_indices:
                continue

            # Similarities between the candidate and rated movies
            similarities = self.sim_matrix[movie_idx, rated_indices]

            # Keep only neighbors with positive similarity
            valid_neighbors = np.where(similarities > 0)[0]

            if len(valid_neighbors) == 0:
                continue

            # Select up to k most similar neighbors
            sorted_neighbors = valid_neighbors[
                np.argsort(similarities[valid_neighbors])
            ]

            neighbor_indices = sorted_neighbors[-self.k:]

            neighbor_similarities = similarities[neighbor_indices]
            neighbor_ratings = ratings[neighbor_indices]

            # Calculate the weighted prediction
            numerator = np.sum(
                neighbor_similarities * neighbor_ratings
            )

            denominator = np.sum(
                neighbor_similarities
            )

            if denominator > 0:
                scores[movie_idx] = numerator / denominator

        return scores

    def recommend(self, user_id, top_n=10):

        # 1. Calculate scores for all items
        scores = self.score_items(user_id)

        # 2. Find items with valid scores
        valid_indices = np.where(np.isfinite(scores))[0]

        # 3. Sort valid items by score (highest first)
        sorted_indices = valid_indices[
            np.argsort(scores[valid_indices])[::-1]
        ]

        # 4. Select the top N items
        top_indices = sorted_indices[:top_n]

        # 5. Convert matrix indices to original MovieIDs
        recommended_movies = self.movie_ids[top_indices]

        return recommended_movies