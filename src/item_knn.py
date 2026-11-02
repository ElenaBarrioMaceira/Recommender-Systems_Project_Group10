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

            raise NotImplementedError(
                "Pearson similarity will be implemented next"
            )

        # 6. Exclude self-similarity
        np.fill_diagonal(self.sim_matrix, 0)

        return self