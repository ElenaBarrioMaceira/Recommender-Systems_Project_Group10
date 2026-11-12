import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBased:

    def __init__(self, weighting="user_mean", year_weight=1.0):

        if weighting not in ["positive", "centered", "user_mean"]:
            raise ValueError("Invalid weighting method")

        if not np.isfinite(year_weight) or year_weight < 0:
            raise ValueError("year_weight must be non-negative and finite")

        self.weighting = weighting
        self.year_weight = year_weight

        self.movie_ids = None
        self.movie_features = None
        self.movie_id_to_index = None

        self.user_profiles = {}
        self.user_seen_movies = {}

    def fit(self, train, movies):

        # Use only movies appearing in training
        train_movie_ids = set(train["MovieID"].unique())

        movies = movies[
            movies["MovieID"].isin(train_movie_ids)
        ].copy()

        # 1. Multi-hot encoding of genres
        movies["GenreList"] = movies["Genres"].apply(
            lambda genres: genres.split("|")
        )

        mlb = MultiLabelBinarizer()

        genre_features = mlb.fit_transform(
            movies["GenreList"]
        ).astype(float)

        # 2. Extract release year
        movies["Year"] = pd.to_numeric(
            movies["Title"].str.extract(
                r"\((\d{4})\)$",
                expand=False
            ),
            errors="coerce"
        )

        # 3. Group release years into decades
        movies["Decade"] = movies["Year"] // 10 * 10

        decade_features = pd.get_dummies(
            movies["Decade"],
            dtype=float
        ).to_numpy()

        # 4. Apply release-year weight
        decade_features *= self.year_weight

        # 5. Combine metadata
        self.movie_features = np.hstack([
            genre_features,
            decade_features
        ])

        self.movie_ids = movies["MovieID"].to_numpy()

        self.movie_id_to_index = {
            movie_id: index
            for index, movie_id in enumerate(self.movie_ids)
        }

        # 6. Build user profiles
        self.user_profiles = {}
        self.user_seen_movies = {}

        for user_id, user_ratings in train.groupby("UserID"):

            self.user_seen_movies[user_id] = set(
                user_ratings["MovieID"]
            )

            user_ratings = user_ratings[
                user_ratings["MovieID"].isin(
                    self.movie_id_to_index
                )
            ]

            movie_indices = [
                self.movie_id_to_index[movie_id]
                for movie_id in user_ratings["MovieID"]
            ]

            ratings = user_ratings["Rating"].to_numpy(
                dtype=float
            )

            if self.weighting == "positive":
                weights = np.where(ratings >= 4, 1.0, 0.0)

            elif self.weighting == "centered":
                weights = ratings - 3.0

            else:
                weights = ratings - ratings.mean()

            user_movie_features = self.movie_features[
                movie_indices
            ]

            denominator = np.sum(np.abs(weights))

            if denominator > 0:
                user_profile = (
                    weights @ user_movie_features
                ) / denominator
            else:
                user_profile = np.zeros(
                    self.movie_features.shape[1]
                )

            self.user_profiles[user_id] = user_profile

        return self

    def recommend(self, user_id, top_n=10):

        if self.movie_features is None:
            raise RuntimeError("Fit the model first")

        if top_n <= 0:
            raise ValueError("top_n must be positive")

        if user_id not in self.user_profiles:
            return []

        user_profile = self.user_profiles[user_id]

        scores = cosine_similarity(
            user_profile.reshape(1, -1),
            self.movie_features
        ).flatten()

        seen_movies = self.user_seen_movies[user_id]

        candidate_indices = np.array([
            index
            for index, movie_id in enumerate(self.movie_ids)
            if movie_id not in seen_movies
        ], dtype=int)

        if len(candidate_indices) == 0:
            return []

        candidate_scores = scores[candidate_indices]
        candidate_ids = self.movie_ids[candidate_indices]

        ranking = np.lexsort((
            candidate_ids,
            -candidate_scores
        ))

        return candidate_ids[ranking[:top_n]].tolist()