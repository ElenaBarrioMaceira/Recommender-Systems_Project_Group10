import numpy as np
import pandas as pd
from pydantic import BaseModel, PrivateAttr


class MatrixFactorizationALS(BaseModel):
    # Hyperparameters
    n_factors: int = 20
    n_iterations: int = 15
    regularization: float = 0.1
    seed: int = 42

    # Learned parameters are private, so Pydantic does not validate or serialize them.
    _mean: float | None = PrivateAttr(default=None)
    _user_factor_matrix: np.ndarray | None = PrivateAttr(default=None)
    _movie_factor_matrix: np.ndarray | None = PrivateAttr(default=None)
    _user_bias: np.ndarray | None = PrivateAttr(default=None)
    _movie_bias: np.ndarray | None = PrivateAttr(default=None)
    _user_id_to_index: dict = PrivateAttr(default_factory=dict)
    _movie_id_to_index: dict = PrivateAttr(default_factory=dict)
    _user_seen_movies: dict = PrivateAttr(default_factory=dict)

    def fit(self, train: pd.DataFrame) -> "MatrixFactorizationALS":
        """Fit using observed rows with columns UserID, MovieID, Rating."""
        columns = ["UserID", "MovieID", "Rating"]
        if train[columns].isna().any().any():
            raise ValueError(
                "UserID, MovieID and Rating cannot contain missing values."
            )

        # Turn ratings into a vector
        ratings = train["Rating"].to_numpy(dtype=float)

        # Construct dictionaries mapping each user/movie ID to its corresponding
        # row index in the user/movie latent factor matrix.
        user_codes, user_ids = pd.factorize(train["UserID"])
        movie_codes, movie_ids = pd.factorize(train["MovieID"])
        self._user_id_to_index = {uid: idx for idx, uid in enumerate(user_ids)}
        self._movie_id_to_index = {mid: idx for idx, mid in enumerate(movie_ids)}

        # Initialize the learnable parameters and the latent factor matrices
        n_users, n_movies = len(user_ids), len(movie_ids)
        rng = np.random.default_rng(self.seed)
        self._mean = float(ratings.mean())
        self._user_bias = np.zeros(n_users)
        self._movie_bias = np.zeros(n_movies)
        self._user_factor_matrix = rng.normal(0, 0.1, (n_users, self.n_factors))
        self._movie_factor_matrix = rng.normal(0, 0.1, (n_movies, self.n_factors))

        # Store the positions of each user's and each movie's observed ratings in the training dataframe
        user_rows = [np.where(user_codes == i)[0] for i in range(len(user_ids))]
        movie_rows = [np.where(movie_codes == i)[0] for i in range(len(movie_ids))]
        identity_matrix = np.eye(self.n_factors + 1)

        self._user_seen_movies = train.groupby("UserID")["MovieID"].agg(set).to_dict()

        for _ in range(self.n_iterations):
            # Fix movie parameters, then solve each user's bias and latent factors.
            for i, rows in enumerate(user_rows):
                movies = movie_codes[rows]
                w = self._solve(
                    rows,
                    movies,
                    ratings,
                    self._movie_factor_matrix,
                    self._movie_bias,
                    identity_matrix,
                )
                self._user_bias[i] = w[0]
                self._user_factor_matrix[i] = w[1:]

            # Fix user parameters, then solve each movie's bias and latent factors.
            for i, rows in enumerate(movie_rows):
                users = user_codes[rows]
                w = self._solve(
                    rows,
                    users,
                    ratings,
                    self._user_factor_matrix,
                    self._user_bias,
                    identity_matrix,
                )
                self._movie_bias[i] = w[0]
                self._movie_factor_matrix[i] = w[1:]

        return self

    # The prediction equation is r_ui = μ + b_u + b_i + p_u.T ⋅ q_i
    # During the user update step, we want to estimate the user bias b_u
    # and the user latent factors p_u while keeping movie parameters fixed.

    # The target vector y = r_ui - μ - b_i is constructed by subtracting
    # the known terms μ and b_i from the observed ratings, leaving the terms
    # we want to estimate on the right-hand side of the equation.

    # Matrix X: The first column contains ones to account for the user bias.
    # The subsequent columns contain the latent factors of the rated movies.

    # We then solve for w = [b_u, p_u1, p_u2, ... ,p_uk], which minimizes
    # the regularized squared error between Xw and the target vector y.
    def _solve(self, rows, indices, ratings, factors, biases, identity):
        X = np.column_stack((np.ones(len(rows)), factors[indices]))
        y = ratings[rows] - self._mean - biases[indices]

        penalty = self.regularization * len(rows) * identity

        return np.linalg.solve(X.T @ X + penalty, X.T @ y)

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Predict ratings for a DataFrame, e.g. a validation or test split."""
        users = data["UserID"].map(self._user_id_to_index).to_numpy(dtype=int)
        movies = data["MovieID"].map(self._movie_id_to_index).to_numpy(dtype=int)

        result = (
            self._mean
            + self._user_bias[users]
            + self._movie_bias[movies]
            + np.sum(
                self._user_factor_matrix[users] * self._movie_factor_matrix[movies],
                axis=1,
            )
        )

        return pd.Series(result, index=data.index, name="Prediction")

    def recommend(self, user_id: int, top_n: int = 10) -> list[int]:
        if user_id not in self._user_id_to_index:
            return []

        # All movies known to the model
        movie_ids = list(self._movie_id_to_index.keys())

        # Exclude movies the user rated during training
        seen_movies = self._user_seen_movies.get(user_id, set())

        candidate_movies = [
            movie_id for movie_id in movie_ids if movie_id not in seen_movies
        ]

        if not candidate_movies:
            return []

        # Predict ratings for candidate movies
        candidates = pd.DataFrame({"UserID": user_id, "MovieID": candidate_movies})

        scores = self.predict(candidates).to_numpy()

        # Rank movies by predicted rating (highest first)
        ranking = np.lexsort((np.array(candidate_movies), -scores))

        return [candidate_movies[i] for i in ranking[:top_n]]
