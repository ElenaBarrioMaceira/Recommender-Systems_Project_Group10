import numpy as np
import pandas as pd
from lenskit.als import BiasedMFScorer
from lenskit.data import ItemList, from_interactions_df
from lenskit.pipeline import topn_pipeline
from lenskit.training import TrainingOptions


class MatrixFactorizationALS:
    def __init__(
        self,
        n_factors: int = 20,
        n_iterations: int = 15,
        regularization: float = 0.1,
        seed: int = 42,
    ):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.seed = seed
        self._pipeline = None
        self._known_users = set()
        self._known_movies = set()
        self._movie_ids = []
        self._user_seen_movies = {}

    def fit(self, train: pd.DataFrame) -> "MatrixFactorizationALS":
        dataset = from_interactions_df(
            train[["UserID", "MovieID", "Rating"]].copy(),
            user_col="UserID",
            item_col="MovieID",
            rating_col="Rating",
        )

        scorer = BiasedMFScorer(
            embedding_size=self.n_factors,
            epochs=self.n_iterations,
            regularization=self.regularization,
            user_embeddings="prefer",
        )

        self._pipeline = topn_pipeline(scorer)
        self._pipeline.train(dataset, options=TrainingOptions(rng=self.seed))

        self._movie_ids = list(pd.unique(train["MovieID"]))
        self._known_movies = set(self._movie_ids)
        self._known_users = set(train["UserID"])
        self._user_seen_movies = train.groupby("UserID")["MovieID"].agg(set).to_dict()

        return self

    def _score_user(self, user_id, movie_ids) -> np.ndarray:
        movie_ids = list(movie_ids)
        scored = self._pipeline.run(
            "scorer", query=user_id, items=ItemList(item_ids=movie_ids)
        )
        scored_df = scored.to_df().set_index("item_id")
        return scored_df.loc[movie_ids, "score"].to_numpy(dtype=float)

    def predict(self, data: pd.DataFrame) -> pd.Series:
        outputs = np.empty(len(data), dtype=float)
        for user_id, group in data.reset_index(drop=True).groupby("UserID", sort=False):
            outputs[group.index.to_numpy()] = self._score_user(
                user_id, group["MovieID"]
            )
        return pd.Series(outputs, index=data.index, name="Prediction")

    def recommend(self, user_id: int, top_n: int = 10) -> list[int]:
        if user_id not in self._known_users or top_n <= 0:
            return []
        seen = self._user_seen_movies.get(user_id, set())
        candidates = [movie for movie in self._movie_ids if movie not in seen]
        if not candidates:
            return []
        scores = self._score_user(user_id, candidates)
        order = np.lexsort((np.asarray(candidates), -scores))
        return [candidates[i] for i in order[:top_n]]

    def get_candidates(self, user_id: int, top_n: int = 100) -> pd.DataFrame:
        """Return the top N unseen movies and their prediction scores."""
        # This excludes movies already rated during training
        seen = self._user_seen_movies.get(user_id, set())
        candidates = [movie for movie in self._movie_ids if movie not in seen]

        if not candidates:
            return pd.DataFrame(columns=["MovieID", "Score"])

        scores = self._score_user(user_id, candidates)

        # Sort by descending score
        order = np.lexsort((np.asarray(candidates), -scores))
        order = order[:top_n]

        return pd.DataFrame(
            {"MovieID": [candidates[i] for i in order], "Score": scores[order]}
        )
