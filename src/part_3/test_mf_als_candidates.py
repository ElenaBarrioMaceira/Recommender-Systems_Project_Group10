from pathlib import Path
import sys
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from data_analysis import load_data_with_splits
from part_2.mf_als import MatrixFactorizationALS


def main():
    data = load_data_with_splits()
    train = data.train
    user_id = int(train["UserID"].iloc[0])

    # Train two models with the same hyperparameters and seed.
    params = dict(n_factors=80, n_iterations=20, regularization=0.1, seed=42)

    print("Training first MF-ALS model...")
    model1 = MatrixFactorizationALS(**params)
    model1.fit(train)

    candidates1 = model1.get_candidates(user_id, top_n=100)
    candidates_repeat = model1.get_candidates(user_id, top_n=100)

    print("Training second MF-ALS model...")
    model2 = MatrixFactorizationALS(**params)
    model2.fit(train)

    candidates2 = model2.get_candidates(user_id, top_n=100)

    # Check that every candidate list excludes movies rated in training.
    seen_movies = set(train.loc[train["UserID"] == user_id, "MovieID"])

    expected_count = min(100, len(set(train["MovieID"]) - seen_movies))

    assert expected_count > 0, "Selected user has no unseen movies!"

    for candidates in (candidates1, candidates_repeat, candidates2):
        assert len(candidates) == expected_count, "ERROR: Unexpected candidate count!"

        assert candidates["MovieID"].is_unique, "ERROR: Duplicate movies!"

        assert not set(candidates["MovieID"]).intersection(seen_movies), (
            "ERROR: Candidates contain movies already rated during training!"
        )

        assert np.isfinite(candidates["Score"]).all(), (
            "ERROR: Invalid prediction scores!"
        )

    print("No previously rated movies in candidates.")

    # Compare repeated calls and independently fitted models.
    for other in (candidates_repeat, candidates2):
        assert candidates1["MovieID"].tolist() == other["MovieID"].tolist(), (
            "ERROR: Candidate movie IDs or ranking differ!"
        )

        np.testing.assert_allclose(
            candidates1["Score"].to_numpy(),
            other["Score"].to_numpy(),
            rtol=1e-6,
            atol=1e-6,
            err_msg="ERROR: Prediction scores differ!",
        )

    print("Reproducibility checks passed!")
    print(f"\nTop {len(candidates1)} candidates for user {user_id}:")
    print(candidates1.to_string(index=False))


if __name__ == "__main__":
    main()
