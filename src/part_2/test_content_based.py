from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from data_analysis import load_data_with_splits
from part_2.content_based import ContentBased


def main():

    print("Loading MovieLens 1M data...")

    data = load_data_with_splits()

    train = data.train
    movies = data.movie_data

    print("Training ratings:", len(train))
    print("Movies:", len(movies))

    model = ContentBased(
        weighting="user_mean",
        year_weight=2.0
    )

    model.fit(train, movies)

    print("Model trained successfully!")

    user_id = int(train["UserID"].iloc[0])

    recommendations = model.recommend(
        user_id=user_id,
        top_n=10
    )

    print("\nRecommended MovieIDs:")
    print(recommendations)

    # Exactly 10 recommendations
    assert len(recommendations) == 10

    # No duplicates
    assert len(recommendations) == len(set(recommendations))

    # No previously rated movies
    seen_movies = set(
        train.loc[
            train["UserID"] == user_id,
            "MovieID"
        ]
    )

    assert not set(recommendations).intersection(seen_movies)

    # Recommendations belong to training catalogue
    catalogue = set(train["MovieID"])

    assert set(recommendations).issubset(catalogue)

    print("\nAll checks passed!")

    recommended_movies = (
        movies[
            movies["MovieID"].isin(recommendations)
        ][["MovieID", "Title", "Genres"]]
        .set_index("MovieID")
        .loc[recommendations]
        .reset_index()
    )

    print("\nRecommended movies:")
    print(recommended_movies.to_string(index=False))


if __name__ == "__main__":
    main()