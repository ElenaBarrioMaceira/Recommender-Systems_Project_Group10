import os
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path

load_dotenv()


def load_movies(data_path: Path):
    """Load movie data"""
    movies = pd.read_csv(
        data_path / "movies.dat",
        sep="::",
        engine="python",
        encoding="latin-1",
        header=None,
        names=["MovieID", "Title", "Genres"],
    )
    return movies


def load_users(data_path: Path):
    """Load users data"""
    users = pd.read_csv(
        data_path / "users.dat",
        sep="::",
        engine="python",
        header=None,
        names=["UserID", "Gender", "Age", "Occupation", "ZipCode"],
        dtype={"ZipCode": str},
    )
    return users


def load_ratings(data_path: Path):
    """Load ratings data"""
    ratings = pd.read_csv(
        data_path / "ratings.dat",
        sep="::",
        engine="python",
        header=None,
        names=["UserID", "MovieID", "Rating", "Timestamp"],
    )
    return ratings


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_path = Path(os.environ.get("DATA_PATH"))
    movies: pd.DataFrame = load_movies(data_path)
    users: pd.DataFrame = load_users(data_path)
    ratings: pd.DataFrame = load_ratings(data_path)
    return movies, users, ratings
