import pandas as pd
import numpy as np
import data_loader as data_loader
import models.data as data_models


def get_interactions_per_user(rating_data, user_data) -> pd.Series:
    """Number of interactions per user"""
    interactions_per_user = (
        rating_data.groupby("UserID").size().reindex(user_data["UserID"], fill_value=0)
    )
    return interactions_per_user


def get_interactions_per_item(rating_data, movie_data) -> pd.Series:
    """Number of interactions per item (movie)"""
    interactions_per_item = (
        rating_data.groupby("MovieID")
        .size()
        .reindex(movie_data["MovieID"], fill_value=0)
    )
    return interactions_per_item


def get_interactions_per_time(rating_data: pd.DataFrame) -> pd.Series:
    """Number of interactions per time"""
    timestamps = pd.to_datetime(rating_data["Timestamp"], unit="s", utc=True)
    counts = pd.Series(1, index=timestamps).resample("MS").sum().astype(int)
    counts.name = "Interactions"
    return counts


def get_time_characteristics(rating_data):
    timestamps = pd.to_datetime(rating_data["Timestamp"], unit="s", utc=True)
    first_interaction = timestamps.min()
    last_interaction = timestamps.max()
    time_span = last_interaction - first_interaction
    interactions_per_month = (
        pd.Series(1, index=pd.DatetimeIndex(timestamps.dropna()))
        .resample("MS")
        .sum()
        .astype(int)
    )
    return first_interaction, last_interaction, time_span, interactions_per_month


def get_user_characteristics(
    user_data, rating_data
) -> tuple[int, int, pd.DataFrame, pd.DataFrame]:
    number_of_users = len(user_data)
    number_of_active_users = rating_data.UserID.nunique()

    age_distribution = pd.DataFrame(
        {
            "Count": user_data["Age"].value_counts(dropna=False),
            "Percentage": user_data["Age"].value_counts(normalize=True, dropna=False)
            * 100,
        }
    ).sort_index()

    occupation_distribution = pd.DataFrame(
        {
            "Count": user_data["Occupation"].value_counts(dropna=False),
            "Percentage": user_data["Occupation"].value_counts(
                normalize=True, dropna=False
            )
            * 100,
        }
    ).sort_index()

    return (
        number_of_users,
        number_of_active_users,
        age_distribution,
        occupation_distribution,
    )


def get_movie_characteristics(movie_data, rating_data) -> tuple[int, int]:
    number_of_movies = len(movie_data)
    number_of_movies_reviewed = rating_data.MovieID.nunique()
    return number_of_movies, number_of_movies_reviewed


def get_rating_characteristics(rating_data) -> tuple[int, float, float, pd.DataFrame]:
    number_of_ratings = len(rating_data)
    mean_rating = rating_data.Rating.mean()
    median_rating = rating_data.Rating.median()
    counts = rating_data["Rating"].value_counts().sort_index()
    rating_distribution = pd.DataFrame(
        {
            "Count": counts,
            "Percentage": counts / number_of_ratings * 100,
        }
    )
    return number_of_ratings, mean_rating, median_rating, rating_distribution


def get_data_characteristics(rating_data, user_data, movie_data) -> tuple[float, float]:
    number_of_ratings = len(rating_data)
    number_of_users = len(user_data)
    number_of_movies = len(movie_data)
    density = number_of_ratings / (number_of_users * number_of_movies)
    sparsity = 100 - density
    return density, sparsity


def remove_unknown_items_from_data(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove validation and test rows whose items are absent from training."""
    known_items = set(train["MovieID"])

    val_before, test_before = len(val), len(test)
    val = val[val["MovieID"].isin(known_items)].copy()
    test = test[test["MovieID"].isin(known_items)].copy()

    print(f"Unknown items: removed {val_before - len(val)} validation rows")
    print(f"Unknown items: removed {test_before - len(test)} test rows")
    return val, test


def remove_low_activity_users_from_data(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, threshold: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Remove users with fewer than threshold training interactions."""
    counts = train.groupby("UserID").size()
    active_users = counts.index[counts >= threshold]

    train_before, val_before, test_before = len(train), len(val), len(test)
    train = train[train["UserID"].isin(active_users)].copy()
    val = val[val["UserID"].isin(active_users)].copy()
    test = test[test["UserID"].isin(active_users)].copy()

    print(f"Low-activity users: removed {train_before - len(train)} train rows")
    print(f"Low-activity users: removed {val_before - len(val)} validation rows")
    print(f"Low-activity users: removed {test_before - len(test)} test rows")
    return train, val, test


def remove_low_activity_items_from_data(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, threshold: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Remove items with fewer than threshold training interactions."""
    counts = train.groupby("MovieID").size()
    active_items = counts.index[counts >= threshold]

    train_before, val_before, test_before = len(train), len(val), len(test)
    train = train[train["MovieID"].isin(active_items)].copy()
    val = val[val["MovieID"].isin(active_items)].copy()
    test = test[test["MovieID"].isin(active_items)].copy()

    print(f"Low-activity items: removed {train_before - len(train)} train rows")
    print(f"Low-activity items: removed {val_before - len(val)} validation rows")
    print(f"Low-activity items: removed {test_before - len(test)} test rows")
    return train, val, test


def temporal_split(
    rating_data,
    train_ratio,
    val_ratio,
    remove_unknown_items: bool,
    remove_low_activity_users: bool,
    remove_low_activity_items: bool,
    low_activity_user_threshold: int,
    low_activity_item_threshold: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split ratings temporally into train, validation, and test.
    Set remove_unknown_items to True if you want that the validation
    and test sets do not contain movies absent from training data.

    Set remove_low_activity_users to True if you want to remove user with low activity from all data
    Set remove_low_activity_items to True if you want to remove items with low activity from all data
    For both, give a threshold
    """
    df = rating_data.sort_values("Timestamp").copy()
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    if remove_low_activity_users:
        train, val, test = remove_low_activity_users_from_data(
            train=train, val=val, test=test, threshold=low_activity_user_threshold
        )

    if remove_low_activity_items:
        train, val, test = remove_low_activity_items_from_data(
            train=train, val=val, test=test, threshold=low_activity_item_threshold
        )

    if remove_unknown_items:
        val, test = remove_unknown_items_from_data(train=train, val=val, test=test)

    return train, val, test


def load_data_with_charasteristics() -> data_models.DataWithCharasteristics:
    movie_data, user_data, rating_data = data_loader.load_data()
    (
        number_of_users,
        number_of_active_users,
        age_distribution,
        occupation_distribution,
    ) = get_user_characteristics(user_data=user_data, rating_data=rating_data)

    number_of_movies, number_of_movies_reviewed = get_movie_characteristics(
        movie_data=movie_data, rating_data=rating_data
    )
    number_of_ratings, mean_rating, median_rating, rating_distribution = (
        get_rating_characteristics(rating_data=rating_data)
    )
    density, sparsity = get_data_characteristics(
        rating_data=rating_data, user_data=user_data, movie_data=movie_data
    )
    interactions_per_user = get_interactions_per_user(
        rating_data=rating_data, user_data=user_data
    )
    interactions_per_item = get_interactions_per_item(
        rating_data=rating_data, movie_data=movie_data
    )
    interactions_per_time = get_interactions_per_time(rating_data)

    first_interaction, last_interaction, time_span, interactions_per_month = (
        get_time_characteristics(rating_data)
    )

    user_charasteristics = data_models.UserCharasteristics(
        number_of_users=number_of_users,
        number_of_active_users=number_of_active_users,
        age_distribution=age_distribution,
        occupation_distribution=occupation_distribution,
    )
    movie_charasteristics = data_models.MovieCharasteristics(
        number_of_movies=number_of_movies,
        number_of_movies_reviewed=number_of_movies_reviewed,
    )
    rating_charasteristics = data_models.RatingCharasteristics(
        number_of_ratings=number_of_ratings,
        mean_rating=mean_rating,
        median_rating=median_rating,
        rating_distribution=rating_distribution,
    )
    data_charasteristics = data_models.DataCharasteristics(
        density=density, sparsity=sparsity
    )
    interactions = data_models.Interactions(
        interactions_per_user=interactions_per_user,
        interactions_per_item=interactions_per_item,
        interactions_per_time=interactions_per_time,
    )
    time_charasteristics = data_models.TimeCharasteristics(
        first_interaction=first_interaction,
        last_interaction=last_interaction,
        time_span=time_span,
        interactions_per_month=interactions_per_month,
    )
    data = data_models.DataWithCharasteristics(
        user_data=user_data,
        movie_data=movie_data,
        rating_data=rating_data,
        user_charasteristics=user_charasteristics,
        rating_charasteristics=rating_charasteristics,
        movie_charasteristics=movie_charasteristics,
        data_charasteristics=data_charasteristics,
        interactions=interactions,
        time_charasteristics=time_charasteristics,
    )
    return data


def load_data_with_splits(
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    remove_unknown_items: bool = False,
    remove_low_activity_users=False,
    remove_low_activity_items=False,
    low_activity_user_threshold=1,
    low_activity_item_threshold=1,
) -> data_models.DataForModeling:
    """Loads data that includes train/validation/test splits for modeling"""
    movie_data, user_data, rating_data = data_loader.load_data()
    train, validation, test = temporal_split(
        rating_data=rating_data,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        remove_unknown_items=remove_unknown_items,
        remove_low_activity_users=remove_low_activity_users,
        remove_low_activity_items=remove_low_activity_items,
        low_activity_user_threshold=low_activity_user_threshold,
        low_activity_item_threshold=low_activity_item_threshold,
    )
    data_for_modeling = data_models.DataForModeling(
        movie_data=movie_data,
        user_data=user_data,
        rating_data=rating_data,
        train=train,
        validation=validation,
        test=test,
    )
    return data_for_modeling


def data_summary():
    """Loads data with charasteristics and prints summaries and plots"""
    data = load_data_with_charasteristics()
    data.print_summary()
    data.plot_interaction_distributions()
    data.plot_rating_distribution()
    data.plot_time_distribution()
