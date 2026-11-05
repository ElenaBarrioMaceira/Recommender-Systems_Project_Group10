import pandas as pd

#Rank movies by their total number of ratings in the training set
def most_popular_all_ratings(train: pd.DataFrame) -> pd.Series:
    popularity = train.groupby("MovieID").size()
    popularity = popularity.sort_index().sort_values(ascending = False, kind = "stable")
    return popularity

#Recommend the N most popular movies not previouly rated by the user
def recommend_top_n(popularity : pd.Series, train: pd.DataFrame, user_id: int, n: int = 10) -> list[int]:
    rated_movies = set(train.loc[train["UserID"] == user_id, "MovieID"])
    recommendations = [movie_id for movie_id in popularity.index if movie_id not in rated_movies]
    return recommendations[:n]

#Rank movies by the number of positive ratings (4 or 5) in the training set
def most_popular_positive_ratings(train: pd.DataFrame) -> pd.Series:
    positive_ratings = train[train["Rating"] >= 4]
    popularity = positive_ratings.groupby("MovieID").size()
    popularity = popularity.sort_index().sort_values(ascending = False, kind = "stable")
    return popularity