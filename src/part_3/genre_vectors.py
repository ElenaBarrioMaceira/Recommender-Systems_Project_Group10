import numpy as np
import pandas as pd


#Build binary genre vectors for MovieLens movies
"""
Parameters
    movies: pandas.DataFrame
        Movie metadata containing MovieID and Genres

Returns
    dict
        MovieID -> binary genre vector
"""
def build_genre_vectors(movies):
    #Mekr sure MovieID is available as a column
    if "MovieID" not in movies.columns:
        if movies.index.name == "MovieID":
            movies = movies.reset_index()
        else:
            raise ValueError("MovieID column not found")
    if "Genres" not in movies.columns:
        raise ValueError("Genres column not found")

    #Get the genre list for every movie
    movie_genres = {}

    for _, row in movies.iterrows():
        movie_id = int(row["MovieID"])
        genres = str(row["Genres"]).split("|")
        movie_genres[movie_id] = genres

    #Find every unique genre and sort them
    all_genres = sorted({genre for genres in movie_genres.values() for genre in genres})
    #Create a fixed position for each genre
    genre_to_idx = {genre: idx for idx, genre in enumerate(all_genres)}

    #Construct binary vectors
    genre_vectors = {}

    for movie_id, genres in movie_genres.items():
        vector = np.zeros(len(all_genres), dtype = float)

        for genre in genres:
            vector[genre_to_idx[genre]] = 1.0
        genre_vectors[movie_id] = vector

    return genre_vectors, all_genres