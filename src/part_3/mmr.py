import numpy as np

#Calculate cosine similarity between two genre vectors
def cosine_similarity(vector1, vector2):
    vector1 = np.asarray(vector1, dtype = float)
    vector2 = np.asarray(vector2, dtype = float)

    if vector1.shape != vector2.shape:
        raise ValueError("Genre vectors must have the shame shape")

    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(vector1, vector2) / (norm1 * norm2))

#Re-rank movies using Maximal Marginal Relevance (MMR)
"""
    Parameters:
    candidate_scores : dict
        MovieID -> predicted MF-ALS score
        The same candidate pool must be used for every lambda

    genre_vectors _ dict
        MovieID -> genre vector

    lambda_value : float
        Relevance-diversity trade-off parameter
        1.0 means relevance only
        0.0 means maximum emphasis on diversity

    top_n : int
        Number of movies to recommend

    Returns:
    list
        MovieIDs ordered by MMR score
"""
def mmr_rerank(candidate_scores, genre_vectors, lambda_value, top_n = 10):
    if not 0 <= lambda_value <= 1:
        raise ValueError("Lambda must be between 0 and 1")
    if top_n < 1:
        raise ValueError("top:n must be positive")
    if not candidate_scores:
        return []

    movie_ids = list(candidate_scores.keys())

    for movie_id in movie_ids:
        if movie_id not in genre_vectors:
            raise ValueError(f"Missing genre vector for movie {movie_id}")

    #Normalize relevance scores to [0, 1]
    #Min and max normalization preserves their original ranking
    scores = np.array([candidate_scores[movie_id] for movie_id in movie_ids], dtype = float)
    if not np.all(np.isfinite(scores)):
        raise ValueError("Candidate scores must be finite")
    
    minimum = np.min(scores)
    maximum = np.max(scores)
    if maximum > minimum:
        normalized = (scores - minimum) / (maximum - minimum)
    else:
        normalized = np.ones(len(scores))

    relevance = {movie_id : float(score) for movie_id, score in zip(movie_ids, normalized)}

    #Cache pairwise cosine similarities between candidate movies
    similarity_cache = {}

    def get_similarity(movie1, movie2):
        pair = tuple(sorted((movie1, movie2)))
        if pair not in similarity_cache:
            similarity_cache[pair] = cosine_similarity(genre_vectors[movie1], genre_vectors[movie2])

        return similarity_cache[pair]

    selected = []
    remaining = set(movie_ids)

    #Greedily select one movie at a time
    while remaining and len(selected) < top_n:
        best_movie = None
        best_score = float("-inf")

        for movie_id in sorted(remaining):
            if selected:
                maximum_similarity = max(get_similarity(movie_id, selected_movie) for selected_movie in selected)
            else:
                maximum_similarity = 0.0

            #MMR balances relevance and similarity penalties
            mmr_score = (lambda_value * relevance[movie_id] - (1 - lambda_value) * maximum_similarity)

            #Iterating in MovieID order resolves exact ties
            if mmr_score > best_score:
                best_score = mmr_score
                best_movie = movie_id

        selected.append(best_movie)
        remaining.remove(best_movie)

    return selected
