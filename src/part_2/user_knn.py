import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix

class UserKNN:

    def __init__(self, k=20, similarity="cosine", significance_threshold=None):

        if k < 1:
            raise ValueError("k must be positive")

        if similarity not in ["cosine", "pearson"]:
            raise ValueError("Similarity must be cosine or pearson")

        if significance_threshold is not None and significance_threshold <= 0:
            raise ValueError("Significance threshold must be positive")

        self.k = k
        self.similarity = similarity
        self.significance_threshold = significance_threshold

    def fit(self, train):
        # 1. Get unique users and movies
        self.user_ids = np.sort(train["UserID"].unique())
        self.movie_ids = np.sort(train["MovieID"].unique())
    
        # 2. Create mappings from IDs to matrix indices
        self.user_to_idx = {
            user_id: idx
            for idx, user_id in enumerate(self.user_ids)
        }
    
        self.movie_to_idx = {
            movie_id: idx
            for idx, movie_id in enumerate(self.movie_ids)
        }
    
        # 3. Convert original IDs to matrix indices
        user_indices = train["UserID"].map(
            self.user_to_idx
        ).to_numpy()
    
        movie_indices = train["MovieID"].map(
            self.movie_to_idx
        ).to_numpy()
    
        ratings = train["Rating"].to_numpy(dtype=float)
    
        # 4. Build sparse user-item matrix
        self.user_item = csr_matrix(
            (ratings, (user_indices, movie_indices)),
            shape=(len(self.user_ids), len(self.movie_ids))
        )
        self.user_item.sort_indices()
        self.user_means = np.array([np.mean(self.user_item.getrow(user_idx).data) for user_idx in range(len(self.user_ids))])

        self.movie_user = self.user_item.tocsc()
        self.movie_user.sort_indices()

        #Used for storing the users who rated each movie and their corresponding ratings
        self.movie_raters = {}
        self.movie_ratings = {}
        for movie_idx in range(len(self.movie_ids)):
            start = self.movie_user.indptr[movie_idx]
            end = self.movie_user.indptr[movie_idx + 1]
            self.movie_raters[movie_idx] = self.movie_user.indices[start:end]
            self.movie_ratings[movie_idx] = self.movie_user.data[start:end]

        #Used for storing calculates similarities to avoid repeated calculations
        self.similarity_cache = {}
        return self

    def calculate_similarity(self, user1:int, user2: int) -> float:
        if user1 not in self.user_to_idx or user2 not in self.user_to_idx:
            raise ValueError("Unknown user")

        #Use the same key regardless of user order
        pair = tuple(sorted((user1, user2)))
        #Return the stored similarity if already calclulated
        if pair in self.similarity_cache:
            return self.similarity_cache[pair]

        #Get the ratings of both users
        user1_idx = self.user_to_idx[user1]
        user2_idx = self.user_to_idx[user2]

        start = self.user_item.indptr[user1_idx]
        end = self.user_item.indptr[user1_idx + 1]
        start2 = self.user_item.indptr[user2_idx]
        end2 = self.user_item.indptr[user2_idx + 1]
        movies = self.user_item.indices[start:end]
        movies2 = self.user_item.indices[start2:end2]
        values = self.user_item.data[start:end]
        values2 = self.user_item.data[start2:end2]

        #Find common movies and their positions in both rating arrays
        common_movies, positions1, positions2 = np.intersect1d(movies, movies2, assume_unique = True, return_indices = True)
        if len(common_movies) < 2:
            self.similarity_cache[pair] = 0.0
            return 0.0

        #Get ratings for the common movies
        ratings_1 = values[positions1]
        ratings_2 = values2[positions2]

        if self.similarity == "pearson":
            if np.std(ratings_1) == 0 or np.std(ratings_2) == 0:
                return 0.0
            else:
                similarity = np.corrcoef(ratings_1, ratings_2)[0, 1]
        else:
            numerator = np.dot(ratings_1, ratings_2)
            denominator = np.linalg.norm(ratings_1) * np.linalg.norm(ratings_2)

            if denominator == 0:
                return 0.0
            else:
                similarity = numerator/denominator
        if self.significance_threshold is not None:
            weight = min(len(common_movies), self.significance_threshold) / self.significance_threshold
            similarity = similarity * weight
        self.similarity_cache[pair] = float(similarity)
        return float(similarity)

    def get_neighbours(self, user_id: int, movie_id: int) -> list[tuple[int, float]]:
        if user_id not in self.user_to_idx:
            raise ValueError("Unknown value")

        if movie_id not in self.movie_to_idx:
            raise ValueError("Unknown value")

        movie_idx = self.movie_to_idx[movie_id]

        #Find users who rated the candidate movie
        rated_users = self.movie_raters[movie_idx]

        neighbours = []
        for user_idx in rated_users:
            neighbour_id = int(self.user_ids[user_idx])

            #Exclude the target user
            if neighbour_id == user_id:
                continue

            similarity = self.calculate_similarity(user_id, neighbour_id)
            #Keep only users with positive similarity
            if similarity > 0:
                neighbours.append((neighbour_id, similarity))

        neighbours.sort(key=lambda x: (-x[1], x[0]))
        return neighbours[:self.k]

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        if user_id not in self.user_to_idx:
            raise ValueError("Unknown user")
        
        if movie_id not in self.movie_to_idx:
            raise ValueError("Unknown movie")

        #Get the target user's average rating
        user_idx = self.user_to_idx[user_id]
        user_mean = self.user_means[user_idx]

        #Select neighbours who rated the candidate movie
        neighbours = self.get_neighbours(user_id, movie_id)
        if not neighbours:
            return float("-inf")

        numerator = 0.0
        denominator = 0.0

        movie_idx = self.movie_to_idx[movie_id]
        #Calculate similarity-weighted rating deviations
        for neighbour_id, similarity in neighbours:
            neighbour_idx = self.user_to_idx[neighbour_id]
            neighbour_mean = self.user_means[neighbour_idx]
            neighbour_rating = self.user_item[neighbour_idx, movie_idx]
            numerator += similarity * (neighbour_rating - neighbour_mean)
            denominator += abs(similarity)

        if denominator == 0:
            return float("-inf")

        #Adjust the prediction using the target user's average
        prediction = user_mean + numerator / denominator
        return float(prediction)

    def score_items(self, user_id: int) -> np.ndarray:
        if user_id not in self.user_to_idx:
            raise ValueError("Unknown user")

        user_idx = self.user_to_idx[user_id]
        user_mean = self.user_means[user_idx]

        #Get movies already rated by the target user
        rated_indices = set(self.user_item.getrow(user_idx).indices)

        #Calculate similarities with other users
        similarities = np.zeros(len(self.user_ids))
        for neighbour_id in self.user_ids:
            if neighbour_id != user_id:
                neighbour_idx = self.user_to_idx[neighbour_id]
                similarities[neighbour_idx] = self.calculate_similarity(user_id, int(neighbour_id))
        
        #Initialize scores for all movies
        scores = np.full(len(self.movie_ids), -np.inf)
        for movie_idx in range(len(self.movie_ids)):
            if movie_idx in rated_indices:
                continue

            #Retrieve users and ratings directly from cached CSC columns
            rated_users = self.movie_raters[movie_idx]
            movie_ratings = self.movie_ratings[movie_idx]
            #Keep only users with positive similarity
            valid_mask = similarities[rated_users] > 0
            valid_users = rated_users[valid_mask]
            valid_ratings = movie_ratings[valid_mask]
            if len(valid_users) == 0:
                continue

            #Sort by similarity, breaking ties using UserIDç
            order = np.lexsort((self.user_ids[valid_users], -similarities[valid_users]))

            #Select the k most similar neighbours
            selected = order[:self.k]
            neighbours = valid_users[selected]
            ratings = valid_ratings[selected]
            weights = similarities[neighbours]
            neighbour_means = self.user_means[neighbours]
            numerator = np.sum(weights *(ratings - neighbour_means))
            denominator = np.sum(np.abs(weights))

            if denominator > 0:
                scores[movie_idx] = user_mean + numerator / denominator

        return scores

    def recommend(self, user_id: int, top_n: int = 10) -> list[int]:
        if user_id not in self.user_to_idx:
            raise ValueError("Unknown user")

        if top_n < 1:
            raise ValueError("top_n must be positive")

        #Calculate scores for all movies
        scores = self.score_items(user_id)
        #Keep only movies with valid predictions
        valid_indices = np.where(np.isfinite(scores))[0]

        #Rank by predicted score and break ties using MovieID
        sorted_indices = valid_indices[np.lexsort((self.movie_ids[valid_indices], -scores[valid_indices]))]
        #Select the top N movies
        top_indices = sorted_indices[:top_n]
        return [int(movie_id) for movie_id in self.movie_ids[top_indices]]