import math
import pandas as pd

#Calculate precision@k based on the number of relevant recommendations
def precision_at_k(recommendations: list[int], relevant_items: set[int], k: int = 10) -> float:
    top_k = recommendations[:k]
    hits = sum(1 for movie_id in top_k if movie_id in relevant_items)
    return hits/k

#Calculate recall@k based on the total number of relevant movies
def recall_at_k(recommendations: list[int], relevant_items: set[int], k: int = 10) -> float:
    if not relevant_items:
        return 0.0

    top_k = recommendations[:k]
    hits = sum(1 for movie_id in top_k if movie_id in relevant_items)
    return hits/len(relevant_items)

#Calculate NDCG@k giving more importance to relevant movies ranked higher
def ndcg_at_k(recommendations: list[int], relevant_items: set[int], k: int = 10) -> float:
    if not relevant_items:
        return 0.0
    
    top_k = recommendations[:k]
    dcg = 0.0

    #Calculate the discounted gain if relevant recommendations
    for position, movie_id in enumerate(top_k, start = 1):
        if movie_id in relevant_items:
            dcg += 1/math.log2(position + 1)

    #Calculate the ideal ranking and normalize the result
    ideal_hits = min(len(relevant_items), k)
    idcg = sum(1 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
    return dcg/idcg

#Calculate the proportion of catalogue movies recommended to users
def catalogue_coverage(all_recommendations: dict[int, list[int]], catalogue: set[int]) -> float:
    if not catalogue:
        return 0.0

    #Collect all distinct recommended movies
    recommended_items = set()
    for recommendations in all_recommendations.values():
        recommended_items.update(recommendations)

    recommended_items = recommended_items.intersection(catalogue)
    return len(recommended_items)/len(catalogue)

#Identify relevant movies (>= 4) for each user in validation
def get_relevant_items(validation: pd.DataFrame) -> dict[int, set[int]]:
    positive_ratings = validation[validation["Rating"] >= 4]
    relevant_items = (positive_ratings.groupby("UserID")["MovieID"].apply(set).to_dict())
    return relevant_items

#Prepare validation data for a fair comparision between Most popular and Item-based kNN
def prepare_evaluation_data(train: pd.DataFrame, validation: pd.DataFrame) -> tuple[pd.DataFrame, set[int]]:
    train_users = set(train["UserID"])
    train_movies = set(train["MovieID"])
    #Select positive validation ratings for known users and movies
    positive_validation = validation[validation["Rating"] >= 4]
    eligible_ratings = positive_validation[positive_validation["UserID"].isin(train_users) & positive_validation["MovieID"].isin(train_movies)]
    eligible_users = set(eligible_ratings["UserID"])
    #Keep validation ratings for eligible users and known movies
    validation_filtered = validation[validation["UserID"].isin(eligible_users) & validation["MovieID"].isin(train_movies)].copy()
    return validation_filtered, train_movies 

#Evaluate recommender using the four evaluation metrics
def evaluate_recommender(validation: pd.DataFrame, recommend_function, catalogue: set[int], k: int = 10) -> dict:
    if k <= 0:
        raise ValueError("k must be positive")

    #Get relevant movies for each user
    relevant_by_user = get_relevant_items(validation)
    if not relevant_by_user:
        raise ValueError("No users with relevant items to evaluate")
    
    all_recommendations = {}
    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    #Generate recommendations and calculate metrics for each user
    for user_id, relevant_items in relevant_by_user.items():
        recommendations = list(recommend_function(user_id, k))
        if len(recommendations) != len(set(recommendations)):
            raise ValueError("Duplicate movies in recommendations")
        
        all_recommendations[user_id] = recommendations
        precision_scores.append(precision_at_k(recommendations, relevant_items, k))
        recall_scores.append(recall_at_k(recommendations, relevant_items, k))
        ndcg_scores.append(ndcg_at_k(recommendations, relevant_items, k))

    #Calculate average ranking metrics and catalogue coverage
    results = {
        f"Precision@{k}": sum(precision_scores)/len(precision_scores),
        f"Recall@{k}": sum(recall_scores)/len(recall_scores),
        f"NDCG@{k}": sum(ndcg_scores)/len(ndcg_scores),
        "Catalogue Coverage": catalogue_coverage(all_recommendations, catalogue)
    }
    return results
