import sys
import numpy as np 
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_analysis import load_data_with_splits
from evaluation import prepare_evaluation_data, get_relevant_items
from user_knn import UserKNN

#Selected UserKNN configuration
SIMILARITY = "pearson"
NEIGHBOURS = 10
SIGNIFICANCE_THRESHOLD = 50
TOP_N = 10
#Number of users for the initial check
N_USERS = 5

def main():
    #Load training and validation data
    data = load_data_with_splits()
    train = data.train
    validation = data.validation

    #Prepare the same evaluation users and catalogue
    validation_filtered, catalogue = prepare_evaluation_data(train, validation)
    relevant_by_user = get_relevant_items(validation_filtered)
    user_ids = sorted(relevant_by_user.keys())[:N_USERS]

    print("Users to check:", len(user_ids))
    print("Catalogue size:", len(catalogue))

    #Train the selected UserKNN model
    model = UserKNN(k = NEIGHBOURS, similarity = SIMILARITY, significance_threshold = SIGNIFICANCE_THRESHOLD)
    model.fit(train)

    recommendation_lengths = []
    users_with_fewer_than_10 = 0
    total_hits = 0

    #Check recommendations for each user
    for user_id in user_ids:
        recommendations = model.recommend(user_id, top_n = TOP_N)
        recommendation_lengths.append(len(recommendations))
        #Check recommendation list length
        assert len(recommendations) <= TOP_N
        if len(recommendations) < TOP_N:
            users_with_fewer_than_10 +=1
        #Check that recommended movies belong to the catalogue
        assert set(recommendations).issubset(catalogue)
        #Check that recommendations do not contain duplicates
        assert len(recommendations) == len(set(recommendations))
        #Check that previously reted movies are excluded
        user_idx = model.user_to_idx[user_id]
        rated_indices = model.user_item.getrow(user_idx).indices
        rated_movies = set(model.movie_ids[rated_indices])
        assert not set(recommendations).intersection(rated_movies)
        #Count relevant recommendations
        relevant_items = relevant_by_user[user_id]
        hits = sum(movie_id in relevant_items for movie_id in recommendations)
        total_hits += hits

        print(f"\nUser {user_id}")
        print("Recommendations:", recommendations)
        print("Number of recommendations:", len(recommendations))
        print("Relevant recommendations:", hits)
        
    # Display summary
    print("\nUSERKNN RECOMMENDATION CHECKS")

    print("Average recommendations per user:",
          round(float(np.mean(recommendation_lengths)), 2))

    print("Minimum recommendations:", min(recommendation_lengths))
    print("Maximum recommendations:", max(recommendation_lengths))
    print("Users with fewer than 10:", users_with_fewer_than_10)
    print("Total relevant recommendations:", total_hits)

    print("\nAll recommendation checks passed!")


if __name__ == "__main__":
    main()
