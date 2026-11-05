import time

from data_analysis import load_data_with_splits
from item_knn import ItemKNN


# 1. Load MovieLens 1M with temporal splits
data = load_data_with_splits()
train = data.train

print("Ratings:", len(train))
print("Users:", train["UserID"].nunique())
print("Movies:", train["MovieID"].nunique())

# 2. Create Pearson ItemKNN model
model = ItemKNN(k=20, similarity="pearson")

# 3. Measure training time
start_time = time.perf_counter()

model.fit(train)

end_time = time.perf_counter()

print("\nPEARSON")
print("Training time:", round(end_time - start_time, 2), "seconds")
print("Similarity matrix shape:", model.sim_matrix.shape)

# 4. Test recommendations for one user
user_id = train["UserID"].iloc[0]

recommendations = model.recommend(user_id, top_n=10)

print("\nUser:", user_id)
print("Recommendations:", recommendations)

# 5. Verify that already rated movies are excluded
rated_movies = set(
    train.loc[train["UserID"] == user_id, "MovieID"]
)

already_rated = rated_movies.intersection(recommendations)

print("Already rated movies in recommendations:", already_rated)

assert len(already_rated) == 0, "Recommended an already rated movie!"

print("\nTest passed!")