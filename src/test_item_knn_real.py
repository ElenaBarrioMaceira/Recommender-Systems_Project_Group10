import time

from data_analysis import load_data_with_splits
from item_knn import ItemKNN


# 1. Load MovieLens 1M with temporal splits
data = load_data_with_splits()

train = data.train

print("Training ratings:", len(train))
print("Training users:", train["UserID"].nunique())
print("Training movies:", train["MovieID"].nunique())


# 2. Create Item-based kNN with cosine similarity
model = ItemKNN(k=20, similarity="cosine")


# 3. Train the model and measure execution time
start_time = time.perf_counter()

model.fit(train)

end_time = time.perf_counter()

print("\nCOSINE")
print("Training time:", round(end_time - start_time, 2), "seconds")


# 4. Select a user from the training set
user_id = train["UserID"].iloc[0]

print("User:", user_id)


# 5. Generate Top-10 recommendations
recommendations = model.recommend(user_id, top_n=10)

print("Recommendations:", recommendations)


# 6. Check that recommended movies were not already rated
rated_movies = set(
    train.loc[train["UserID"] == user_id, "MovieID"]
)

already_rated = set(recommendations) & rated_movies

print("Already rated movies in recommendations:", already_rated)

assert len(already_rated) == 0

print("\nTest passed!")