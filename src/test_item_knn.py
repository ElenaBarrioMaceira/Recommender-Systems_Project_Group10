import pandas as pd
import numpy as np

from item_knn import ItemKNN


# 1. Create a small dataset
train = pd.DataFrame({
    "UserID":  [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
    "MovieID": [10, 20, 30, 10, 20, 40, 10, 30, 40, 20, 30, 40],
    "Rating":  [5, 4, 3, 4, 5, 3, 3, 4, 5, 3, 2, 4]
})


# 2. Test cosine similarity
model_cosine = ItemKNN(k=2, similarity="cosine")

model_cosine.fit(train)

print("COSINE")
print("Similarity matrix:")
print(model_cosine.sim_matrix)

print("Scores for user 1:")
print(model_cosine.score_items(1))

print("Recommendations for user 1:")
print(model_cosine.recommend(1, top_n=2))


# 3. Test Pearson similarity
model_pearson = ItemKNN(k=2, similarity="pearson")

model_pearson.fit(train)

print("\nPEARSON")
print("Similarity matrix:")
print(model_pearson.sim_matrix)

print("Scores for user 1:")
print(model_pearson.score_items(1))

print("Recommendations for user 1:")
print(model_pearson.recommend(1, top_n=2))