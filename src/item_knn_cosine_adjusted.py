import numpy as np

from item_knn import ItemKNN
from item_knn_adjusted import ItemKNNAdjusted


class ItemKNNCosineAdjusted(ItemKNNAdjusted):
    """
    Experimental ItemKNN using cosine similarity and
    item-mean-adjusted rating predictions.

    Uses the same adjusted prediction formula as
    ItemKNNAdjusted (Pearson).

    The original ItemKNN implementation remains unchanged.
    """

    def __init__(self, k=20):

        if k < 1:
            raise ValueError("k must be positive")

        self.k = k
        self.similarity = "cosine"

    def fit(self, train):

        # 1. Fit the original cosine ItemKNN
        ItemKNN.fit(self, train)

        # 2. Calculate each movie's average rating
        rating_sums = np.asarray(
            self.user_item.sum(axis=0)
        ).ravel()

        rating_counts = np.diff(
            self.user_item.tocsc().indptr
        )

        self.item_means = (
            rating_sums / rating_counts
        )

        return self