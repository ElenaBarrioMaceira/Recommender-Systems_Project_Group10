# Item-based kNN Recommender System

This document describes the implementation, testing, and hyperparameter tuning of the Item-based k-Nearest Neighbors (ItemKNN) recommender system developed for the Recommender Systems project.

The model uses the MovieLens 1M dataset and supports two item-item similarity measures: cosine similarity and Pearson correlation.

## 1. Implementation

**File:** `src/item_knn.py`

The `ItemKNN` class implements an item-based collaborative filtering recommender system.

### Main functionality

- Builds a sparse user-item rating matrix from the training data.
- Computes item-item similarities using cosine similarity or Pearson correlation.
- Uses an optimized Pearson implementation based on `pandas.DataFrame.corr()`.
- Selects up to `k` positively similar rated items as neighbors for each candidate item.
- Predicts item scores using a similarity-weighted average of the user's ratings.
- Generates Top-N recommendations.
- Excludes items already rated by the user.

### Main methods

| Method | Description |
|---|---|
| `__init__(k=20, similarity="cosine")` | Initializes the model and its hyperparameters. |
| `fit(train)` | Builds the user-item matrix and computes item-item similarities. |
| `score_items(user_id)` | Calculates prediction scores for candidate items. |
| `recommend(user_id, top_n=10)` | Returns the highest-ranked recommended MovieIDs. |

The model supports the following hyperparameters:

- `k`: Maximum number of neighbors used for each prediction.
- `similarity`: Similarity measure (`"cosine"` or `"pearson"`).

## 2. Similarity Measures

### Cosine Similarity

Cosine similarity is computed between item rating vectors using `sklearn.metrics.pairwise.cosine_similarity()`.

The implementation uses the sparse user-item matrix to calculate item-item similarities efficiently.

### Pearson Correlation

Pearson correlation is computed using ratings from users who have rated both items.

The implementation:

1. Converts the sparse rating matrix to a dense matrix.
2. Represents missing ratings as `NaN`.
3. Computes pairwise Pearson correlations using `pandas.DataFrame.corr(method="pearson", min_periods=2)`.
4. Replaces undefined correlations with zero.

The original loop-based Pearson implementation is preserved as commented code in `item_knn.py` for comparison.

Self-similarities are set to zero for both methods.

## 3. Testing

### `src/test_item_knn.py`

Tests the model using a small synthetic dataset.

It checks:

- Cosine similarity calculations.
- Pearson correlation calculations.
- Predicted item scores.
- Generated recommendations.

### `src/test_item_knn_real.py`

Tests ItemKNN with the MovieLens 1M training data.

It verifies that:

- The model trains successfully.
- Recommendations can be generated.
- Previously rated movies are excluded from recommendations.

### `src/test_pearson_speed.py`

Measures the training time of the Pearson implementation.

The optimized implementation was tested using the full MovieLens 1M training split.

**Full-dataset results:**

| Property | Value |
|---|---:|
| Training ratings | 800,167 |
| Training users | 5,400 |
| Training movies | 3,662 |
| Pearson training time | 215.12 seconds |
| Similarity matrix shape | (3662, 3662) |

Training times depend on the execution environment.

## 4. Hyperparameter Tuning

**File:** `src/tune_item_knn.py`

This script prepares the comparison of different ItemKNN configurations.

### Configurations

| Similarity | k values |
|---|---|
| Cosine | 10, 20, 50 |
| Pearson | 10, 20, 50 |

The script:

1. Loads the temporal training and validation splits.
2. Selects validation users who also appear in training.
3. Trains one model for each similarity measure.
4. Reuses the computed similarity matrix for different values of `k`.
5. Generates Top-10 recommendations for a small sample of users.

Reusing the similarity matrix avoids recalculating Pearson correlations for every value of `k`.

**Current status:** The script generates recommendations for all six configurations, but it does not yet calculate evaluation metrics or select the optimal hyperparameters.

## 5. Evaluation — Pending Integration

The following evaluation metrics will be used to compare configurations:

- Precision@10
- Recall@10
- NDCG@10
- Catalogue coverage

These metrics are being implemented separately and will be integrated into the tuning process.

The best combination of similarity measure and `k` will be selected using validation results.

The test split will be reserved for final evaluation.

### Cold-start considerations

The temporal split can produce validation users who are absent from training.

Currently, `ItemKNN` requires the user to exist in the training data. The tuning script therefore selects users present in both splits.

The final evaluation protocol must define how to handle unknown users and items consistently across the recommender models.

## 6. Running the Tests

From the project root, with the Python environment activated and the `DATA_PATH` environment variable configured:

**Synthetic dataset test:**

```bash
python src/test_item_knn.py
```

**MovieLens 1M integration test:**

```bash
python src/test_item_knn_real.py
```

**Pearson performance test:**

```bash
python src/test_pearson_speed.py
```

**Hyperparameter tuning preparation:**

```bash
python src/tune_item_knn.py
```

The Pearson tests may take several minutes when using the complete training dataset.

## 7. Implementation Status

| Component | Status |
|---|---|
| User-item matrix construction | Completed |
| Cosine similarity | Completed |
| Pearson correlation | Completed |
| Pearson optimization | Completed |
| Weighted item scoring | Completed |
| Top-N recommendations | Completed |
| Exclusion of previously rated items | Completed |
| Synthetic dataset testing | Completed |
| MovieLens 1M integration testing | Completed |
| Hyperparameter configurations | Prepared |
| Evaluation metrics integration | Pending |
| Optimal hyperparameter selection | Pending |