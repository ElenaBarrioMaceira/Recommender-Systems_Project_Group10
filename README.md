# Recommender Systems Project — Group 10

## 1. Project Overview

This repository contains the implementation and evaluation of recommendation algorithms using the MovieLens 1M dataset.

The project compares two Most Popular baselines with several configurations of Item-based k-Nearest Neighbors (ItemKNN).

The evaluation follows a temporal train/validation/test split and uses four ranking metrics:

- Precision@10
- Recall@10
- NDCG@10
- Catalogue Coverage

Hyperparameters are selected using validation data, while the test set is reserved for final evaluation.

## 2. Requirements

The project was developed and tested using:

- Python 3.12.7
- NumPy 2.2.6
- Pandas 2.3.3
- Matplotlib 3.10.7
- Scikit-learn 1.7.2
- SciPy 1.16.3
- python-dotenv 1.2.4
- Pydantic 2.14.2

All Python dependencies are listed in `requirements.txt`.

## 3. Installation

### 3.1 Clone the repository

Clone the GitHub repository and navigate to the project directory.

### 3.2 Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3.3 Install dependencies

```bash
python -m pip install -r requirements.txt
```

## 4. Dataset Configuration

The project uses the MovieLens 1M dataset provided by GroupLens.

Download and extract MovieLens 1M before running the experiments.

The extracted dataset directory must contain:

```text
ratings.dat
movies.dat
users.dat
```

### 4.1 Configure the dataset path

Create a `.env` file in the project root directory, not inside `src`.

Add the following variable:

```env
DATA_PATH="path/to/ml-1m"
```

Replace the example path with the actual location of the extracted dataset.

For example, on Windows:

```env
DATA_PATH="C:/Users/YourName/Datasets/ml-1m"
```

The dataset path can also be configured using the `DATA_PATH` environment variable.

The main experiment script additionally supports an explicit dataset path:

```bash
python run_all.py --data-path "path/to/ml-1m"
```

## 5. Data Loading

The data-loading functionality is implemented in `src/data_loader.py`.

The `load_data()` function loads the MovieLens dataset into three Pandas DataFrames:

- `movies`: Movie information.
- `users`: User information.
- `ratings`: User–movie interactions and ratings.

## 6. Data Analysis and Preprocessing

The data analysis and splitting functionality is implemented in `src/data_analysis.py`.

### 6.1 Loading the modeling data

The `load_data_with_splits()` function returns an object of the `DataForModeling` class.

This object contains the original data and the three modeling subsets:

- `train`: Training interactions.
- `validation`: Validation interactions.
- `test`: Test interactions.

Example:

```python
import data_analysis as data_analysis

data = data_analysis.load_data_with_splits()

train = data.train
validation = data.validation
test = data.test
```

### 6.2 Temporal data splitting

The default split proportions are:

| Dataset | Proportion |
|---|---:|
| Training | 80% |
| Validation | 10% |
| Test | 10% |

Interactions are ordered chronologically.

The earliest 80% of interactions form the training set, the following 10% form the validation set, and the latest 10% form the test set.

This approach evaluates recommendations using interactions that occur after the training period.

### 6.3 Optional preprocessing

The `load_data_with_splits()` function supports configurable preprocessing options.

**Removing unknown items**

Set `remove_unknown_items=True` to exclude validation and test interactions involving movies that are absent from the training data.

**Removing low-activity users**

Set `remove_low_activity_users=True` and configure `low_activity_user_threshold`.

For example, a threshold of 3 removes users with fewer than three interactions according to the preprocessing implementation.

**Removing low-activity items**

Set `remove_low_activity_items=True` and configure `low_activity_item_threshold`.

For example, a threshold of 3 removes movies with fewer than three interactions according to the preprocessing implementation.

### 6.4 Dataset characterization

The data analysis functionality provides descriptive statistics and visualizations of:

- User activity.
- Item activity.
- Rating distribution.
- Temporal distribution of interactions.

The main experiment script saves the generated figures in the `figures/` directory.

## 7. Recommendation Models

The project evaluates two Most Popular baselines and four ItemKNN configurations.

### 7.1 Most Popular — All Ratings

Implemented in `src/baselines.py`.

The `most_popular_all_ratings()` function ranks movies according to the total number of ratings received in the training set, regardless of rating value.

### 7.2 Most Popular — Positive Ratings

Implemented in `src/baselines.py`.

The `most_popular_positive_ratings()` function ranks movies according to the number of positive training ratings.

A positive rating is defined as a rating greater than or equal to 4.

Only movies with at least one positive training rating are included in this popularity ranking.

### 7.3 Most Popular Recommendations

The `recommend_top_n()` function generates Top-N recommendations for a given user.

- Movies already rated by the user during training are excluded.
- Movies are ranked by popularity score in descending order.
- Ties are resolved by selecting the smaller `MovieID` first.
- The default recommendation list length is 10.

### 7.4 Item-based k-Nearest Neighbors

ItemKNN is implemented in `src/item_knn.py`.

The model computes similarities between movies using the user–item rating matrix.

Two similarity measures are considered:

**Cosine similarity**

Measures similarity between item rating vectors using the cosine of the angle between them.

**Pearson correlation**

Measures the linear relationship between ratings of two movies, using users who rated both movies.

The original Pearson implementation requires at least five common observations when calculating correlations.

For each candidate movie, the original ItemKNN model:

1. Identifies movies previously rated by the target user.
2. Retains neighbors with positive similarity.
3. Selects up to `k` neighbors with the highest similarity.
4. Computes a similarity-weighted average of their ratings.
5. Ranks candidate movies by predicted score.

Previously rated movies are excluded from the recommendation list.

### 7.5 Adjusted ItemKNN

Two additional configurations use an adjusted rating prediction formula:

- Pearson Adjusted: `src/item_knn_adjusted.py`
- Cosine Adjusted: `src/item_knn_cosine_adjusted.py`

These models account for item mean ratings when computing predictions.

Instead of directly averaging neighboring ratings, they aggregate deviations from the neighboring items' mean ratings and add the target item's mean rating.

The adjusted predictions are restricted to the MovieLens rating range of 1 to 5.

### 7.6 Hyperparameter Selection

ItemKNN configurations are tuned using the validation set.

The tested neighborhood sizes are:

```text
k = 10, 20, 50
```

The adjusted Pearson model additionally evaluates different minimum numbers of common ratings.

The selection criterion is **NDCG@10 on validation data**.

The selected configurations are:

| Model | Selected parameters |
|---|---|
| ItemKNN Cosine Original | k = 10 |
| ItemKNN Pearson Original | k = 20; min_periods = 5 |
| ItemKNN Cosine Adjusted | k = 10 |
| ItemKNN Pearson Adjusted | k = 20; min_periods = 4 |

The test set is not used for hyperparameter selection.

## 8. Evaluation Metrics

The evaluation functions are implemented in `src/evaluation.py`.

Four metrics are calculated.

### Precision@10

The proportion of recommended movies in the Top-10 list that are relevant.

### Recall@10

The proportion of a user's relevant movies that appear in the Top-10 recommendations.

### NDCG@10

Normalized Discounted Cumulative Gain measures ranking quality, assigning greater importance to relevant movies appearing earlier in the recommendation list.

### Catalogue Coverage

The proportion of distinct training-catalogue movies that appear in at least one recommendation list across the evaluated users.

A movie is considered relevant when its rating is 4 or 5.

The `evaluate_recommender()` function provides a common evaluation interface for Most Popular and ItemKNN models.

It accepts a recommendation function that receives a user ID and a recommendation count and returns an ordered list of movie IDs.

## 9. Evaluation Protocol

The project follows a warm-start evaluation protocol.

- Only users with training history are evaluated.
- Evaluated users must have at least one relevant validation or test movie that is present in the training catalogue.
- Movies rated 4 or 5 are considered relevant.
- Movies already rated during training are excluded from recommendations.
- Validation data is used for model comparison and hyperparameter tuning.
- Test data is reserved for final evaluation.

The `prepare_evaluation_data()` function applies the evaluation filters.

This protocol excludes cold-start users and unknown movies. Consequently, the reported results describe warm-start recommendation performance rather than performance for entirely new users or items.

## 10. Reproducing the Experiments

The main entry point is:

```bash
python run_all.py
```

Run this command from the project root directory after installing the dependencies and configuring the dataset path.

The script executes the complete experimental workflow:

1. Load and characterize the MovieLens 1M dataset.
2. Generate dataset visualizations.
3. Tune original ItemKNN configurations.
4. Tune Pearson Adjusted ItemKNN.
5. Tune Cosine Adjusted ItemKNN.
6. Check ItemKNN recommendation validity.
7. Evaluate the selected ItemKNN configurations on the test set.
8. Compare ItemKNN models with the Most Popular baselines.

The script stops if one of the execution steps fails.

### Running individual experiments

The scripts can also be executed separately:

**Original ItemKNN tuning**

```bash
python src/tune_item_knn.py
```

**Pearson Adjusted tuning**

```bash
python src/tune_item_knn_adjusted.py
```

**Cosine Adjusted tuning**

```bash
python src/tune_cosine_adjusted.py
```

**Recommendation validity checks**

```bash
python src/check_item_knn.py
```

**Final ItemKNN evaluation**

```bash
python src/test_item_knn_final.py
```

**Comparison of all models**

```bash
python src/compare_models.py
```

## 11. Generated Outputs

Running the complete workflow generates the following CSV files:

| File | Description |
|---|---|
| `item_knn_validation_results.csv` | Validation results for original ItemKNN |
| `item_knn_adjusted_validation_results.csv` | Validation results for Pearson Adjusted ItemKNN |
| `cosine_adjusted_validation_results.csv` | Validation results for Cosine Adjusted ItemKNN |
| `item_knn_test_results.csv` | Final test results for ItemKNN |
| `model_comparison_test.csv` | Final comparison of all six models |

The following figures are saved in the `figures/` directory:

```text
figures/
├── interactions_distribution.png
├── rating_distribution.png
└── time_distribution.png
```

## 12. Reproducibility

The project uses:

- Python 3.12.7.
- Fixed dependency versions in `requirements.txt`.
- A chronological train/validation/test split.
- Validation-based hyperparameter selection.
- A separate final test evaluation.
- A common evaluation protocol for all recommendation models.
- A single command to reproduce the experimental workflow.

The dataset path is configurable, allowing the project to be executed on different computers.

## 13. Contributors

This project was developed collaboratively by Group 10.

Contributions include dataset characterization, preprocessing, temporal splitting, recommendation model implementation, hyperparameter tuning, evaluation, and result analysis.

The detailed contribution statement is provided in the project report.

## 14. AI Assistance Disclosure

AI assistance was used during parts of the development and documentation process, including code review, debugging, and explanations of recommendation algorithms.

The project team is responsible for reviewing the implementations, executing the experiments, validating the results, and interpreting the findings.