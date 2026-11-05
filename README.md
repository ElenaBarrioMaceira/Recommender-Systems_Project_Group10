# Recommender-Systems_Project

## General instructions
- Add a .env file to the root of the project(not inside src folder). In that file put DATA_PATH="your/path/to/the/movielensdata"
- I added two new packages to the requirements.txt, so remember to download them!

## Instructions for the file data_loader.py
The function **load_data()** loads the MovieLens data into 3 dataframes, movies, users and ratings

## Instructions for the file data_analysis.py
Use the function **load_data_with_splits** to get the data for the modelling. The function returns an object of a class **DataForModeling**. The object has the following fields in addition to the 3 dataframes loaded by the load_data():
  - train: The DataFrame for the training data
  - validation: The DataFrame for the validation data
  - test: The DataFrame for the test data

You can use this file by e.g. importing it with the line: **import data_analysis as data_analysis**

Then, get the data for modeling with the line: **data_analysis.load_data_with_splits()**

### Instructions for the function load_data_with_splits
All the parameters below are already given default values, but you can change them if you want:
- Default train/validation/test split is set to 0.8/0.1/0.1. The split is made temporally by just taking the first 80% of data for testing, the next 10% for validation and the last 10% for testing.
  
- If you want that the validation and testing data does not have movies in it, that are absent from the training data, e.g. a cold start, set the parameter **remove_unknown_items** to **True**
- If you want to remove users with low activity from the training, validation and testing data, set the parameter **remove_low_activity_users** to **True**, and give a some value to the parameter **low_activity_user_threshold**. If you set **remove_low_activity_users** to **True**, and **low_activity_user_threshold** to 3, then users that have given less than 3 reviews are removed from the data.
- If you want to remove movies with low activity from the training, validation and testing data, set the parameter **remove_low_activity_items** to **True**, and give a some value to the parameter **low_activity_item_threshold**. If you set **remove_low_activity_items** to **True**, and **low_activity_item_threshold** to 3, then movies that have been given less than 3 reviews are removed from the data.

### Instructions for the function data_summary
Call this function if you want to print the characteristics of data and the plots for the interactions per user, interactions per item and the time span and how activity is distributed across it.

## Most popular baselines
Two most popular recommendation baselines are implemented in 'src/baselines.py'
  - The 'most_popular_all_ratings()' function ranks movies acccording to the total number od ratings they receive in the training set, regarless of their rating value.
  - The 'most_popular_positive_ratings' function ranks movies according to the number of positive ratings (>= 4) they receive in the training set. Only movies with at least one positive training rating are included in this ranking.

### Generating recommendations
The 'recommend_top_n()' function generates Top_n recommendations for a given user.
  - Movies already rated by the user in the training set are excluded.
  - Movies are ranked by their popularity score in descending order.
  - Ties are resolved using the smalles "MovieID" first.
  - The default number of recommendations is 10.

## Evaluation Metrics
The evaluation functions are implemented is 'src/evaluation.py'
The following metrics are supported:
  - **Precision@10.** Proportion of the Top-10 recommendations that are relevant.
  - **Recall@10.** Proportion od the user's relevant movies retrieved in the Top-10.
  - **NDCG@10.** Measures ranking quality by assigning greater importance to relevant movies appearing earlier.
  - **Catalogue Coverage:** Proportion of distinct catalogue movies appearing in the recommendations across evaluated users.

A movie is considered relevant if its rating is 4 or 5. The 'evaluate_recommender()' function provides a common evaluation interface for Most popular and Item-based kNN. It accepts a recommendation function that takes a used ID and a recommendation count and return an ordered list of movie IDs.

## Evaluation protocol
The evaluation uses a warm-start protocol to ensure a consistent comparision between Most popular and Item-based kNN
  - Only users with training history and at least one relevant validation movie known during training are evaluated.
  - Movies with a rating of 4 ot 5 are considered relevant
  - Previous rated movies are excluded from recommendations
  - Validation is used dor model comparision and hyperparameter tuning, while the test ser is reserved for final evaluation

The 'prepare_evaluation_data()' function applies these filters. This protocol excludes cold-start users and unknown movies, which should be considered when interpreting the results.