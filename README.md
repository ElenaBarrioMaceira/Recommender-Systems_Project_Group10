# Recommender-Systems_Project

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
- Default train/validation/test split is set to 0.8/0.1/0.1
  
- If you want that the validation and testing data does not have movies in it, that are absent from the training data, e.g. a cold start, set the parameter **remove_unknown_items** to **True**
- If you want to remove users with low activity from the training, validation and testing data, set the parameter **remove_low_activity_users** to **True**, and give a some value to the parameter **low_activity_user_threshold**. If you set **remove_low_activity_users** to **True**, and **low_activity_user_threshold** to 3, then users that have given less than 3 reviews are removed from the data.
- If you want to remove movies with low activity from the training, validation and testing data, set the parameter **remove_low_activity_items** to **True**, and give a some value to the parameter **low_activity_item_threshold**. If you set **remove_low_activity_items** to **True**, and **low_activity_item_threshold** to 3, then movies that have been given less than 3 reviews are removed from the data.
