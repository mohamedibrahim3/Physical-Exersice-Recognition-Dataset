# Pose Classification

This is a python script that uses machine learning techniques to classify the pose of a person in an image. The script uses the [scikit-learn](https://scikit-learn.org/stable/) library to train and test two different classifiers: [K-Nearest Neighbors (KNN)](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) and [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier).

## Dependencies

* [pandas](https://pandas.pydata.org/)
* [scikit-learn](https://scikit-learn.org/stable/)

## How to use

1. Install the dependencies by running `pip install -r requirements.txt`
2. Modify the `dataset` variable to point to the location of the training data CSV file.
3. Run the script with `python classify.py`

## What the script does

1. The script loads the training data from a CSV file using pandas.
2. It then splits the data into feature variables (stored in `x`) and target variables (stored in `y`).
3. The data is then split into training and testing sets using scikit-learn's `train_test_split` function.
4. The script trains a KNN classifier and a Naive Bayes classifier on the training data.
5. The classifiers are then used to make predictions on the testing data.
6. The script then calculates the accuracy of the predictions using scikit-learn's `confusion_matrix` and `accuracy_score` functions.
7. The results are printed to the console.
