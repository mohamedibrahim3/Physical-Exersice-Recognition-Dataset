import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB

dataset = pd.read_csv('train.csv', delimiter=',')
featureNames = dataset.columns[2:-1]

# print(featureNames)

target = dataset['pose'].tolist()
target = list(set(target))

# print(target)

x = dataset[featureNames].values

# print(x[0])

y = dataset['pose']

# print(y)

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=5)

# print(xTrain.shape, xTest.shape)
# print(yTrain.shape, yTest.shape)

# ******************
# KNN Classification
# ******************

neighbor = KNeighborsClassifier(n_neighbors=6)
neighbor.fit(xTrain, yTrain)

predictedKNN = neighbor.predict(xTest)

print('Predicted by KNN: ', predictedKNN)

resultsKNN = confusion_matrix(yTest, predictedKNN)

# print('KNN confusion Matrix: \n', resultsKNN)
print('\nKNN Accuracy: ', round(accuracy_score(yTest, predictedKNN) * 100), '%', sep='', end='\n\n')

print('-' * 100, end='\n\n')

# **************************
# Naive Bayes Classification
# **************************

model = GaussianNB()
model.fit(xTrain, yTrain)

predictedNB = model.predict(xTest)

# print('Predicted by Naive Bayes: ', predictedNB)

resultsNB = confusion_matrix(yTest, predictedNB)

print('Naive Bayes Confusion Matrix: \n', resultsNB)
print('\nNaive Bayes Accuracy: ', round(accuracy_score(yTest, predictedNB) * 100), '%', sep='')