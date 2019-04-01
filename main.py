import getData
import features
import metrics

from logger import Logger

import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report

# Get the data
with open('datasets/dataset.json','r') as json_file:  
   dataset = json.load(json_file)
   #print(dataset)

#print(dataset)
# Calculate the entire set of features
logger = Logger(show = True, html_output = True, config_file = "config.txt")
feature_extraction = features.FeatureExtraction(logger)
featuresDataset = feature_extraction.compute_features(dataset)


# Cut in train, test, validation set
(trainingClaimSet, validationClaimSet, testingClaimSet) = getData.cutTrainTestValidationSet(dataset, 0.7, 0.1)

# We get the traing set
trainingSet = featuresDataset.loc[featuresDataset["claimId"].isin(trainingClaimSet)]
# We get the traing set
validationSet = featuresDataset.loc[featuresDataset["claimId"].isin(validationClaimSet)]
# We get the traing set
testingSet = featuresDataset.loc[featuresDataset["claimId"].isin(testingClaimSet)]



# print(trainingSet)
# #print(trainingClaimSet)
# print(len(trainingSet))

# wantedSize = 0
# for claimId in dataset:
#    if claimId in trainingClaimSet:
#       wantedSize += len(dataset[claimId]['articles'])

# print(wantedSize)


# Separate features and labels in the training set
trainingFeatures = trainingSet[trainingSet.columns[:-2]]
trainingFeatures = trainingFeatures.values.tolist()

trainingLabels = trainingSet[trainingSet.columns[-2:-1]]
trainingLabels = np.ravel(trainingLabels.values)


# Train the classifiers on training sets
logisticClass = LogisticRegression(solver='lbfgs',multi_class="auto").fit(trainingFeatures, trainingLabels)


# Do some Validation stuff ?

# Get error rates and evaluation metrics from all the classifiers on the test set

# Get only features of testingSet
testingFeatures = testingSet[trainingSet.columns[:-2]]
testingFeatures = testingFeatures.values

# Get the labels of the testing set
testingLabels = testingSet[trainingSet.columns[-2:-1]]
testingLabels = np.ravel(testingLabels.values)

# Get the confusion matrix
predLabels = logisticClass.predict(testingFeatures)
confMat = metrics.getConfusionMatrix(predLabels,testingLabels)
print(confMat)

# Get All the metrics :
acc = metrics.getAccuracy(confMat)
print(acc)

prec = metrics.getPrecisionPerClass(confMat)
print(prec)

recall = metrics.getRecallPerClass(confMat)
print(recall)

#print(classification_report(testingLabels, predLabels))


# Save data
