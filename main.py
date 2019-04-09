import getData
import features
import metrics
import utils
import baselineClassifier

from logger import Logger

import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import classification_report
import sklearn.model_selection

# Get the data
with open('datasets/dataset.json','r') as json_file:  
   dataset = json.load(json_file)


# Calculate the entire set of features
logger = Logger(show = True, html_output = True, config_file = "config.txt")
feature_extraction = features.FeatureExtraction(logger)
#TODO if features are already calculated, read it from the directory instead
featuresDataset = feature_extraction.compute_features(dataset)

# Save features
featuresDataset.to_csv("datasets/features.csv")


# Cut in train, test, validation set
(trainingClaimSet, validationClaimSet, testingClaimSet) = getData.cutTrainTestValidationSet(dataset, 0.7, 0.1)

# We get the traing set
trainingSet = featuresDataset.loc[featuresDataset["claimId"].isin(trainingClaimSet)]
# We get the traing set
validationSet = featuresDataset.loc[featuresDataset["claimId"].isin(validationClaimSet)]
# We get the traing set
testingSet = featuresDataset.loc[featuresDataset["claimId"].isin(testingClaimSet)]



# Separate features and labels in the training set
trainingFeatures = trainingSet[trainingSet.columns[:-2]]
trainingFeatures = trainingFeatures.values.tolist()

trainingLabels = trainingSet[trainingSet.columns[-2:-1]]
trainingLabels = np.ravel(trainingLabels.values)


# Define validation set
validationFeatures = validationSet[validationSet.columns[:-2]]
validationFeatures = validationFeatures.values.tolist()

validationLabels = validationSet[validationSet.columns[-2:-1]]
validationLabels = np.ravel(validationLabels.values)


# Get features and labels of testingSet
testingFeatures = testingSet[trainingSet.columns[:-2]]
testingFeatures = testingFeatures.values

testingLabels = testingSet[trainingSet.columns[-2:-1]]
testingLabels = np.ravel(testingLabels.values)


# Create the big training+ValidationSet
gridFeatures = np.concatenate((trainingFeatures,validationFeatures))
gridLabels = np.concatenate((trainingLabels, validationLabels))
indexValid = list(range(len(trainingLabels),len(trainingLabels)+len(validationLabels)))


# Train the classifiers on training sets and save the data in a specific folder


accuracy = []
indexName = []
# =========== LOGISTIC CLASSIFIER =============

nameClassifier = "LogisticClassifier"
logisticClass = LogisticRegression(solver='lbfgs',multi_class="auto").fit(trainingFeatures, trainingLabels)

# get the predicted labels
predLabels = logisticClass.predict(testingFeatures)

# Get the metrics and save them
acc = utils.getAllMetricsAndSave(nameClassifier,predLabels,testingLabels)

# Accuracy
accuracy.append(acc)
indexName.append(nameClassifier)


# =========== SVM ==============
nameClassifier = "SVM"
svc = svm.SVC(gamma="auto")
#params = {"kernel":("linear","poly","rbf","sigmoid"), 'C':[i/2 for i in range(1,20)]}
params = {"kernel":("linear",), 'C':[i/2 for i in range(1,2)]}
ps = sklearn.model_selection.PredefinedSplit(indexValid)
clf = sklearn.model_selection.GridSearchCV(svc,params,cv=ps)
clf.fit(gridFeatures,gridLabels)

print("done")

# get the predicted labels
predLabels = clf.predict(testingFeatures)

# Get the metrics and save them
acc = utils.getAllMetricsAndSave(nameClassifier,predLabels,testingLabels)

# Accuracy
accuracy.append(acc)
indexName.append(nameClassifier)

# ========== Baseline Classifier ==========
nameClassifier = "BaselineClassifier"
baseClassifier = baselineClassifier.BaselineClassifier()
trainedClassifier = baseClassifier.get_overlaps(trainingFeatures)
thresholds = baseClassifier.calculate_classifier_thresholds(trainedClassifier)
# get the predicted labels
predLabels = baseClassifier.predict(thresholds,testingFeatures)

# Get the metrics and save them
acc = utils.getAllMetricsAndSave(nameClassifier,predLabels,testingLabels)

# Accuracy
accuracy.append(acc)
indexName.append(nameClassifier)


# Save data (the accuracy)

accData = pd.DataFrame(accuracy, columns=["Accuracy"], index=indexName)
accData.to_csv("results/Accuracy.csv")
