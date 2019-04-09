import numpy as np
import pandas as pd
import json


# We get the dataset
with open('datasets/dataset.json','r') as json_file:  
   dataset = json.load(json_file)


# We load the data with features

# We initialize our data
accuracyMean = []
accuracyStd = []

nbRun = 10
# Define which columns represents each features
nbColFeat = len(data) - 2
svoCol = [nbColFeat - i for i in range(3)]
ppdbCol = [nbColFeat - 3]
qCol = [nbColFeat - 4]
bowCol = [i for i in range(nbColFeat-4)]

# Aggregate all this information into one array
colPerFeat = [bowCol, qCol, ppdbCol, svoCol]

# For all features to remove
for featToRemove in colPerFeat:
    colToSave = [i for i in range(nbColFeat+2) if i not in featToRemove]
    newData = data.iloc[:,colToSave]
    
    # For all the runs
    accuracy = []
    for run in range(nbRun):
        # Get training Set and testing set
        (trainingClaimSet, validationClaimSet, testingClaimSet) = getData.cutTrainTestValidationSet(dataset, 0.7, 0.1)
        # We get the traing set
        trainingSet = newData.loc[featuresDataset["claimId"].isin(trainingClaimSet)]
        # We get the traing set
        validationSet = newData.loc[featuresDataset["claimId"].isin(validationClaimSet)]
        # We get the traing set
        testingSet = newData.loc[featuresDataset["claimId"].isin(testingClaimSet)]

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
        
        
        # Get accuracy of one specific classifier
        classifier = 
        # Save this into accuracy

    # Maybe reverse accuracy stuff
    # Get mean of accuracy
    accuracyMean.append(np.mean(accuracy))
    # Get std of accuracy
    accuracyStd.append(np.std(accuracy))
    # Reset accuracy
    accuracy = []

    

    


