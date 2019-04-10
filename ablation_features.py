import utils
import metrics
import getData

import numpy as np
import pandas as pd
import json
from sklearn import svm
from sklearn.linear_model import LogisticRegression

# We get the dataset
with open('datasets/dataset.json','r') as json_file:  
   dataset = json.load(json_file)

data = pd.read_csv("datasets/features.csv")
data.rename(columns = {str(len(data.columns)-2):"claimId",str(len(data.columns)-4):"SVO3",str(len(data.columns)-5):"SVO2",str(len(data.columns)-6):"SVO1",str(len(data.columns)-7):"ppdb",str(len(data.columns)-8):"q"} ,inplace=True)
print(data.columns)


# We load the data with features

# We initialize our data
accuracyMean = []
accuracyStd = []

nbRun = 10
# Define which columns represents each features
nbColFeat = len(data.columns) - 3
print(data.columns[nbColFeat])
word2VecCol = [nbColFeat]
svoCol = [nbColFeat - i for i in range(1,4)]
rootDistCol = [nbColFeat - 4]
ppdbCol = [nbColFeat - 5]
qCol = [nbColFeat - 6]
negCol = [nbColFeat - 7]
bowCol = [i for i in range(nbColFeat-7)]
allCol = []

# Aggregate all this information into one array
colPerFeat = [bowCol, negCol, qCol, ppdbCol, rootDistCol, svoCol, word2VecCol, allCol]

# For all features to remove
for featToRemove in colPerFeat:
    colToSave = [i for i in range(len(data.columns)) if i not in featToRemove]
    newData = data.iloc[:,colToSave]
    #print(newData[newData.columns[-8:-2]])
    print(newData.columns)
    
    # For all the runs
    accuracy = []
    for run in range(nbRun):
        # Get training Set and testing set
        (trainingClaimSet, validationClaimSet, testingClaimSet) = getData.cutTrainTestValidationSet(dataset, 0.7, 0.1)
        # We get the traing set
        trainingSet = newData.loc[newData["claimId"].isin(trainingClaimSet)]
        # We get the traing set
        validationSet = newData.loc[newData["claimId"].isin(validationClaimSet)]
        # We get the traing set
        testingSet = newData.loc[newData["claimId"].isin(testingClaimSet)]

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
        
        
        # Get accuracy of one specific classifier (get the optimal params)
        # YOU NEED TO KNOW THE OPTIMAL PARAMETERS DUMBASS
        logisticClass = LogisticRegression(solver='lbfgs',multi_class="auto",max_iter=250).fit(gridFeatures, gridLabels)
        predLabels = logisticClass.predict(testingFeatures)

        # Get confMatrix and accuracy:
        confMat = metrics.getConfusionMatrix(predLabels,testingLabels)
        acc = metrics.getAccuracy(confMat)
        # Save this into accuracy
        accuracy.append(acc)
        
        
    # Maybe reverse accuracy stuff
    # Get mean of accuracy
    accuracyMean.append(np.mean(accuracy))
    # Get std of accuracy
    accuracyStd.append(np.std(accuracy))
    # Reset accuracy
    accuracy = []

# Save the ablation analysis
featuresNames = ["BoW","neg","q","ppdb","rootDist","svo","word2Vec","all"]
df = pd.DataFrame.from_dict({"Accuracy Mean":accuracyMean, "Accuracy Std":accuracyStd, "Features":featuresNames})
df.set_index("Features",inplace=True)
df.to_csv("results/ablation.csv")

    

    


