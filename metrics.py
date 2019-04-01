import numpy as np
import pandas as pd

def getConfusionMatrix(predictedLabels, trueLabels):
    ''' Calculate the confusion matrix and return it in a matrix
        The rows are the true values while the columns are the predicted values'''

    if len(predictedLabels) != len(trueLabels):
        print("Error in GetMetrics : The number of predicted labels do not correspond to the number of trueLabels")
        return 0
    
    else:
        # We create the dataframe that will hold the information
        possibleResult = ["for", "against", "observing"]
        result = pd.DataFrame(np.zeros([3,3]), index=possibleResult, columns=possibleResult)

        # We fill the dataframe
        for i in range(len(predictedLabels)):
            # In Dataframe, first one is the column, second one the rows
            result[predictedLabels[i]][trueLabels[i]] += 1

        return result


def getAccuracy(confMatrix):
    ''' Return the accuracy of the confusion matrix '''

    totalNbElem = sum(np.sum(confMatrix))
    possibleResult = ["for", "against", "observing"]
    ElementRight = sum([confMatrix[i][i] for i in possibleResult])
    return ElementRight/totalNbElem


def getPrecisionPerClass(confMatrix):
    ''' return the precision of each class '''

    possibleResult = ["for", "against", "observing"]
    result = pd.DataFrame(np.zeros([1,3]), index=["Precision"], columns = possibleResult)

    # We get each precision
    for stance in possibleResult:
        elementsRight = confMatrix[stance][stance]
        result[stance]["Precision"] = elementsRight / np.sum(confMatrix[stance])

    return result


def getRecallPerClass(confMatrix):
    ''' return the recall of each class '''

    possibleResult = ["for", "against", "observing"]
    result = pd.DataFrame(np.zeros([1,3]), index=["Recall"], columns = possibleResult)

    # We get each precision
    for stance in possibleResult:
        elementsRight = confMatrix[stance][stance]
        result[stance]["Recall"] = elementsRight / np.sum(confMatrix.loc[stance,:])

    return result


