import numpy as np
import metrics
import re
from logger import Logger


def __init__(self, logger):
    self.logger = logger

def BoW (self, claim, headline):

    wordsClaim = re.sub("[^\w]", " ", claim).split()
    wordsClaim_cleaned = [w.lower() for w in wordsClaim]
    wordsClaim_cleaned = sorted(list(set(wordsClaim_cleaned)))

    wordsHeadline = re.sub("[^\w]", " ", headline).split()
    wordsHeadline_cleaned = [w.lower() for w in wordsHeadline]
    wordsHeadline_cleaned = sorted(list(set(wordsHeadline_cleaned)))

    bag = np.zeros(len(wordsClaim_cleaned))
    for hw in wordsHeadline_cleaned:
        for i, cw in enumerate(wordsClaim_cleaned):
            if hw == cw:
                bag[i] += 1

    self.logger.log("Feature Bag of Words completed.")
    return np.array(bag)


def Q (self, claim, headline):
    if "?" in headline:
        self.logger.log("Feature Question completed.")
        return 1
    else:
        self.logger.log("Feature Question completed.")
        return 0


def getAllMetricsAndSave(nameClassifier,predLabels,testingLabels):
    ''' Get all the metrics from the predicted labels and save the results in the directory results/ and return the accuracy'''

    # get the confusion Matrix
    confMat = metrics.getConfusionMatrix(predLabels,testingLabels)
    confMat.to_csv("results/"+nameClassifier+"-confMat.csv")
    
    # Precision
    prec = metrics.getPrecisionPerClass(confMat)
    prec.to_csv("results/"+nameClassifier+"-Precision.csv")
    
    # Recall
    recall = metrics.getRecallPerClass(confMat)
    recall.to_csv("results/"+nameClassifier+"-Recall.csv")

    acc = metrics.getAccuracy(confMat)
    return acc
