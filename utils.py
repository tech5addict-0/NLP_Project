import numpy as np
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
