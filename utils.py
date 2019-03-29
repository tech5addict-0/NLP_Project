import numpy as np
import re



def BoW (claim, headline):

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

    return np.array(bag)


def Q (claim, headline):
    if "?" in headline:
        return 1
    else:
        return 0
