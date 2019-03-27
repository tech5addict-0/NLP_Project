import os
import itertools as it
import functools as ft
import functools
import operator as op

from sklearn.model_selection._split import _BaseKFold

try:
    from functools import lru_cache
except:
    from repoze.lru import lru_cache

try:
    import cPickle as pickle
except:
    import pickle

import numpy as np
import csv
import gensim
import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.porter import PorterStemmer

VALID_STANCE_LABELS = ['for', 'against', 'observing']

_data_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

def get_dataset(filename='url-versions-2015-06-14-clean.csv'):
    folder = os.path.join(_data_folder, 'emergent')
    return pd.DataFrame.from_csv(os.path.join(folder, filename))

def get_stanparse_data():
    with open(os.path.join(_pickled_data_folder, 'stanparse-data.pickle'), 'rb') as f:
        return pickle.load(f)

def get_stanford_idx(x):
    i = x.rfind('-')
    return x[:i].lower(), int(x[(i+1):])

class ClaimKFold(_BaseKFold):

    def __init__(self, data, n_folds=10, shuffle=False):
        super(ClaimKFold, self).__init__(len(data), n_folds, None, False, None)
        self.shuffle = shuffle
        self.data = data.copy()
        self.data['iloc_index'] = range(len(self.data))

    def _iter_test_indices(self):
        claim_ids = np.unique(self.data.claimId)
        cv = KFold(len(claim_ids), self.n_folds, shuffle=self.shuffle)

        for _, test in cv:
            test_claim_ids = claim_ids[test]
            test_data = self.data[self.data.claimId.isin(test_claim_ids)]
            yield test_data.iloc_index.values

    def __len__(self):
        return self.n_folds

df_clean_train = get_dataset('url-versions-2015-06-14-clean-train.csv')



#Neg Zhang
#PPDB Priya
#SVO Priya
#word2vec Priya