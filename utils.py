import numpy as np
import metrics
import re
import pandas as pd
import os
import nltk
import gensim

from nltk import stem
from logger import Logger
try:
    import cPickle as pickle
except:
    import pickle


_pickled_data_folder = "pickled/stanparse-data.pickle"
_pickled_data_folder2 = "pickled/aligned-data.pickle"
_pickled_data_folder3 = "pickled/stanparse-depths.pickle"
MIN_ALIGNMENT_SCORE = -10
MAX_ALIGNMENT_SCORE = 10

def getLinesFromFile(filePath, fileOption ):
    with open(filePath, fileOption) as f:
        lines = f.readlines()
    return lines

def load_ppdb_data():
    with open("data/pickled/ppdb.pickle", 'rb') as f:
        return pickle.load(f,encoding='latin1')

def cosine_similarity_by_vector(vector1, vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

def alignment_score(ppdbLines, word1, word2):
    stemmer = stem.SnowballStemmer("english")
    if(stemmer.stem(word1) == stemmer.stem(word2)):
        return MAX_ALIGNMENT_SCORE
    else:
        return _paraphrase_score(ppdbLines, word1, word2)


def _paraphrase_score(ppdbLines, word1, word2):
    paraphrase_score = MIN_ALIGNMENT_SCORE
    #lines = getLinesFromFile(file,'r')
    if ppdbLines.get(word1) != None:
        tuples = [tup for tup in ppdbLines.get(word1) if tup[0] == word2]
        if tuples:
            paraphrase_score =  max(tuples)[1]
    return paraphrase_score



def get_stanparse_data():
    with open(_pickled_data_folder, 'rb') as f:
        return pickle.load(f)

def get_stanford_idx(x):
    i = x.rfind('-')
    return x[:i].lower(), int(x[(i+1):])

def get_stanparse_depths():
    with open(_pickled_data_folder3, 'rb') as f:
        return pickle.load(f)

def get_aligned_data():
    with open(_pickled_data_folder2, 'rb') as f:
        return pickle.load(f)

def get_dataset(filename='url-versions-2015-06-14-clean.csv'):
    folder = "emergent/url-versions-2015-06-14-clean.csv"
    return pd.read_csv(os.path.join(folder))

def calc_depths(grph, n=0, d=0, depths=None):
    if depths is None:
        depths = {n: d}
    sx = grph.get(n)
    if sx:
        for s in sx:
            depths[s] = d+1
            calc_depths(grph, s, d+1, depths)
    return depths

def build_dep_graph(deps):
    dep_graph = {}
    dep_graph_labels = {}

    for d in deps:
        rel, head, dep = d
        _, head_idx = get_stanford_idx(head)
        _, dep_idx = get_stanford_idx(dep)
        dep_graph.setdefault(head_idx, set()).add(dep_idx)
        dep_graph_labels[(head_idx, dep_idx)] = rel
    return dep_graph, dep_graph_labels

def normalize_word(w):
    #nltk.download('punkt')
    #nltk.download('wordnet')
    _wnl = nltk.WordNetLemmatizer()
    return _wnl.lemmatize(w).lower()

def get_tokenized_lemmas(s):
    return [normalize_word(t) for t in nltk.word_tokenize(s)]

###############################################################
#Neg
###############################################################
def get_w2v_model():
    folder = "pickled/w2vec-data.pickle"
    return gensim.models.KeyedVectors.load_word2vec_format(folder,
                                                       binary=True)
def cosine_sim(u, v):
    """Returns the cosine similarity between two 1-D vectors, u and v"""
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))



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

def calculate_overlap(claim,headline):
    wordnet_lemmatizer = stem.WordNetLemmatizer()
    punctuations = "?:!.,;"
    lemmas = {0:[],1:[]}
    item = 0
    for sentence in [claim,headline]:
        lemmas[item] = [wordnet_lemmatizer.lemmatize(word).lower() for word in nltk.word_tokenize(sentence) if word not in punctuations]
        item = item + 1
    common_lemma = set(lemmas[0]).intersection(lemmas[1])
    union_lemma = set(lemmas[0]).union(lemmas[1])
    overlap = float(len(common_lemma) / len(union_lemma))
    return overlap


