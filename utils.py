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


_pickled_data_folder = "E:\git\PycharmProjects\\NLP_Project\pickled\\stanparse-data.pickle"
_pickled_data_folder2 = "E:\git\PycharmProjects\\NLP_Project\pickled\\aligned-data.pickle"
MIN_ALIGNMENT_SCORE = -10
MAX_ALIGNMENT_SCORE = 10


def cosine_similarity_by_vector(vector1, vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

def alignment_score(file, word1, word2):
    stemmer = stem.SnowballStemmer("english")
    if(stemmer.stem(word1) == stemmer.stem(word2)):
        return MAX_ALIGNMENT_SCORE
    else:
        return _paraphrase_score(file, word1, word2)


def _paraphrase_score(file, word1, word2):
    paraphrase_score = MIN_ALIGNMENT_SCORE
    with open(file, 'r') as ppdb_lex:
        lines = ppdb_lex.readlines()
        matches = re.findall(word1,lines)
        if matches:
            final_matches = re.findall(word2,matches)
            if final_matches:
                for line in final_matches:
                    match_build = line.split("|||")
                    if ((match_build[1] == word1 & match_build[2] == word2) | (match_build[1] == word2 & match_build[2] == word1)):
                        metrics = [metric for metric in match_build[3].split(" ")]
                        scorings = {key_val[0]: int(key_val[1]) for key_val in [scoring.split("=") for scoring in metrics]}
                        paraphrase_score = -np.log(scorings['p(f|e)']) -np.log(scorings['p(f|e)']) - np.log(scorings['p(e|f,LHS)']) - np.log(scorings['p(f|e,LHS)']) + 0.3 * (-np.log(scorings['p(LHS|e)'])) + 0.3 * (-np.log(scorings['p(LHS|f)'])) + 100 * scorings['RarityPenalty']
    return paraphrase_score


def get_stanparse_data():
    with open(_pickled_data_folder, 'rb') as f:
        return pickle.load(f)

def get_stanford_idx(x):
    i = x.rfind('-')
    return x[:i].lower(), int(x[(i+1):])

def get_stanparse_depths():
    with open(_pickled_data_folder, 'rb') as f:
        return pickle.load(f)

def get_aligned_data():
    with open(_pickled_data_folder2, 'rb') as f:
        return pickle.load(f)

def get_dataset(filename='url-versions-2015-06-14-clean.csv'):
    folder = "E:\git\PycharmProjects\\NLP_Project\emergent\\url-versions-2015-06-14-clean.csv"
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
    folder = "E:\git\PycharmProjects\\NLP_Project\pickled\\w2vec-data.pickle"
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
