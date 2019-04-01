import numpy as np
import re
import pandas as pd
import os
import nltk
import gensim
from gensim.models import Word2Vec,KeyedVectors

from logger import Logger
try:
    import cPickle as pickle
except:
    import pickle

_pickled_data_folder = "E:\git\PycharmProjects\\NLP_Project\pickled\\stanparse-data.pickle"
_pickled_data_folder2 = "E:\git\PycharmProjects\\NLP_Project\pickled\\aligned-data.pickle"


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

def get_word2vec_cosine_similarity(self, model, claim, headline):
    headline_vectors = [model.wv[word] for word in headline.lower().split()]
    headline_vector = np.prod(headline_vectors, axis=0)

    claim_vectors = [model.wv[word] for word in claim.lower().split()]
    claim_vector = np.prod(claim_vectors, axis=0)
    return self.cosine_similarity_by_vector(claim_vector, headline_vector)

def cosine_similarity_by_vector(self, vector1, vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))


    #need to confirm how data is passed here
def compute_features(self,data_dict):
    self.logger.log("Start computing features...")
    features = []

    #iteration over each row will change based on datastructure
    for claim,headline in enumerate(data_dict.items()):
        bow = BoW(self, claim, headline)
        q = Q(self, claim, headline)
        #root_dist =
        #neg =
        #ppdb =
        #svo =

        #model = KeyedVectors.load_word2vec_format(self.logger.config_dict['GOOGLE_NEWS_VECTOR_FILE'], binary=True)
        model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)

        word2vec_feature = self.get_word2vec_cosine_similarity(model, claim, headline)
        #features.append([bow, q, root_dist, neg, ppdb, svo, word2vec_feature])
        features.append([word2vec_feature])

        #colnames = ["BoW","Q","RootDist","Neg","PPDB","SVO","word2vec"]
        colnames = ["word2vec"]
        self.logger.log("Finished computing features", show_time=True)

        return pd.DataFrame(features,colnames = colnames)