
from sklearn.model_selection._split import _BaseKFold


import numpy as np
import nltk
import os

from utils import get_stanparse_data,get_stanford_idx,get_stanparse_depths,get_aligned_data,get_dataset,calc_depths,build_dep_graph,normalize_word,get_tokenized_lemmas,get_w2v_model,cosine_sim
import corenlp




import numpy as np
import pandas as pd

from nltk.tokenize import sent_tokenize, word_tokenize
import networkx as nx

from logger import Logger
from utils import BoW, Q

class FeatureExtraction():

    def __init__(self, logger):
        self.logger = logger

class ClaimKFold(_BaseKFold):

    def __init__(self, data, n_folds=10, shuffle=False):
        super(ClaimKFold, self).__init__(len(data), n_folds, None, False, None)
        self.shuffle = shuffle
        self.data = data.copy()
        self.data['iloc_index'] = range(len(self.data))


    def __len__(self):
        return self.n_folds

#RootDist Zhang
def rootDist():
    nltk.download('punkt')
    nltk.download('wordnet')

    VALID_STANCE_LABELS = ['for', 'against', 'observing']

    _data_folder = os.path.join(os.path.dirname(__file__), 'emergent')
    df_clean_train = get_dataset('url-versions-2015-06-14-clean-train.csv')
    example = df_clean_train.ix[0, :]
    #print(example)
    dep_parse_data = get_stanparse_data()
    example_parse = dep_parse_data[example.articleId]
    print(example_parse)
    grph, grph_labels = build_dep_graph(example_parse['sentences'][0]['dependencies'])
    print("The grph is:",grph)
    example_parse['sentences'][0]['dependencies']
    print("The example_parse is",example_parse)
    print("the grph_label is:",grph_labels)
    calc_depths(grph)
    depths = get_stanparse_depths()
    d = depths['116a3920-c41c-11e4-883c-a7fa7a3c5066']
    print(d)

    sp_data = get_stanparse_data()

    more_than_one_sentence = [v for v in sp_data.values() if len(v['sentences']) > 1]
    more_than_one_sentence[0]
    print(more_than_one_sentence[0])


#################################################################
#Neg Zhang
#################################################################
def neg():
    cId, aId = '4893f040-a5c6-11e4-aa4f-ff16e52e0d56', '53faf1e0-a5c6-11e4-aa4f-ff16e52e0d56'
    aligned_data = get_aligned_data()
    print(aligned_data)
    aligned_data[(cId, aId)]
    df = get_dataset()
    #print(df.shape)
    penis = df[df.articleId == aId]
    #print(penis.claimHeadline[569])
    claim = get_tokenized_lemmas(penis.claimHeadline[569])
    article = get_tokenized_lemmas(penis.articleHeadline[569])
    #print(claim)
    [(claim[i], article[j]) for (i,j) in aligned_data[(cId, aId)]]
    print(claim)
    print(article)
    #w2vec_model = get_w2v_model()
    #cosine_sim(w2vec_model['having'], w2vec_model['finding'])
    stanparse_data = get_stanparse_data()
    stanparse_data[cId]['sentences'][0]['dependencies']
    stanparse_data[aId]['sentences']#[0]['dependencies']
    #cosine(w2vec_model['safe'], w2vec_model['stolen'])
    stanparse_data['6d937d80-3c20-11e4-bc0b-3f922b93930d']['sentences'][0]['dependencies']
    stanparse_data['ee3af700-3ab9-11e4-bc0b-3f922b93930d']['sentences'][0]['dependencies']
    get_tokenized_lemmas('not having a girlfriend')
    s = 'because of not having a girlfriend'
    s.find('not')
    toks = get_tokenized_lemmas(s)


#PPDB Priya
def get_ppdb_feature(self, claim, headline):
    min_score = -10
    maxscore = 10
    normalize_constant = min(len(claim.split(), headline.split()))
    word_pairs = [(c,h) for c in sent_tokenize(claim) for h in sent_tokenize(headline)]
    #edges = ((i,j,kuhn_munkres_score(i,j)) for (i,j) in enumerate(word_pairs))
    alignment_graph = nx.Graph()
    #alignment_graph.add_edges_from(edges)
    return


#SVO Priya
#word2vec Priya
rootDist()
logger = Logger(show = True, html_output = True, config_file = "config.txt")
feature_extraction = FeatureExtraction(logger)
data_dict = {'this is an apple': 'the apple was red', 'Cherries are sweet': 'fruits are sweet'}
feature_extraction.compute_features(data_dict)
