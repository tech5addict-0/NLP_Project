import numpy as np
import pandas as pd

from gensim.models import Word2Vec,KeyedVectors
from nltk.tokenize import sent_tokenize, word_tokenize
import networkx as nx

from logger import Logger
from random import randint

class FeatureExtraction():

    def __init__(self, logger):
        self.logger = logger

#BoW marjolein
#Q marjolein
#RootDist Zhang
#Neg Zhang
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





    def get_word2vec_cosine_similarity(self, model, claim, headline):
        headline_vectors = [model.wv[word] for word in headline.lower().split()]
        headline_vector = np.prod(headline_vectors, axis=0)

        claim_vectors = [model.wv[word] for word in claim.lower().split()]
        claim_vector = np.prod(claim_vectors, axis=0)
        return self.cosine_similarity_by_vector(claim_vector, headline_vector)

    def cosine_similarity_by_vector(self, vector1, vector2):
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))


    #need to confirm how data is passed here
    def compute_features2(self,data_dict):
        self.logger.log("Start computing features...")
        features = []
        

        #iteration over each row will change based on datastructure
        for claim,headline in enumerate(data_dict.items()):
            #bow =
            #q =
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

    
    def compute_features(self,data_dict):
        #print(data_dict)
        features = []
        #print(data_dict)
        for claimId in data_dict:
            #print(data_dict[claimId])
            for article in data_dict[claimId]["articles"]:
                stance = data_dict[claimId]["articles"][article][1]
                features.append([randint(1,20), randint(1,20), stance, claimId])
        colNames = ["feat1", "feat2", "stance", "claimId"]
        return pd.DataFrame(features, columns=colNames)


#logger = Logger(show = True, html_output = True, config_file = "config.txt")
#feature_extraction = FeatureExtraction(logger)
#data_dict = {'this is an apple': 'the apple was red', 'Cherries are sweet': 'fruits are sweet'}
#feature_extraction.compute_features(data_dict)
