import numpy as np
import pandas as pd
import re

from gensim.models import Word2Vec,KeyedVectors
from nltk.tokenize import sent_tokenize, word_tokenize
import networkx as nx

from logger import Logger
from utils import cosine_similarity_by_vector, alignment_score

class FeatureExtraction():

    def __init__(self, logger):
        self.logger = logger

    def get_BoW_feature(self, claim, headline):

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

    def get_question_feature(self, claim, headline):
        if "?" in headline:
            self.logger.log("Feature Question completed.")
            return 1
        else:
            self.logger.log("Feature Question completed.")
            return 0

#RootDist Zhang
#Neg Zhang
#PPDB Priya
    def get_ppdb_feature(self, claim, headline):
        dummy_word = ";"
        length_claim = len(claim.split())
        length_headline = len(headline.split())

        word_pairs = [(c,h) for c in sent_tokenize(claim) for h in sent_tokenize(headline)]
        if(length_claim < length_headline):
            word_pairs.append([(dummy_word, word) for word in headline.split()])
            normalize_constant = length_claim
        else:
            word_pairs.append([(word, dummy_word) for word in claim.split()])
            normalize_constant = len(headline.split())

        ppdb_lexical_file = self.logger.config_dict['PPDB_LEXICAL']
        edges = [(i,j,alignment_score(ppdb_lexical_file, i,j)) for (i,j) in enumerate(word_pairs)]
        alignment_graph = nx.Graph()
        alignment_graph.add_nodes_from([word_pairs[i][0] for i in range(0,len(word_pairs))], bipartite=0)
        alignment_graph.add_nodes_from([word_pairs[i][1] for i in range(0, len(word_pairs))], bipartite=1)
        #alignment_graph.add_edges_from(edges)
        #[alignment_graph.add_edge(i,j,weight=alignment_score(ppdb_lexical_file, i, j)) for i, j in enumerate(word_pairs)]
        for i,j in enumerate(word_pairs):
            alignment_graph.add_edge(i,j,weight=alignment_score(ppdb_lexical_file, i, j))
        # kuhn-munkers algo
        max_scorings = nx.max_weight_matching(alignment_graph)
        weights = [alignment_graph.get_edge_data(node_pair[0],node_pair[1]) for node_pair in max_scorings]
        return np.sum([weights[i]['weight'] for i in range(0,len(weights))]) / normalize_constant


#SVO Priya
    def get_svo_feature(selfself,claim, headline):
        #get svo pair for claim and headline

        # for each svo pair get label
        return

#word2vec Priya
    def get_word2vec_cosine_similarity(self, model, claim, headline):
        headline_vectors = [model.wv[word] for word in headline.lower().split()]
        headline_vector = np.prod(headline_vectors, axis=0)

        claim_vectors = [model.wv[word] for word in claim.lower().split()]
        claim_vector = np.prod(claim_vectors, axis=0)
        return cosine_similarity_by_vector(claim_vector, headline_vector)




    #need to confirm how data is passed here
    def compute_features(self,data_dict):
        self.logger.log("Start computing features...")
        features = []

        #iteration over each row will change based on datastructure
        for claim,headline in enumerate(data_dict.items()):
            bow = self.get_BoW_feature( claim, headline)
            q = self.get_question_feature( claim, headline)
            #root_dist =
            #neg =
            ppdb = self.get_ppdb_feature(claim,headline)
            svo = self.get_svo_feature(claim, headline)

            #model = KeyedVectors.load_word2vec_format(self.logger.config_dict['GOOGLE_NEWS_VECTOR_FILE'], binary=True)
            model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)

            word2vec_feature = self.get_word2vec_cosine_similarity(model, claim, headline)
            #features.append([bow, q, root_dist, neg, ppdb, svo, word2vec_feature])
            features.append([word2vec_feature])

        #colnames = ["BoW","Q","RootDist","Neg","PPDB","SVO","word2vec"]
        colnames = ["word2vec"]
        self.logger.log("Finished computing features", show_time=True)

        return pd.DataFrame(features,colnames = colnames)


logger = Logger(show = True, html_output = True, config_file = "config.txt")
feature_extraction = FeatureExtraction(logger)
data_dict = {'this is an apple': 'the apple was red', 'Cherries are sweet': 'fruits are sweet'}
feature_extraction.compute_features(data_dict)
