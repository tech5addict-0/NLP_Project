import numpy as np
import pandas as pd
import re
import nltk
import os
import corenlp
import networkx as nx
from gensim.models import Word2Vec,KeyedVectors
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection._split import _BaseKFold
from utils import get_stanparse_data,get_stanford_idx,get_stanparse_depths,get_aligned_data,get_dataset,calc_depths,build_dep_graph,normalize_word,get_tokenized_lemmas,get_w2v_model,cosine_sim
from pycorenlp import StanfordCoreNLP
import stanfordnlp

from logger import Logger
from utils import cosine_similarity_by_vector, alignment_score
from random import randint

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

    def get_key(dict, value):
        return [k for k, v in dict.items() if v == value]

    # RootDist Zhang
    def rootDist(claim,headline,count):
        df_clean_train = get_dataset('url-versions-2015-06-14-clean-train.csv')
        example = df_clean_train.ix[count, :]
        dep_parse_data = get_stanparse_data()
        example_parse = dep_parse_data[example.articleId]
        #print(corenlp.to_text())
        print(example_parse)
        grph, grph_labels = build_dep_graph(example_parse['sentences'][0]['dependencies'])
        print("The grph is:", grph)
        example_parse['sentences'][0]['dependencies']
        print("The example_parse is", example_parse)
        print("the grph_label is:", grph_labels)
        key = FeatureExtraction.get_key(grph_labels,"neg")
        print(len(key))
        if(len(key)):
            depth_key = key[0][0]
        else:
            depth_key = 0
        dicta = calc_depths(grph)
        print("the root dist is:",dicta.get(depth_key))
        #depths = get_stanparse_depths()
        #d = depths['116a3920-c41c-11e4-883c-a7fa7a3c5066']
        #print("the depths:", d)
        #e = list(d.items())
        #print(e[-1])
        sp_data = get_stanparse_data()

        more_than_one_sentence = [v for v in sp_data.values() if len(v['sentences']) > 1]
        u_depent = more_than_one_sentence[count]
        print(u_depent['sentences'][0]['dependencies'])
        print(more_than_one_sentence[count])

    #################################################################
    # Neg Zhang
    #################################################################
    def neg(count,claim,headline):
        dummy_word = ";"

        claim = "Apple will sell 19 million Apple Watches in 2015"
        headline = "BMO forecasts 19M Apple Watch sales in 2015, with more than half selling in holiday season"
        length_claim = len(claim.split())
        length_headline = len(headline.split())
        stanfordnlp.download('en')
        nlp = stanfordnlp.Pipeline()
        doc = nlp(claim)
        doc.sentences[0].print_dependencies()

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
    def compute_features2(self,data_dict):
        self.logger.log("Start computing features...")
        features = []
        count = 0
       #iteration over each row will change based on datastructure
        for claim,headline in enumerate(data_dict.items()):
            bow = self.get_BoW_feature( claim, headline)
            q = self.get_question_feature( claim, headline)
            root_dist = self.rootDist(claim,headline,count)
            neg = self.neg(claim,headline)
            ppdb = self.get_ppdb_feature(claim,headline)
            svo = self.get_svo_feature(claim, headline)

            #model = KeyedVectors.load_word2vec_format(self.logger.config_dict['GOOGLE_NEWS_VECTOR_FILE'], binary=True)
            model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)

            word2vec_feature = self.get_word2vec_cosine_similarity(model, claim, headline)
            #features.append([bow, q, root_dist, neg, ppdb, svo, word2vec_feature])
            features.append([word2vec_feature])
            count = count + 1
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

class ClaimKFold(_BaseKFold):
    def __init__(self, data, n_folds=10, shuffle=False):
        super(ClaimKFold, self).__init__(len(data), n_folds, None, False, None)
        self.shuffle = shuffle
        self.data = data.copy()
        self.data['iloc_index'] = range(len(self.data))

    def __len__(self):
        return self.n_folds

claim = "Apple will sell 19 million Apple Watches in 2015"
headline = "BMO forecasts 19M Apple Watch sales in 2015, with more than half selling in holiday season"
FeatureExtraction.neg(0,claim,headline)
#logger = Logger(show = True, html_output = True, config_file = "config.txt")
#feature_extraction = FeatureExtraction(logger)
#data_dict = {'this is an apple': 'the apple was red', 'Cherries are sweet': 'fruits are sweet'}
#feature_extraction.compute_features(data_dict)
