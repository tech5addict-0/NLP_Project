import re
from random import randint

import networkx as nx
import numpy as np
import pandas as pd
import stanfordnlp
from gensim.models import KeyedVectors
from sklearn.model_selection._split import _BaseKFold
from nltk.parse.stanford import StanfordDependencyParser as sdp
from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.parse.dependencygraph import DependencyGraph
from pntl.tools import Annotator

import utils
from logger import Logger


class FeatureExtraction():

    def __init__(self, logger):
        self.logger = logger
        self.stanfordParseData = utils.get_stanparse_data()
        self.nlp = stanfordnlp.Pipeline()
        self.ppdbLines = utils.load_ppdb_data()
        self.word2vecModel = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True, limit=500000)
        rawWords = utils.get_rootDist_words()
        self.rootDistWords = [word.strip() for word in rawWords]
        self.svoMapping = {
            "equivalence" : 0,
            "forwardEntailment" : 1,
            "backwardEntailment" : 3,
            "independence" : 4,
            "noRelation" : 5
        }
        print("Initialisation complete")

    def get_BoW_feature(self, claim, headline, bag):

        wordsHeadline = re.sub("[^\w]", " ", headline).split()
        wordsHeadline_cleaned = [w.lower() for w in wordsHeadline]
        wordsHeadline_cleaned = sorted(list(set(wordsHeadline_cleaned)))

        result = [0 for i in range(len(bag))]
        for hw in wordsHeadline_cleaned:
            for i, cw in enumerate(bag):
                if hw == cw:
                    result[i] += 1

        self.logger.log("Feature Bag of Words completed.")
        return result

    def get_question_feature(self, claim, headline):
        if "?" in headline:
            self.logger.log("Feature Question completed.")
            return 1
        else:
            self.logger.log("Feature Question completed.")
            return 0

    def get_key(self,dict, value):
        return [k for k, v in dict.items() if v == value]

    def rootDist(self, headline):
        my_depen = []
        words_to_check = set(headline.lower().split()).intersection(self.rootDistWords)
        indexed_headline = {word:index+1 for index,word in enumerate(headline.lower().split())}
        doc = self.nlp(headline.lower())
        for dep_edge in doc.sentences[0].dependencies:
            my_depen.append((dep_edge[0].text + "-" + dep_edge[0].index, dep_edge[2].text + "-" + dep_edge[2].index))

        graph = nx.Graph(my_depen)
        root_dist = -1
        root_dists = []
        for target in words_to_check:
            target_node = target + "-" + str(indexed_headline[target])
            try:
                root_dists.append(nx.shortest_path_length(graph, source='ROOT-0', target=target_node))
            except nx.NodeNotFound:
                continue
        if root_dists:
            root_dist = min(root_dists)
        return root_dist


    def neg(self,claim,headline):
        count = 0
        #cId, aId = '4893f040-a5c6-11e4-aa4f-ff16e52e0d56', '53faf1e0-a5c6-11e4-aa4f-ff16e52e0d56'
        df_clean_train = utils.get_dataset('url-versions-2015-06-14-clean-train.csv')
        example = df_clean_train.ix[count, :]
        cId, aId = example.claimId,example.articleId
        aligned_data = utils.get_aligned_data()
        print(aligned_data)
        aligned_data[(cId, aId)]
        df = utils.get_dataset()
        # print(df.shape)
        pen = df[df.articleId == aId]
        # print(pen)
        claim = utils.get_tokenized_lemmas(pen.claimHeadline[569])
        print("the penis is:", pen.claimHeadline[569])
        article = utils.get_tokenized_lemmas(pen.articleHeadline[569])
        print("the claim is", claim)
        [(claim[i], article[j]) for (i, j) in aligned_data[(cId, aId)]]
        print(claim)
        print(article)
        w2vec_model = utils.get_w2v_model()
        # cosine_sim(w2vec_model['having'], w2vec_model['finding'])
        stanparse_data = utils.get_stanparse_data()
        stanparse_data[cId]['sentences'][0]['dependencies']
        stanparse_data[aId]['sentences']  # [0]['dependencies']
        # cosine(w2vec_model['safe'], w2vec_model['stolen'])
        stanparse_data['6d937d80-3c20-11e4-bc0b-3f922b93930d']['sentences'][0]['dependencies']
        stanparse_data['ee3af700-3ab9-11e4-bc0b-3f922b93930d']['sentences'][0]['dependencies']
        utils.get_tokenized_lemmas('not having a girlfriend')
        s = 'because of not having a girlfriend'
        s.find('not')
        toks = utils.get_tokenized_lemmas(s)
        # filter(lambda (i, t): t == 'not', enumerate(toks))

    def neg(self, claim, headline):
        is_negated = False
        # for c_word in claim.lower().spli():
        #     for h_word in headline.lower().split():
        #         #check whether negation
        return is_negated

#PPDB Priya
    def get_ppdb_feature(self, claim, headline):
        dummy_word = ";"
        length_claim = len(claim.split())
        length_headline = len(headline.split())

        claim_nodes = claim.lower().split()
        headline_nodes = headline.lower().split()
        word_pairs = [(c,h) for c in claim_nodes for h in headline_nodes]

        if(length_claim < length_headline):
            num_words_to_add = length_headline - length_claim
            dummy_words = [dummy_word+str(i) for i in range(0,num_words_to_add)]
            [word_pairs.append((dummy_word,word)) for dummy_word in dummy_words for word in headline_nodes]
            claim_nodes = claim_nodes + dummy_words
            normalize_constant = length_claim
        else:
            num_words_to_add = length_claim - length_headline
            dummy_words = [dummy_word + str(i) for i in range(0, num_words_to_add)]
            [word_pairs.append((word, dummy_word)) for word in claim_nodes for dummy_word in dummy_words]
            headline_nodes = headline_nodes + dummy_words
            normalize_constant = length_headline

        #ppdb_lexical_file = self.logger.config_dict['PPDB_LEXICAL']
        ppdb_lexical_file = "../ppdb-2.0-tldr"
        alignment_graph = nx.Graph()
        nodes = {i:j for i, j in enumerate(claim_nodes + headline_nodes)}
        alignment_graph.add_nodes_from(list(range(0,len(claim_nodes))), bipartite=0)
        alignment_graph.add_nodes_from(list(range(len(claim_nodes),2*len(claim_nodes))), bipartite=1)
        for i in list(range(0, len(claim_nodes))):
            for j in list(range(len(claim_nodes), 2 * len(claim_nodes))):
                alignment_graph.add_edge(i, j, weight=utils.alignment_score(self.ppdbLines, nodes[i], nodes[j]))
        # kuhn-munkers algo
        max_scorings = nx.max_weight_matching(alignment_graph)
        weights = [alignment_graph.get_edge_data(node_pair[0],node_pair[1]) for node_pair in max_scorings]
        return np.sum([weights[i]['weight'] for i in range(0,len(weights))]) / normalize_constant


#SVO Priya
    def get_svo_feature(self,claim, headline):
        #Get the SVO triples fro claim and headline
        svoDict = {0:{}, 1:{}}
        items = 0
        for sentence in [claim, headline]:
            doc = self.nlp(sentence)
            allDependencies = doc.sentences[0].dependencies
            for count,dependency in enumerate(allDependencies):
                if dependency[1].lower() == "root":
                    svoDict[items]["v"] = dependency[2].text
                elif (dependency[1].lower() == "nn" or dependency[1].lower() == "nsubj"):
                    svoDict[items]["s"] = dependency[2].text
                elif (dependency[1].lower() == "dep" or dependency[1].lower() == "obj"):
                        svoDict[items]["o"] = dependency[2].text
            items = items + 1

        # for each svo pair get label
        label = []
        for svo in ["s","v","o"]:
            try:
                c_word = svoDict[0][svo]
                h_word = svoDict[1][svo]
                if self.ppdbLines.get(c_word) != None:
                    tuples = [tup for tup in self.ppdbLines.get(c_word) if tup[0] == h_word]
                label.append(self.svoMapping[tuples[0][2]])
            except (KeyError, IndexError, UnboundLocalError,IndexError):
                label.append(self.svoMapping["noRelation"])
                continue
        return label

#word2vec Priya
    def get_word2vec_cosine_similarity(self, claim, headline):
        headline_vectors = []
        for word in headline.lower().split():
            try:
                h_vector_array = self.word2vecModel.wv[word]
            except KeyError:
                h_vector_array = np.ones(300)
            headline_vectors.append(h_vector_array)
        headline_vector = np.prod(headline_vectors, axis=0)

        claim_vectors = []
        for word in claim.lower().split():
            try:
                c_vector_array = self.word2vecModel.wv[word]
            except KeyError:
                c_vector_array = np.ones(300)
            claim_vectors.append(c_vector_array)
        claim_vector = np.prod(claim_vectors, axis=0)

        return utils.cosine_similarity_by_vector(claim_vector, headline_vector)



    def compute_features(self,data_dict):
        self.logger.log("Start computing features...")
        features = []
        count = 0
        bag = utils.createBagTrain(data_dict)
        for claimId in data_dict:
            print(data_dict[claimId]["claim"])
            for articleId in data_dict[claimId]["articles"]:
                article = data_dict[claimId]["articles"][articleId]
                stance = article[1]
                headline = article[0]
                print(headline)
                claim = data_dict[claimId]["claim"]

                #get all the features for the claim and headline
                bow = self.get_BoW_feature( claim, headline, bag)
                q = self.get_question_feature( claim, headline)
                root_dist = self.rootDist(headline)
                # neg = self.neg(claim,headline)
                ppdb = self.get_ppdb_feature(claim,headline)
                svo = self.get_svo_feature(claim, headline)
                #word2vec_feature = self.get_word2vec_cosine_similarity(claim, headline)
                word2vec_feature = self.get_word2vec_cosine_similarity(claim, headline)
                #features.append([bow, q, root_dist, neg, ppdb, svo, word2vec_feature, stance, claimId])
                features.append(list(bow) + [q,ppdb,root_dist] + list(svo) +[word2vec_feature] + [stance,claimId])
                count = count + 1
        #colnames = ["BoW","Q","RootDist","Neg","PPDB","SVO","word2vec","stance", "claimId"]
        #colnames = ["BoW","Q","PPDB","SVO","stance", "claimId"]
        self.logger.log("Finished computing features", show_time=True)
        return pd.DataFrame(features)

class ClaimKFold(_BaseKFold):
    def __init__(self, data, n_folds=10, shuffle=False):
        super(ClaimKFold, self).__init__(len(data), n_folds, None, False, None)
        self.shuffle = shuffle
        self.data = data.copy()
        self.data['iloc_index'] = range(len(self.data))

    def __len__(self):
        return self.n_folds

#logger = Logger(show = True, html_output = True, config_file = "config.txt")
#fe = FeatureExtraction(logger)
#minD = fe.rootDist("20-Year-Old Quarter Pounder Looks About the Same")
#svo = fe.get_svo_feature("Barack Obama was born in Hawaii", "He was on drugs")
#neg = fe.neg("a","b")
#w2v = fe.get_word2vec_cosine_similarity("Barack Obama was born in Hawaii", "He was on drugs")
