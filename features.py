import re
from random import randint

import networkx as nx
import numpy as np
import pandas as pd
import stanfordnlp
from sklearn.model_selection._split import _BaseKFold
from nltk.parse.stanford import StanfordDependencyParser as sdp
from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.parse.dependencygraph import DependencyGraph

import utils
from logger import Logger


class FeatureExtraction():

    def __init__(self, logger):
        self.logger = logger
        self.stanfordParseData = utils.get_stanparse_data()
        self.nlp = stanfordnlp.Pipeline()
        self.ppdbLines = utils.load_ppdb_data()
        #self.word2vecModel = KeyedVectors.load_word2vec_format(self.logger.config_dict['GOOGLE_NEWS_VECTOR_FILE'], binary=True)
        #self.word2vecModel = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
        print("Done")

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

    def get_key(self,dict, value):
        return [k for k, v in dict.items() if v == value]

    # RootDist Zhang
    def rootDist_old(self, claim,headline,count):
        df_clean_train = utils.get_dataset('url-versions-2015-06-14-clean-train.csv')
        example = df_clean_train.ix[count, :]
        dep_parse_data = utils.get_stanparse_data()
        example_parse = dep_parse_data[example.articleId]
        #print(corenlp.ParseTree(claim))
        print(example_parse)
        grph, grph_labels = utils.build_dep_graph(example_parse['sentences'][0]['dependencies'])
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
        dicta = utils.calc_depths(grph)
        print("the root dist is:",dicta.get(depth_key))
        #depths = get_stanparse_depths()
        #d = depths['116a3920-c41c-11e4-883c-a7fa7a3c5066']
        #print("the depths:", d)
        #e = list(d.items())
        #print(e[-1])
        sp_data = utils.get_stanparse_data()

        more_than_one_sentence = [v for v in sp_data.values() if len(v['sentences']) > 1]
        u_depent = more_than_one_sentence[count]
        print(u_depent['sentences'][0]['dependencies'])
        print(more_than_one_sentence[count])



    #################################################################
    # Neg Zhang
    #################################################################
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
                label.append(tuples[0][2])
            except (KeyError, IndexError, UnboundLocalError,IndexError):
                label.append("noRelation")
                continue
        return label

#word2vec Priya
    def get_word2vec_cosine_similarity(self, claim, headline):
        headline_vectors = [self.word2vecModel.wv[word] for word in headline.lower().split()]
        headline_vector = np.prod(headline_vectors, axis=0)

        claim_vectors = [self.word2vecModel.wv[word] for word in claim.lower().split()]
        claim_vector = np.prod(claim_vectors, axis=0)
        return utils.cosine_similarity_by_vector(claim_vector, headline_vector)



    def compute_features2(self,data_dict):
        self.logger.log("Start computing features...")
        features = []
        count = 0
        for claimId in data_dict:
            for articleId in data_dict[claimId]["articles"]:
                article = data_dict[claimId]["articles"][articleId]
                stance = article[1]
                headline = article[0]
                claim = data_dict[claimId]["claim"]

                #get all the features for the claim and headline
                bow = self.get_BoW_feature( claim, headline)
                q = self.get_question_feature( claim, headline)
                #root_dist = self.rootDist(claim,headline,count)
                # neg = self.neg(claim,headline)
                ppdb = self.get_ppdb_feature(claim,headline)
                svo = self.get_svo_feature(claim, headline)
                #word2vec_feature = self.get_word2vec_cosine_similarity(claim, headline)
                #features.append([bow, q, root_dist, neg, ppdb, svo, word2vec_feature, stance, claimId])
                features.append([bow, q,ppdb,svo, stance,claimId])
                count = count + 1
        #colnames = ["BoW","Q","RootDist","Neg","PPDB","SVO","word2vec","stance", "claimId"]
        colnames = ["BoW","Q","PPDB","SVO","stance", "claimId"]
        self.logger.log("Finished computing features", show_time=True)
        return pd.DataFrame(features,columns = colnames)

class ClaimKFold(_BaseKFold):
    def __init__(self, data, n_folds=10, shuffle=False):
        super(ClaimKFold, self).__init__(len(data), n_folds, None, False, None)
        self.shuffle = shuffle
        self.data = data.copy()
        self.data['iloc_index'] = range(len(self.data))

    def __len__(self):
        return self.n_folds

