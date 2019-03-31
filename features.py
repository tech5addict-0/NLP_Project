import os

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
import pandas as pd
import gensim
import nltk
nltk.download('punkt')
nltk.download('wordnet')

VALID_STANCE_LABELS = ['for', 'against', 'observing']

_data_folder = os.path.join(os.path.dirname(__file__), 'emergent')
_pickled_data_folder = "E:\git\PycharmProjects\\NLP_Project\pickled\\stanparse-data.pickle"
_pickled_data_folder2 = "E:\git\PycharmProjects\\NLP_Project\pickled\\aligned-data.pickle"



def get_dataset(filename='url-versions-2015-06-14-clean.csv'):
    folder = "E:\git\PycharmProjects\\NLP_Project\emergent\\url-versions-2015-06-14-clean.csv"
    return pd.read_csv(os.path.join(folder))

def get_stanparse_data():
    with open(_pickled_data_folder, 'rb') as f:
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
example = df_clean_train.ix[0, :]
#print(example)
dep_parse_data = get_stanparse_data()
example_parse = dep_parse_data[example.articleId]


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

grph, grph_labels = build_dep_graph(example_parse['sentences'][0]['dependencies'])


#print(example.articleHeadline)
print(grph)
example_parse['sentences'][0]['dependencies']
print(example_parse)
print(grph_labels)

def get_stanparse_depths():
    with open(_pickled_data_folder, 'rb') as f:
        return pickle.load(f)

def calc_depths(grph, n=0, d=0, depths=None):
    if depths is None:
        depths = {n: d}
    sx = grph.get(n)
    if sx:
        for s in sx:
            depths[s] = d+1
            calc_depths(grph, s, d+1, depths)
    return depths
calc_depths(grph)

def get_aligned_data():
    with open(_pickled_data_folder2, 'rb') as f:
        return pickle.load(f)

_wnl = nltk.WordNetLemmatizer()


def normalize_word(w):
    return _wnl.lemmatize(w).lower()

def get_tokenized_lemmas(s):
    return [normalize_word(t) for t in nltk.word_tokenize(s)]





depths = get_stanparse_depths()
d = depths['116a3920-c41c-11e4-883c-a7fa7a3c5066']
print(d)

sp_data = get_stanparse_data()
#print(len(sp_data.values()[0]['sentences']))


more_than_one_sentence = [v for v in sp_data.values() if len(v['sentences']) > 1]
more_than_one_sentence[0]
print(more_than_one_sentence[0])



#Neg Zhang



#################################################################

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

def get_w2v_model():
    folder = "E:\git\PycharmProjects\\NLP_Project\pickled\\w2vec-data.pickle"
    return gensim.models.KeyedVectors.load_word2vec_format(folder,
                                                       binary=True)

def cosine_sim(u, v):
    """Returns the cosine similarity between two 1-D vectors, u and v"""
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

w2vec_model = get_w2v_model()
cosine_sim(w2vec_model['having'], w2vec_model['finding'])
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
#SVO Priya
#word2vec Priya