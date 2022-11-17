from __future__ import print_function, division
import re
import numpy as np
import scipy.sparse
# from collections import Counter

unicode = str


DEFAULT_NUM_WORDS = 27000
FILENAMES = {"g_wiki": "glove.6B.300d.small.txt",
             "g_twitter": "glove.twitter.27B.200d.small.txt",
             "g_crawl": "glove.840B.300d.small.txt",
             "w2v": "GoogleNews-word2vec.small.txt",
             "w2v_large": "GoogleNews-word2vec.txt"}


def dedup(seq):
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


def safe_word(w):
    # ignore words with numbers, etc.
    # [a-zA-Z\.'_\- :;\(\)\]] for emoticons
    return (re.match(r"^[a-z_]*$", w) and len(w) < 20 and not re.match(r"^_*$", w))


def to_utf8(text, errors='strict', encoding='utf8'):
    """Convert a string (unicode or bytestring in `encoding`), to bytestring in utf8."""
    if isinstance(text, unicode):
        return text.encode('utf8')
    # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
    return unicode(text, encoding, errors=errors).encode('utf8')


class WordEmbedding:
    def __init__(self):
        self.thresh = None
        self.max_words = None
        self.desc = ''
        # list of words 
        self.words = None
        # embeddings corresponding to words
        self.embeddings = None
        # dict mapping word to its index
        self.word2index = None 
        
        self._neighbors = None 


    def load_embeddings(self, f_name):
        '''
            Read input file and build, word list, embeddings list and vocabulary
        '''
        embeds = []
        words = []

        with open(f_name, "r", encoding='utf8') as f:
            for line in f:
                s = line.split()
                v = np.array([float(x) for x in s[1:]])

                # if len(embeds) and embeds[-1].shape !=v.shape:
                #     print("Got weird line", line)
                #     continue

                words.append(s[0])
                embeds.append(v)
        self.embeddings = np.array(embeds, dtype='float32')
        print(self.embeddings.shape)
        self.words = words
        self.build_vocab()
        norms = np.linalg.norm(self.embeddings, axis=1)
        if max(norms)-min(norms) > 0.0001:
            self.normalize()

    def build_vocab(self):
        '''
            Build vocabulary from words list
        '''
        self.word2index = {w: i for i, w in enumerate(self.words)}

    def get_embedding(self, word):
        '''
            return embedding vector for a word
        '''
        return self.embeddings[self.word2index[word]]

    def diff(self, word1, word2):
        '''
            subtract embedding vectors and return their difference
        '''
        v = self.embeddings[self.word2index[word1]] - self.embeddings[self.word2index[word2]]
        return v/np.linalg.norm(v)

    def normalize(self):
        self.desc += ", normalize"
        self.embeddings /= np.linalg.norm(self.embeddings, axis=1)[:, np.newaxis]
        self.build_vocab()

    def get_neighbors(self, thresh, max_words):
        '''
            Neightbors are those words which have similarity score more than a 
            certain threshold. User can pick a thresh hold value and neighboring 
            words are collected accordingly.
        '''

        # do not recalculate if done already for a certain setting
        if self._neighbors is not None and self.thresh == thresh and self.max_words == max_words:
            return

        print("Computing neighbors")
        # threshold to get the similarity of words e.g. words are similar if it is > 0.5
        self.thresh = thresh
        # how many words you need 
        self.max_words = max_words
        # embeddings of the words 
        vecs = self.embeddings[:max_words]
        # calculate similarity 
        dots = vecs.dot(vecs.T)
        # get row, col index and value of similar words 
        dots = scipy.sparse.csr_matrix(dots * (dots >= 1-thresh/2))

        
        rows, cols = dots.nonzero()
        # count the number of values, not important at this time 
        # nums = list(Counter(rows).values())
        # print("Mean:", np.mean(nums)-1)
        # print("Median:", np.median(nums)-1)

        # get index and value of words which are similar apart from the word itself
        rows, cols, vecs = zip(*[(i, j, vecs[i]-vecs[j]) for i, j, x in zip(rows, cols, dots.data) if i<j])

        self._neighbors = rows, cols, np.array([v/np.linalg.norm(v) for v in vecs])

    def get_analogy_based_thresh(self, v, thresh=1, topn=500, max_words=50000):
        """
            Score = cos(a-c, b-d) if |b-d|^2 < thresh, otherwise 0
            return a list of tuple of: 
            (she occupation, he occupation, score)
        """
        vecs, vocab = self.embeddings[:max_words], self.words[:max_words]
        self.get_neighbors(thresh, max_words)
        rows, cols, vecs = self._neighbors
        # print('neighbor size:', vecs.shape)
        scores = vecs.dot(v/np.linalg.norm(v))
        
        # print('size of score:', scores.shape)

        pi = np.argsort(-abs(scores))

        ans = []
        she_set = set()
        he_set = set()
        for i in pi:
            if abs(scores[i])<0.001:
                break
            she = rows[i] if scores[i] > 0 else cols[i]
            he = cols[i] if scores[i] > 0 else rows[i]
            if she in she_set or he in he_set:
                continue
            she_set.add(she)
            he_set.add(he)
            ans.append((vocab[she], vocab[he], abs(scores[i])))
            if len(ans)==topn:
                break

        return ans