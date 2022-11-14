from __future__ import print_function, division
import re
import sys
import numpy as np
import scipy.sparse
from sklearn.decomposition import PCA


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

        print("*** Reading data from " + f_name)
        if f_name.endswith(".bin"):
            import gensim.models
            model =gensim.models.KeyedVectors.load_word2vec_format(f_name, binary=True)
            words = sorted([w for w in model.vocab], key=lambda w: model.vocab[w].index)
            embeds = [model[w] for w in words]
        else:
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
        # self.n, self.d = self.embeddings.shape
        # assert self.n == len(self.words) == len(self.word2index)
        # self._neighbors = None
        # print(self.n, "words of dimension", self.d, ":", ", ".join(self.words[:4] + ["..."] + self.words[-4:]))

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

#     def shrink(self, numwords):
#         self.desc += ", shrink " + str(numwords)
#         self.filter_words(lambda w: self.word2index[w]<numwords)

#     def filter_words(self, test):
#         """
#         Keep some words based on test, e.g. lambda x: x.lower()==x
#         """
#         self.desc += ", filter"
#         kept_indices, words = zip(*[[i, w] for i, w in enumerate(self.words) if test(w)])
#         self.words = list(words)
#         self.embeddings = self.embeddings[kept_indices, :]
#         self.build_vocab()

#     def save(self, filename):
#         with open(filename, "w") as f:
#             f.write("\n".join([w+" " + " ".join([str(x) for x in v]) for w, v in zip(self.words, self.embeddings)]))
#         print("Wrote", self.n, "words to", filename)

#     def save_w2v(self, filename, binary=True):
#         with open(filename, 'wb') as fout:
#             fout.write(to_utf8("%s %s\n" % self.embeddings.shape))
#             # store in sorted order: most frequent words at the top
#             for i, word in enumerate(self.words):
#                 row = self.embeddings[i]
#                 if binary:
#                     fout.write(to_utf8(word) + b" " + row.tostring())
#                 else:
#                     fout.write(to_utf8("%s %s\n" % (word, ' '.join("%f" % val for val in row))))

#     def remove_directions(self, directions): #directions better be orthogonal
#         self.desc += ", removed"
#         for direction in directions:
#             self.desc += " "
#             if type(direction) is np.ndarray:
#                 v = direction / np.linalg.norm(direction)
#                 self.desc += "vector "
#             else:
#                 w1, w2 = direction
#                 v = self.diff(w1, w2)
#                 self.desc += w1 + "-" + w2
#             self.embeddings = self.embeddings - self.embeddings.dot(v)[:, np.newaxis].dot(v[np.newaxis, :])
#         self.normalize()

    def compute_neighbors_if_necessary(self, thresh, max_words):
        # thresh = float(thresh) # dang python 2.7!
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

        from collections import Counter
        rows, cols = dots.nonzero()
        # count the number of values, not important at this time 
        nums = list(Counter(rows).values())
        print("Mean:", np.mean(nums)-1)
        print("Median:", np.median(nums)-1)

        # get index and value of words which are similar apart from the word itself
        rows, cols, vecs = zip(*[(i, j, vecs[i]-vecs[j]) for i, j, x in zip(rows, cols, dots.data) if i<j])

        self._neighbors = rows, cols, np.array([v/np.linalg.norm(v) for v in vecs])

#     def neighbors(self, word, thresh=1):
#         dots = self.embeddings.dot(self.v(word))
#         return [self.words[i] for i, dot in enumerate(dots) if dot >= 1-thresh/2]

#     def more_words_like_these(self, words, topn=50, max_freq=100000):
#         v = sum(self.v(w) for w in words)
#         dots = self.embeddings[:max_freq].dot(v)
#         thresh = sorted(dots)[-topn]
#         words = [w for w, dot in zip(self.words, dots) if dot>=thresh]
#         return sorted(words, key=lambda w: self.v(w).dot(v))[-topn:][::-1]

    def best_analogies_dist_thresh(self, v, thresh=1, topn=500, max_words=50000):
        """Metric is cos(a-c, b-d) if |b-d|^2 < thresh, otherwise 0
        """
        vecs, vocab = self.embeddings[:max_words], self.words[:max_words]
        self.compute_neighbors_if_necessary(thresh, max_words)
        rows, cols, vecs = self._neighbors
        # print('neighbor size:', vecs.shape)
        scores = vecs.dot(v/np.linalg.norm(v))
        
        # print('size of score:', scores.shape)

        pi = np.argsort(-abs(scores))

        ans = []
        usedL = set()
        usedR = set()
        for i in pi:
            if abs(scores[i])<0.001:
                break
            row = rows[i] if scores[i] > 0 else cols[i]
            col = cols[i] if scores[i] > 0 else rows[i]
            if row in usedL or col in usedR:
                continue
            usedL.add(row)
            usedR.add(col)
            ans.append((vocab[row], vocab[col], abs(scores[i])))
            if len(ans)==topn:
                break

        return ans


# def viz(analogies):
#     print("\n".join(str(i).rjust(4)+a[0].rjust(29) + " | " + a[1].ljust(29) + (str(a[2]))[:4] for i, a in enumerate(analogies)))


# def text_plot_words(xs, ys, words, width = 90, height = 40, filename=None):
#     PADDING = 10 # num chars on left and right in case words spill over
#     res = [[' ' for i in range(width)] for j in range(height)]
#     def rescale(nums):
#         a = min(nums)
#         b = max(nums)
#         return [(x-a)/(b-a) for x in nums]
#     print("x:", (min(xs), max(xs)), "y:",(min(ys),max(ys)))
#     xs = rescale(xs)
#     ys = rescale(ys)
#     for (x, y, word) in zip(xs, ys, words):
#         i = int(x*(width - 1 - PADDING))
#         j = int(y*(height-1))
#         row = res[j]
#         z = list(row[i2] != ' ' for i2 in range(max(i-1, 0), min(width, i + len(word) + 1)))
#         if any(z):
#             continue
#         for k in range(len(word)):
#             if i+k>=width:
#                 break
#             row[i+k] = word[k]
#     string = "\n".join("".join(r) for r in res)
# #     return string
#     if filename:
#         with open(filename, "w", encoding="utf8") as f:
#             f.write(string)
#         print("Wrote to", filename)
#     else:
#         print(string)


# def doPCA(pairs, embedding, num_components = 10):
#     matrix = []
#     for a, b in pairs:
#         center = (embedding.v(a) + embedding.v(b))/2
#         matrix.append(embedding.v(a) - center)
#         matrix.append(embedding.v(b) - center)
#     matrix = np.array(matrix)
#     pca = PCA(n_components = num_components)
#     pca.fit(matrix)
#     # bar(range(num_components), pca.explained_variance_ratio_)
#     return pca


# def drop(u, v):
#     return u - v * u.dot(v) / v.dot(v)
