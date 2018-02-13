import sys
import nltk
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD


class LSA:
    def __init__(self, stopwords):
        self.stopwords = stopwords
        self.token2idx = {}
        self.idx2token = {}
        self.documents = []
        self.token_idx = 0
        self.X = None
    # end constructor


    def fit(self, documents):
        for line in documents:
            if int(sys.version[0]) == 2:
                line = line.decode('ascii', 'ignore')
            tokens = self.tokenize(line)
            self.documents.append(tokens)
            for token in tokens:
                if token not in self.token2idx: # tests the existence of a key in a dict
                    self.token2idx[token] = self.token_idx
                    self.idx2token[self.token_idx] = token
                    self.token_idx += 1
        assert len(self.idx2token) == len(self.token2idx), "The length of idx2token is unequal to token2idx"

        self.X = np.zeros((len(self.token2idx), len(self.documents))) # Term-Document Matrix
        for i, tokens in enumerate(self.documents):
            self.X[:, i] = self.tokens2vec(tokens)
    # end method fit

    
    def transform_plot(self):
        model = TruncatedSVD()
        X_2d = model.fit_transform(self.X)
        plt.scatter(X_2d[:, 0], X_2d[:, 1])
        for i in range(len(self.idx2token)):
            plt.annotate(s=self.idx2token[i], xy=(X_2d[i, 0], X_2d[i, 1]))
        plt.show()
    # end method plot


    def tokenize(self, string):
        string = string.lower()
        tokens = nltk.tokenize.word_tokenize(string) # more powerful split()
        tokens = [token for token in tokens if len(token)>2] # remove too short words
        tokens = [token for token in tokens if token not in self.stopwords] # remove stopwords
        tokens = [token for token in tokens if not any(c.isdigit() for c in token)] # remove any token that contains number
        return tokens
    # end method tokenize


    def tokens2vec(self, tokens):
        vec = np.zeros(len(self.token2idx))
        for token in tokens:
            idx = self.token2idx[token]
            vec[idx] += 1
        return vec
    # end method tokens2vec
