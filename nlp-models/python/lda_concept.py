from __future__ import print_function
import sys
import nltk
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer


class LDA:
    def __init__(self, stopwords, n_components=20):
        self.stopwords = stopwords
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer()
        self.X = None
    # end constructor


    def fit(self, documents):
        _documents = []
        for line in documents:
            if int(sys.version[0]) == 2:
                line = line.decode('ascii', 'ignore')
            tokens = self.tokenize(line)
            _documents.append(' '.join(tokens))
        self.X = self.vectorizer.fit_transform(_documents)
    # end method


    def concepts(self, top_k=5):
        lda = LatentDirichletAllocation(self.n_components, learning_offset=50, max_iter=100)
        lda.fit(self.X)
        terms = self.vectorizer.get_feature_names()
        self.print_top_words(lda, terms, top_k)
    # end method        


    def tokenize(self, string):
        string = string.lower()
        tokens = nltk.tokenize.word_tokenize(string) # more powerful split()
        tokens = [token for token in tokens if len(token)>2] # remove too short words
        tokens = [token for token in tokens if token not in self.stopwords] # remove stopwords
        tokens = [token for token in tokens if not any(c.isdigit() for c in token)] # remove any token that contains number
        return tokens
    # end method


    def print_top_words(self, model, feature_names, n_top_words):
        # .components_ is V of USV, of shape (concepts, terms)
        for topic_idx, term_vals in enumerate(model.components_):
            message = "Topic #%d: " % topic_idx
            message += " ".join([feature_names[i]
                                for i in term_vals.argsort()[:-n_top_words - 1:-1]])
            print(message)
        print()
    # end method
# end class