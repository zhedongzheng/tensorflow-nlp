from __future__ import print_function
import sys
import nltk
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


class LSA:
    def __init__(self, stopwords, n_compoents=20):
        self.stopwords = stopwords
        self.n_compoents = n_compoents
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
        lsa = TruncatedSVD(self.n_compoents, n_iter=100)
        lsa.fit(self.X)
        terms = self.vectorizer.get_feature_names()
        for i, comp in enumerate(lsa.components_): # lsa.components_ is V of USV, of shape (concepts, terms)
            terms_comp = zip(terms, comp)
            sorted_tc = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:top_k]
            print("Concept %d :" % i, end = ' ')
            for term_comp in sorted_tc:
                print(term_comp[0], end=' | ')
            print()
    # end method        


    def tokenize(self, string):
        string = string.lower()
        tokens = nltk.tokenize.word_tokenize(string) # more powerful split()
        tokens = [token for token in tokens if len(token)>2] # remove too short words
        tokens = [token for token in tokens if token not in self.stopwords] # remove stopwords
        tokens = [token for token in tokens if not any(c.isdigit() for c in token)] # remove any token that contains number
        return tokens
    # end method
# end class