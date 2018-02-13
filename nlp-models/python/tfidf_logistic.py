import numpy as np
import time

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression


class TfidfLogistic:
    def __init__(self, vocab_size):
        self.tfidf_model = TfidfTransformer()
        self.logistic_model = LogisticRegression(solver='saga')
        self.vocab_size = vocab_size
    
    
    def fit(self, X_train, y_train):
        X_train = self.transform(X_train)
        self.logistic_model.fit(X_train, y_train)


    def predict(self, X_test):
        X_test = self.transform(X_test)
        return self.logistic_model.predict(X_test)


    def transform(self, X):
        t0 = time.time()
        X_DT = np.zeros((len(X), self.vocab_size))
        for i, indices in enumerate(X):
            for idx in indices:
                X_DT[i, idx] += 1
        print("%.2f secs ==> Document-Term Matrix"%(time.time()-t0))

        t0 = time.time()
        X = self.tfidf_model.fit_transform(X_DT).toarray()
        print("%.2f secs ==> TF-IDF transform"%(time.time()-t0))
        return X
# end class