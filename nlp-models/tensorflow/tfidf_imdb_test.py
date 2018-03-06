import tensorflow as tf
import numpy as np
import time

from logistic_regression import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer


VOCAB_SIZE = 20000


def transform(X, tfidf):
    t0 = time.time()
    count = np.zeros((len(X), VOCAB_SIZE))
    for i, indices in enumerate(X):
        for idx in indices:
            count[i, idx] += 1
    print("%.2f secs ==> Document-Term Matrix"%(time.time()-t0))

    t0 = time.time()
    X = tfidf.fit_transform(count).toarray()
    print("%.2f secs ==> TF-IDF transform"%(time.time()-t0))
    return X


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = tf.contrib.keras.datasets.imdb.load_data(
        num_words=VOCAB_SIZE)
    
    tfidf = TfidfTransformer()
    X_train = transform(X_train, tfidf)
    X_test = transform(X_test, tfidf)
    print(X_train.shape, X_test.shape)

    model = LogisticRegression(VOCAB_SIZE, 2)
    model.fit(X_train, y_train, n_epoch=2, batch_size=32, val_data=(X_test, y_test))
    y_pred = model.predict(X_test)
    
    final_acc = (y_pred == y_test).mean()
    print("final testing accuracy: %.4f" % final_acc)
