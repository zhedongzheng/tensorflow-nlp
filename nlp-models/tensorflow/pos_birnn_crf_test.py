import pos
import numpy as np
import tensorflow as tf

from birnn_crf_clf import BiRNN_CRF
from sklearn.metrics import classification_report


SEQ_LEN = 20
BATCH_SIZE = 128
sample = ['I', 'love', 'you']


def to_train_seq(*args):
    data = []
    for x in args:
        data.append(iter_seq(x))
    return data


def to_test_seq(*args):
    data = []
    for x in args:
        x = x[: (len(x) - len(x) % SEQ_LEN)]
        data.append(np.reshape(x, [-1, SEQ_LEN]))
    return data


def iter_seq(x, text_iter_step=1):
    return np.array([x[i : i+SEQ_LEN] for i in range(0, len(x)-SEQ_LEN, text_iter_step)])


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, vocab_size, n_class, word2idx, tag2idx = pos.load_data()
    X_train, Y_train = to_train_seq(x_train, y_train)
    X_test, Y_test = to_test_seq(x_test, y_test)
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    clf = BiRNN_CRF(vocab_size, n_class)
    clf.fit(X_train, Y_train, keep_prob=0.8, n_epoch=1, batch_size=BATCH_SIZE)
    
    y_pred = clf.predict(X_test, batch_size=BATCH_SIZE)
    print(classification_report(Y_test.ravel(), y_pred.ravel(), target_names=tag2idx.keys()))
    
    idx2tag = {idx : tag for tag, idx in tag2idx.items()}
    labels = clf.infer([word2idx[w] for w in sample])
    print(' '.join(sample))
    print(' '.join([idx2tag[idx] for idx in labels]))
