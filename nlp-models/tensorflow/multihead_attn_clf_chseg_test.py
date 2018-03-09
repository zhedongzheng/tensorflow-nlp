# -*- coding: utf-8 -*-
import sys
import chseg
import numpy as np
import tensorflow as tf

from birnn_crf_clf import BiRNN_CRF
from collections import Counter
from multihead_attn_clf import Tagger
from sklearn.metrics import classification_report


SEQ_LEN = 50
N_CLASS = 4 # B: 0, M: 1, E: 2, S: 3
N_EPOCH = 1
BATCH_SIZE = 128
sample = '我来到大学读书，希望学到知识'
py = int(sys.version[0])


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


def iter_seq(x, text_iter_step=10):
    return np.array([x[i : i+SEQ_LEN] for i in range(0, len(x)-SEQ_LEN, text_iter_step)])


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, vocab_size, word2idx, idx2word = chseg.load_data()
    X_train, Y_train = to_train_seq(x_train, y_train)
    X_test, Y_test = to_test_seq(x_test, y_test)
    print('Vocab size: %d' % vocab_size)

    clf = Tagger(vocab_size, N_CLASS, SEQ_LEN, num_blocks=2)
    clf.fit(X_train, Y_train, n_epoch=N_EPOCH, batch_size=BATCH_SIZE)

    y_pred = clf.predict(X_test, batch_size=BATCH_SIZE)
    print(classification_report(Y_test.ravel(), y_pred.ravel(), target_names=['B', 'M', 'E', 'S']))

    chars = list(sample) if py == 3 else list(sample.decode('utf-8'))
    _test = [word2idx[w] for w in sample] + [0] * (SEQ_LEN-len(sample))
    labels = clf.infer(_test, len(sample))
    labels = labels[:len(sample)]
    res = ''
    for i, l in enumerate(labels):
        c = sample[i] if py == 3 else sample.decode('utf-8')[i]
        if l == 2 or l == 3:
            c += ' '
        res += c
    print(res)
    