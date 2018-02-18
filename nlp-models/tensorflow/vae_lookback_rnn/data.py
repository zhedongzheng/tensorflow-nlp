from config import args

import sklearn
import numpy as np
import tensorflow as tf


class BaseDataLoader(object):
    def __init__(self):
        self.enc_inp = None
        self.dec_inp = None # word dropout
        self.dec_out = None
        self.word2idx = None
        self.idx2word = None

    def next_batch(self):
        for i in range(0, len(self.enc_inp), args.batch_size):
            yield (self.enc_inp[i : i + args.batch_size],
                   self.dec_inp[i : i + args.batch_size],
                   self.dec_out[i : i + args.batch_size])


class IMDB(BaseDataLoader):
    def __init__(self):
        super().__init__()
        self._index_from = 4
        self.word2idx = self._load_word2idx()
        self.idx2word = self._load_idx2word()
        self.enc_inp, self.dec_inp_full, self.dec_out = self._load_data()
        self.dec_inp = self._word_dropout(self.dec_inp_full)
    
    def _load_data(self):
        (X_train, _), (X_test, _) = tf.contrib.keras.datasets.imdb.load_data(
            num_words=None, index_from=self._index_from)
        print("Data Loaded")
        X_tr_enc_inp, X_tr_dec_inp, X_tr_dec_out = self._pad(X_train)
        X_te_enc_inp, X_te_dec_inp, X_te_dec_out = self._pad(X_test)
        enc_inp = np.concatenate((X_tr_enc_inp, X_te_enc_inp))
        dec_inp = np.concatenate((X_tr_dec_inp, X_te_dec_inp))
        dec_out = np.concatenate((X_tr_dec_out, X_te_dec_out))
        print("Data Padded")
        return enc_inp, dec_inp, dec_out

    def _pad(self, X):
        enc_inp = []
        dec_inp = []
        dec_out = []
        for x in X:
            x = x[1:]
            if len(x) < args.max_len:
                enc_inp.append(x + [self.word2idx['<pad>']]*(args.max_len-len(x)))
                dec_inp.append([self.word2idx['<start>']] + x + [self.word2idx['<pad>']]*(args.max_len-len(x)))
                dec_out.append(x + [self.word2idx['<end>']] + [self.word2idx['<pad>']]*(args.max_len-len(x)))
            else:
                truncated = x[:args.max_len]
                enc_inp.append(truncated)
                dec_inp.append([self.word2idx['<start>']] + truncated)
                dec_out.append(truncated + [self.word2idx['<end>']])
                
                truncated = x[-args.max_len:]
                enc_inp.append(truncated)
                dec_inp.append([self.word2idx['<start>']] + truncated)
                dec_out.append(truncated + [self.word2idx['<end>']])
                
                
        return np.array(enc_inp), np.array(dec_inp), np.array(dec_out)

    def _load_word2idx(self):
        word2idx = tf.contrib.keras.datasets.imdb.get_word_index()
        print("Word Index Loaded")
        word2idx = {k: (v+self._index_from) for k, v in word2idx.items()}
        word2idx['<pad>'] = 0
        word2idx['<start>'] = 1
        word2idx['<unk>'] = 2
        word2idx['<end>'] = 3
        return word2idx

    def _load_idx2word(self):
        idx2word = {i: w for w, i in self.word2idx.items()}
        idx2word[-1] = '-1' # exception handling
        idx2word[4] = '4'   # exception handling
        return idx2word

    def _word_dropout(self, x):
        is_dropped = np.random.binomial(1, args.word_dropout_rate, x.shape)
        fn = np.vectorize(lambda x, k: self.word2idx['<unk>'] if (k and (x not in range(4))) else x)
        return fn(x, is_dropped)

    def shuffle(self):
        self.enc_inp, self.dec_inp, self.dec_out, self.dec_inp_full = sklearn.utils.shuffle(
            self.enc_inp, self.dec_inp, self.dec_out, self.dec_inp_full)

    def update_word_dropout(self):
        self.dec_inp = self._word_dropout(self.dec_inp_full)


def main():
    def word_dropout_test(d):
        print(' '.join(d.idx2word[idx] for idx in d.dec_inp_full[20]))
        print(' '.join(d.idx2word[idx] for idx in d.dec_inp[20]))

    def update_word_dropout_test(d):
        d.update_word_dropout()
        print(' '.join(d.idx2word[idx] for idx in d.dec_inp_full[20]))
        print(' '.join(d.idx2word[idx] for idx in d.dec_inp[20]))

    def next_batch_test(d):
        enc_inp, dec_inp, dec_out = next(d.next_batch())
        print(enc_inp.shape, dec_inp.shape, dec_out.shape)

    imdb = IMDB()
    word_dropout_test(imdb)
    update_word_dropout_test(imdb)
    next_batch_test(imdb)


if __name__ == '__main__':
    main()