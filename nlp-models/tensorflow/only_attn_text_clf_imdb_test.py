from only_attn_text_clf import OnlyAttentionClassifier
import tensorflow as tf
import numpy as np


vocab_size = 10000
seq_len = 500  # cut texts after this number of words (among top max_features most common words)
n_out = 2


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)

    X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=seq_len, padding='post')
    X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=seq_len, padding='post')
    print('X_train shape:', X_train.shape, 'X_test shape:', X_test.shape)

    clf = OnlyAttentionClassifier(seq_len, vocab_size, n_out)
    log = clf.fit(X_train, y_train, n_epoch=2, batch_size=32, keep_prob=0.8, en_exp_decay=True,
                  val_data=(X_test, y_test))
    pred = clf.predict(X_test)

    final_acc = (pred == y_test).mean()
    print("final testing accuracy: %.4f" % final_acc)
