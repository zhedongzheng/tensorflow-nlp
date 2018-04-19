import tensorflow as tf

from scipy.sparse import isspmatrix_csr
from base_text_clf import BaseTextClassifier


class LogisticRegression(BaseTextClassifier):
    def __init__(self, vocab_size, n_out, sess=tf.Session()):
        self.vocab_size = vocab_size
        self.n_out = n_out
        self.sess = sess
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.X = tf.placeholder(tf.float32, [None, self.vocab_size])
        self.Y = tf.placeholder(tf.int64, [None])
        self.lr = tf.placeholder(tf.float32)
        self.keep_prob = tf.placeholder(tf.float32)

        self.logits = tf.layers.dense(self.X, self.n_out)

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.acc = tf.reduce_mean(tf.cast(tf.equal(
            tf.argmax(self.logits, 1), self.Y), tf.float32))
    # end method build_graph


    def gen_batch(self, arr, batch_size):
        for i in range(0, arr.shape[0], batch_size):
            if isspmatrix_csr(arr):
                yield arr[i : i+batch_size].toarray()
            else:
                yield arr[i : i+batch_size]
    # end method gen_batch
# end class