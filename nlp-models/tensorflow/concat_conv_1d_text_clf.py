from base_text_clf import BaseTextClassifier

import tensorflow as tf


class Conv1DClassifier(BaseTextClassifier):
    def __init__(self, seq_len, vocab_size, n_out, sess=tf.Session(),
                 n_filters=250, embedding_dims=50):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.n_filters = n_filters
        self.embedding_dims = embedding_dims
        self.n_out = n_out
        self.sess = sess
        self._pointer = None
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.add_input_layer()
        self.add_word_embedding()

        kernels = [3, 4, 5]
        parallels = []
        for k in kernels:
            p = self.add_conv1d(self.n_filters//len(kernels), kernel_size=k)
            p = self.add_global_pooling(p)
            parallels.append(p)
        self.merge_layers(parallels)

        self.add_hidden_layer()
        self.add_output_layer()   
        self.add_backward_path()
    # end method build_graph


    def add_input_layer(self):
        self.X = tf.placeholder(tf.int32, [None, self.seq_len])
        self.Y = tf.placeholder(tf.int64, [None])
        self.keep_prob = tf.placeholder(tf.float32)
        self.lr = tf.placeholder(tf.float32)
        self._pointer = self.X
    # end method add_input_layer


    def add_word_embedding(self):
        embedding = tf.get_variable('encoder', [self.vocab_size,self.embedding_dims], tf.float32)
        embedded = tf.nn.embedding_lookup(embedding, self._pointer)
        self._pointer = tf.nn.dropout(embedded, self.keep_prob)
    # end method add_word_embedding_layer


    def add_conv1d(self, n_filters, kernel_size):
        Y = tf.layers.conv1d(
            inputs = self._pointer,
            filters = n_filters,
            kernel_size  = kernel_size,
            activation = tf.nn.relu)
        return Y
    # end method add_conv1d_layer


    def add_global_pooling(self, x):
        Y = tf.layers.max_pooling1d(
            inputs = x,
            pool_size = x.get_shape().as_list()[1],
            strides = 1)
        Y = tf.reshape(Y, [-1, Y.get_shape().as_list()[-1]])
        return Y
    # end method add_global_maxpool_layer


    def merge_layers(self, layers):
        self._pointer = tf.concat(layers, axis=-1)
    # end method merge_layers


    def add_hidden_layer(self):
        self._pointer = tf.layers.dense(self._pointer, self.n_filters, tf.nn.relu)
    # end method merge_layers


    def add_output_layer(self):
        self.logits = tf.layers.dense(self._pointer, self.n_out)
    # end method add_output_layer


    def add_backward_path(self):
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, 1), self.Y), tf.float32))
    # end method add_backward_path
# end class