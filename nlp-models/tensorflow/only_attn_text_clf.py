import tensorflow as tf

from base_text_clf import BaseTextClassifier
from utils import learned_positional_encoding, sinusoidal_positional_encoding


class OnlyAttentionClassifier(BaseTextClassifier):
    def __init__(self, seq_len, vocab_size, n_out, sess=tf.Session(), model_dim=50, pos_dim=20):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.pos_dim = pos_dim
        self.n_out = n_out
        self.sess = sess
        self._pointer = None
        self.build_graph()
    # end constructor
 
 
    def build_graph(self):
        self.add_input_layer()
        self.add_word_embedding()
        self.add_self_attention()
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
        embedding = tf.get_variable('encoder', [self.vocab_size, self.model_dim], tf.float32)
        x = tf.nn.embedding_lookup(embedding, self._pointer)
        position = learned_positional_encoding(x, self.pos_dim)
        x = tf.concat((x, position), -1)
        self._pointer = tf.nn.dropout(x, self.keep_prob)
    # end method add_word_embedding_layer


    def add_self_attention(self):
        x = self._pointer
        masks = tf.sign(self.X)
        
        # alignment
        align = tf.squeeze(tf.layers.dense(x, 1, tf.tanh), -1)
        # masking
        paddings = tf.fill(tf.shape(align), float('-inf'))
        align = tf.where(tf.equal(masks, 0), paddings, align)
        # probability
        align = tf.expand_dims(tf.nn.softmax(align), -1)
        # weighted sum
        x = tf.squeeze(tf.matmul(tf.transpose(x, [0,2,1]), align), -1)

        self._pointer = x
    # end method add_self_attention


    def add_output_layer(self):
        self.logits = tf.layers.dense(self._pointer, self.n_out)
    # end method add_output_layer


    def add_backward_path(self):
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, 1), self.Y), tf.float32))
    # end method add_backward_path
# end class