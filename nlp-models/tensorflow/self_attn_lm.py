from utils import learned_positional_encoding, embed_seq, pointwise_feedforward, layer_norm
import tensorflow as tf
import numpy as np


class LM:
    def __init__(self, text, seq_len, embedding_dims=30, hidden_units=128, n_layers=2,
                 num_heads=8, dropout_rate=0.1, sess=tf.Session()):
        self.sess = sess
        self.text = text
        self.seq_len = seq_len
        self.embedding_dims = embedding_dims
        self.hidden_units = hidden_units
        self.n_layers = n_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.preprocessing()
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.add_input_layer()       
        self.add_decoder()
        self.add_backward_path()
    # end method


    def add_input_layer(self):
        self.sequence = tf.placeholder(tf.int32, [None, self.seq_len])
        self.is_training = tf.placeholder(tf.bool)
        self._batch_size = tf.shape(self.sequence)[0]
    # end method


    def add_decoder(self):
        def forward(X, reuse=None):
            with tf.variable_scope('embed_seq', reuse=reuse):
                encoded = embed_seq(
                    X, self.vocab_size, self.hidden_units, zero_pad=True, scale=True)
            with tf.variable_scope('pos_enc', reuse=reuse):
                encoded += learned_positional_encoding(
                    X, self.hidden_units, zero_pad=False, scale=False)
            encoded = tf.layers.dropout(encoded, self.dropout_rate, training=self.is_training)
            for i in range(self.n_layers):
                with tf.variable_scope('attn%d'%i, reuse=reuse):
                    encoded = self_multihead_attn(encoded,
                        num_units=self.hidden_units, num_heads=self.num_heads,
                        dropout_rate=self.dropout_rate, is_training=self.is_training)
                with tf.variable_scope('feedforward%d'%i, reuse=reuse):
                    encoded = pointwise_feedforward(encoded,
                        num_units=[4*self.hidden_units, self.hidden_units], activation=tf.nn.elu)
            return tf.layers.dense(encoded, self.vocab_size)


        self.logits = forward(self._decoder_input(self.sequence))
        self.predicted_ids = tf.argmax(self.logits, -1, output_type=tf.int32)
    # end method


    def add_backward_path(self):
        targets = self._decoder_output(self.sequence)
        self.loss = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(
            logits = self.logits,
            targets = targets,
            weights = tf.to_float(tf.ones_like(targets)),
            average_across_timesteps = True,
            average_across_batch = True))
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
    # end method


    def preprocessing(self):
        text = self.text 
        chars = set(text)
        self.char2idx = {c: i+3 for i, c in enumerate(chars)}
        self.char2idx['<pad>'] = 0
        self.char2idx['<start>'] = 1
        self.char2idx['<end>'] = 2
        self.vocab_size = len(self.char2idx)
        print('Vocabulary size:', self.vocab_size)
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        self.idx2char[-1] = '-1'
        self.indexed = np.array([self.char2idx[char] for char in list(text)])
    # end method


    def next_batch(self, batch_size, text_iter_step):
        window = self.seq_len * batch_size
        for i in range(0, len(self.indexed)-window-1, text_iter_step):
            yield self.indexed[i : i+window].reshape(-1, self.seq_len)
    # end method


    def fit(self, text_iter_step=25, n_epoch=1, batch_size=64):
        n_batch = (len(self.indexed) - self.seq_len*batch_size - 1) // text_iter_step
        self.sess.run(tf.global_variables_initializer()) # initialize all variables
        
        for epoch in range(n_epoch):
            for local_step, seq_batch in enumerate(self.next_batch(batch_size, text_iter_step)):
                _, train_loss = self.sess.run([self.train_op, self.loss],
                                              {self.sequence: seq_batch,
                                               self.is_training: True})
                if local_step % 10 == 0:
                    print ('Epoch %d/%d | Batch %d/%d | train loss: %.4f' %
                          (epoch+1, n_epoch, local_step, n_batch, train_loss))
                if local_step % 100 == 0:
                    self.decode()
    # end method


    def decode(self):
        print()
        autoregr = np.zeros([1, self.seq_len], np.int32)
        for j in range(self.seq_len):
            preds = self.sess.run(self.predicted_ids,
                                 {self.sequence: autoregr,
                                  self._batch_size: 1,
                                  self.is_training: False})
            autoregr[:, j] = preds[:, j]
            print(self.idx2char[autoregr[:, j][0]], end='')
        print()
        print()
    # end method


    def _decoder_output(self, X):
        _end = tf.fill([self._batch_size, 1], self.char2idx['<end>'])
        return tf.concat([X, _end], 1)
    # end method


    def _decoder_input(self, X):
        _start = tf.fill([self._batch_size, 1], self.char2idx['<start>'])
        return tf.concat([_start, X], 1)
    # end method
# end class


def self_multihead_attn(inputs, num_units, num_heads, dropout_rate, is_training):
    """
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q]
      keys: A 3d tensor with shape of [N, T_k, C_k]
    """
    if num_units is None:
        num_units = queries.get_shape().as_list[-1]
    T_q = T_k = inputs.get_shape().as_list()[1]                                    # max time length of query

    Q_K_V = tf.layers.dense(inputs, 3*num_units, tf.nn.relu)
    Q, K, V = tf.split(Q_K_V, 3, -1)

    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)                         # (h*N, T_q, C/h) 
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)                         # (h*N, T_k, C/h) 
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)                         # (h*N, T_k, C/h)

    align = tf.matmul(Q_, tf.transpose(K_, [0,2,1]))                               # (h*N, T_q, T_k)
    align = align / (K_.get_shape().as_list()[-1] ** 0.5)                          # scale
    
    paddings = tf.fill(tf.shape(align), float('-inf'))                             # exp(-large) -> 0

    # Future Binding
    lower_tri = tf.ones([T_q, T_k])                                                # (T_q, T_k)
    lower_tri = tf.linalg.LinearOperatorLowerTriangular(lower_tri).to_dense()      # (T_q, T_k)
    masks = tf.tile(tf.expand_dims(lower_tri,0), [tf.shape(align)[0], 1, 1])       # (h*N, T_q, T_k)
    align = tf.where(tf.equal(masks, 0), paddings, align)                          # (h*N, T_q, T_k)

    # Softmax
    align = tf.nn.softmax(align)                                                   # (h*N, T_q, T_k)

    align = tf.layers.dropout(align, dropout_rate, training=is_training)           # (h*N, T_q, T_k)

    # Weighted sum
    outputs = tf.matmul(align, V_)                                                 # (h*N, T_q, C/h)
    # Restore shape
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)              # (N, T_q, C)
    # Residual connection
    outputs += inputs                                                              # (N, T_q, C)   
    # Normalize
    outputs = layer_norm(outputs)                                                  # (N, T_q, C)
    return outputs
