import tensorflow as tf
import numpy as np
import math
from sklearn.utils import shuffle


class BiRNN_CRF:
    def __init__(
        self, vocab_size, n_out, embedding_dims=128, cell_size=128, n_layer=1,
        grad_clip=5.0, sess=tf.Session()):
        """
        Parameters:
        -----------
        vocab_size: int
            Vocabulary size
        cell_size: int
            Number of units in the rnn cell
        n_out: int
            Output dimensions
        sess: object
            tf.Session() object
        stateful: boolean
            If true, the final state for each batch will be used as the initial state for the next batch 
        """
        self.vocab_size = vocab_size
        self.embedding_dims = embedding_dims
        self.cell_size = cell_size
        self.n_layer = n_layer
        self.n_out = n_out
        self.grad_clip = grad_clip
        self.sess = sess
        self._pointer = None
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.add_input_layer()
        self.add_word_embedding_layer()
        self.add_bidirectional_dynamic_rnn()
        self.add_output_layer()
        self.add_crf_layer()
        self.add_backward_path()
    # end method build_graph


    def add_input_layer(self):
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None, None])
        self.X_seq_len = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32)
        self.lr = tf.placeholder(tf.float32)
        self._pointer = self.X
    # end method add_input_layer


    def add_word_embedding_layer(self):
        embedding = tf.get_variable('encoder', [self.vocab_size, self.embedding_dims], tf.float32,
                                     tf.random_uniform_initializer(-1.0, 1.0))
        embedded = tf.nn.embedding_lookup(embedding, self._pointer)
        self._pointer = tf.nn.dropout(embedded, self.keep_prob)
    # end method add_word_embedding_layer


    def rnn_cell(self):
        return tf.nn.rnn_cell.GRUCell(self.cell_size, kernel_initializer=tf.orthogonal_initializer())
    # end method lstm_cell


    def add_bidirectional_dynamic_rnn(self):
        birnn_out = self._pointer
        for n in range(self.n_layer):
            (out_fw, out_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = self.rnn_cell(), cell_bw = self.rnn_cell(),
                inputs = birnn_out,
                dtype = tf.float32,
                scope = 'birnn%d'%n)
            birnn_out = tf.concat((out_fw, out_bw), 2)
        self._pointer = birnn_out
    # end method add_dynamic_rnn


    def add_output_layer(self):
        self.logits = tf.layers.dense(tf.reshape(self._pointer, [-1, 2*self.cell_size]), self.n_out)
    # end method add_output_layer


    def add_crf_layer(self):
        logits = tf.reshape(self.logits, [tf.shape(self.X)[0], -1, self.n_out])
        with tf.variable_scope('crf_loss'):
            self.log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
                inputs = logits,
                tag_indices = self.Y,
                sequence_lengths = self.X_seq_len)
        with tf.variable_scope('crf_loss', reuse=True):
            transition_params = tf.get_variable('transitions', [self.n_out, self.n_out])
        self.viterbi_sequence, _ = tf.contrib.crf.crf_decode(
            logits, transition_params, self.X_seq_len)
    # end method add_crf_layer


    def add_backward_path(self):
        self.loss = tf.reduce_mean(-self.log_likelihood)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.viterbi_sequence, self.Y),
            tf.float32))

        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
        self.train_op = tf.train.AdamOptimizer(self.lr).apply_gradients(
            zip(clipped_gradients, params))
    # end method add_backward_path


    def fit(self, X, Y, n_epoch=10, batch_size=128, en_exp_decay=True, en_shuffle=True, keep_prob=1.0):
        global_step = 0
        self.sess.run(tf.global_variables_initializer()) # initialize all variables
        for epoch in range(n_epoch): # batch training
            if en_shuffle:
                X, Y = shuffle(X, Y)
                print("Data Shuffled")
            for local_step, (X_batch, Y_batch) in enumerate(zip(self.gen_batch(X, batch_size),
                                                                self.gen_batch(Y, batch_size))):
                lr = self.decrease_lr(en_exp_decay, global_step, n_epoch, len(X), batch_size)           
                _, loss, acc = self.sess.run([self.train_op, self.loss, self.acc],
                                             {self.X: X_batch, self.Y: Y_batch, self.lr: lr,
                                              self.X_seq_len: [X.shape[1]]*len(X_batch),
                                              self.keep_prob: keep_prob})
                global_step += 1
                if local_step % 50 == 0:
                    print ('Epoch %d/%d | Step %d/%d | train_loss: %.4f | train_acc: %.4f | lr: %.4f'
                           %(epoch+1, n_epoch, local_step, int(len(X)/batch_size), loss, acc, lr))
            # verbose
            print ("Epoch %d/%d | train_loss: %.4f | train_acc: %.4f |" % (epoch+1, n_epoch, loss, acc),
                   "lr: %.4f" % (lr) )
    # end method fit


    def predict(self, X_test, batch_size=128):
        batch_pred_list = []
        for X_test_batch in self.gen_batch(X_test, batch_size):
            batch_pred = self.sess.run(self.viterbi_sequence,
                                      {self.X: X_test_batch,
                                       self.X_seq_len: len(X_test_batch)*[X_test.shape[1]],
                                       self.keep_prob: 1.0})
            batch_pred_list.append(batch_pred)
        return np.vstack(batch_pred_list)
    # end method predict


    def infer(self, xs):
        viterbi_seq = self.sess.run(self.viterbi_sequence,
                                   {self.X: np.atleast_2d(xs),
                                    self.X_seq_len: np.atleast_1d(len(xs)),
                                    self.keep_prob: 1.0})
        return np.squeeze(viterbi_seq,0)
    # end method infer


    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i+batch_size]
    # end method gen_batch


    def decrease_lr(self, en_exp_decay, global_step, n_epoch, len_X, batch_size):
        if en_exp_decay:
            max_lr = 0.005
            min_lr = 0.0005
            decay_rate = math.log(min_lr/max_lr) / (-n_epoch*len_X/batch_size)
            lr = max_lr*math.exp(-decay_rate*global_step)
        else:
            lr = 0.001
        return lr
    # end method adjust_lr


    def list_avg(self, l):
        return sum(l) / len(l)
    # end method list_avg
# end class