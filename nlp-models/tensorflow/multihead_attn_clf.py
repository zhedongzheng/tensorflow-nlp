import tensorflow as tf
import numpy as np
import math
from sklearn.utils import shuffle
from utils import *


class Tagger:
    def __init__(self, vocab_size, n_out, seq_len,
                 dropout_rate=0.1, hidden_units=128, num_heads=8, sess=tf.Session()):
        self.vocab_size = vocab_size
        self.n_out = n_out
        self.seq_len = seq_len
        self.dropout_rate = dropout_rate
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.sess = sess
        self._pointer = None
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.add_input_layer()
        self.add_forward_path()
        self.add_crf_layer()
        self.add_backward_path()
    # end method build_graph


    def add_input_layer(self):
        self.X = tf.placeholder(tf.int32, [None, self.seq_len])
        self.Y = tf.placeholder(tf.int32, [None, self.seq_len])
        self.X_seq_len = tf.placeholder(tf.int32, [None])
        self.is_training = tf.placeholder(tf.bool)
        self.lr = tf.placeholder(tf.float32)
        self._pointer = self.X
    # end method add_input_layer


    def add_forward_path(self):
        with tf.variable_scope('word_embedding'):
            encoded = embed_seq(
                self.X, self.vocab_size, self.hidden_units, zero_pad=False, scale=True)

        with tf.variable_scope('dropout'):
            encoded = tf.layers.dropout(
                encoded, self.dropout_rate, training=self.is_training)

        win_size = 3
        with tf.variable_scope('attn_masked_window_%d'%(win_size)):
            masks = self.window_mask(win_size)
            encoded = multihead_attn(encoded,
                num_units=self.hidden_units, num_heads=self.num_heads, seq_len=self.seq_len, masks=masks)

        win_size = 2
        with tf.variable_scope('attn_masked_window_%d'%(win_size)):
            masks = self.window_mask(win_size)
            encoded = multihead_attn(encoded,
                num_units=self.hidden_units, num_heads=self.num_heads, seq_len=self.seq_len, masks=masks)

        with tf.variable_scope('position_embedding'):
            encoded += learned_positional_encoding(
                self.X, self.hidden_units, zero_pad=False, scale=False)

        win_size = 10

        with tf.variable_scope('attn_masked_fw'):
            masks = np.zeros([self.seq_len, self.seq_len])
            for i in range(self.seq_len):
                if i < win_size:
                    masks[i, :i+1] = 1.
                else:                                                             
                    masks[i, i-win_size:i+1] = 1.
            masks = tf.convert_to_tensor(masks)
            masks = tf.tile(tf.expand_dims(masks,0), [tf.shape(self.X)[0]*self.num_heads, 1, 1])
            encoded = multihead_attn(encoded,
                num_units=self.hidden_units, num_heads=self.num_heads, seq_len=self.seq_len, masks=masks)

        with tf.variable_scope('attn_masked_bw'):
            masks = np.zeros([self.seq_len, self.seq_len])
            for i in range(self.seq_len):
                if i > self.seq_len - win_size - 1:
                    masks[i, i:] = 1.
                else:                                                             
                    masks[i, i:i+win_size+1] = 1.
            masks = tf.convert_to_tensor(masks)
            masks = tf.tile(tf.expand_dims(masks,0), [tf.shape(self.X)[0]*self.num_heads, 1, 1])
            encoded = multihead_attn(encoded,
                num_units=self.hidden_units, num_heads=self.num_heads, seq_len=self.seq_len, masks=masks)

        with tf.variable_scope('pointwise'):
            encoded = pointwise_feedforward(encoded,
                num_units=[4*self.hidden_units, self.hidden_units], activation=tf.nn.relu)

        with tf.variable_scope('output_layer'):
            self.logits = tf.layers.dense(encoded, self.n_out)
    # end method add_forward_path


    def add_crf_layer(self):
        with tf.variable_scope('crf_loss'):
            self.log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
                inputs = self.logits,
                tag_indices = self.Y,
                sequence_lengths = self.X_seq_len)
        with tf.variable_scope('crf_loss', reuse=True):
            transition_params = tf.get_variable('transitions', [self.n_out, self.n_out])
        self.viterbi_sequence, _ = tf.contrib.crf.crf_decode(
            self.logits, transition_params, self.X_seq_len)
    # end method add_crf_layer


    def add_backward_path(self):
        self.loss = tf.reduce_mean(-self.log_likelihood)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.viterbi_sequence, self.Y), tf.float32))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
    # end method add_backward_path


    def fit(self, X, Y, val_data=None, n_epoch=10, batch_size=128, en_exp_decay=True, en_shuffle=True):
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
                                              self.is_training: True})
                global_step += 1
                if local_step % 50 == 0:
                    print ('Epoch %d/%d | Step %d/%d | train_loss: %.4f | train_acc: %.4f | lr: %.4f'
                           %(epoch+1, n_epoch, local_step, int(len(X)/batch_size), loss, acc, lr))
            # verbose
            print ("Epoch %d/%d | train_loss: %.4f | train_acc: %.4f |" % (epoch+1, n_epoch, loss, acc),
                   "lr: %.4f" % (lr) )
            if val_data is not None:
                X_test, Y_test = val_data
                y_pred = self.predict(X_test, batch_size=batch_size)
                final_acc = (y_pred == Y_test).astype(np.float32).mean()
                print("final testing accuracy: %.4f" % final_acc)
    # end method fit


    def predict(self, X_test, batch_size=128):
        batch_pred_list = []
        for X_test_batch in self.gen_batch(X_test, batch_size):
            batch_pred = self.sess.run(self.viterbi_sequence,
                                      {self.X: X_test_batch,
                                       self.X_seq_len: len(X_test_batch)*[X_test.shape[1]],
                                       self.is_training: False})
            batch_pred_list.append(batch_pred)
        return np.vstack(batch_pred_list)
    # end method predict


    def infer(self, xs, x_len):
        viterbi_seq = self.sess.run(self.viterbi_sequence,
                                   {self.X: np.atleast_2d(xs),
                                    self.X_seq_len: np.atleast_1d(x_len),
                                    self.is_training: False})
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


    def window_mask(self, h_w):
        masks = np.zeros([self.seq_len, self.seq_len])
        for i in range(self.seq_len):
            if i < h_w:
                masks[i, :i+h_w+1] = 1.
            elif i > self.seq_len - h_w - 1:
                masks[i, i-h_w:] = 1.
            else:                                                             
                masks[i, i-h_w:i+h_w+1] = 1.
        masks = tf.convert_to_tensor(masks)
        masks = tf.tile(tf.expand_dims(masks,0), [tf.shape(self.X)[0]*self.num_heads, 1, 1])
        return masks
# end class


def multihead_attn(inputs, num_units, num_heads, seq_len, masks):
    T_q = T_k = inputs.get_shape().as_list()[1]           

    Q_K_V = tf.layers.dense(inputs, 3*num_units, tf.nn.relu)
    Q, K, V = tf.split(Q_K_V, 3, -1)
    
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)                         # (h*N, T_q, C/h) 
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)                         # (h*N, T_k, C/h) 
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)                         # (h*N, T_k, C/h)
    
    align = tf.matmul(Q_, tf.transpose(K_, [0,2,1]))                               # (h*N, T_q, T_k)
    align = align / (K_.get_shape().as_list()[-1] ** 0.5)                          # scale
    
    paddings = tf.fill(tf.shape(align), float('-inf'))                             # exp(-large) -> 0
    align = tf.where(tf.equal(masks, 0), paddings, align)                          # (h*N, T_q, T_k)

    align = tf.nn.softmax(align)                                                   # (h*N, T_q, T_k)

    # Weighted sum
    outputs = tf.matmul(align, V_)                                                 # (h*N, T_q, C/h)
    
    # Restore shape
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)              # (N, T_q, C)
    
    # Residual connection
    outputs += inputs                                                              # (N, T_q, C)   
    # Normalize
    outputs = layer_norm(outputs)                                                  # (N, T_q, C)
    return outputs


