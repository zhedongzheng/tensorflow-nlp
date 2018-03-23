import tensorflow as tf
import numpy as np
import math
import sklearn

from tqdm import tqdm
from utils import *


class OnlyAttentionClassifier:
    def __init__(self, seq_len, vocab_size, n_out, sess=tf.Session(),
                 embedding_dims=64, num_heads=8):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.embedding_dims = embedding_dims
        self.hidden_units = embedding_dims
        self.num_heads = num_heads
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
        embedding = tf.get_variable(
            'lookup_table', [self.vocab_size,self.embedding_dims], tf.float32)
        embedded = tf.nn.embedding_lookup(embedding, self._pointer)
        self._pointer = tf.nn.dropout(embedded, self.keep_prob)
    # end method add_word_embedding_layer


    def add_self_attention(self):
        x = self._pointer
        
        for i, win_size in enumerate([1, 2]):
            with tf.variable_scope('attn_masked_window%d'%win_size):
                x = self.multihead_attn(x, self.window_mask(win_size))
        
        x = self.global_pooling(x, tf.layers.max_pooling1d)

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


    def fit(self, X, Y, val_data=None, n_epoch=10, batch_size=128, keep_prob=1.0, en_exp_decay=True,
            en_shuffle=True):
        if val_data is None:
            print("Train %d samples" % len(X))
        else:
            print("Train %d samples | Test %d samples" % (len(X), len(val_data[0])))
        log = {'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}
        global_step = 0

        self.sess.run(tf.global_variables_initializer()) # initialize all variables
        for epoch in range(n_epoch):
            if en_shuffle:
                X, Y = sklearn.utils.shuffle(X, Y)
            
            for local_step, (X_batch, Y_batch) in enumerate(zip(self.next_batch(X, batch_size),
                                                                self.next_batch(Y, batch_size))):
                lr = self.decrease_lr(en_exp_decay, global_step, n_epoch, len(X), batch_size) 
                _, loss, acc = self.sess.run([self.train_op, self.loss, self.acc],
                                             {self.X:X_batch, self.Y:Y_batch,
                                              self.lr:lr, self.keep_prob:keep_prob})
                global_step += 1
                if local_step % 50 == 0:
                    print ("Epoch %d/%d | Step %d/%d | train_loss: %.4f | train_acc: %.4f | lr: %.4f"
                        %(epoch+1, n_epoch, local_step, int(len(X)/batch_size), loss, acc, lr))

            if val_data is not None: # go through test dara, compute averaged validation loss and acc
                val_loss_list, val_acc_list = [], []
                for X_test_batch, Y_test_batch in tqdm(
                    zip(self.next_batch(val_data[0], batch_size),
                        self.next_batch(val_data[1], batch_size)), total=len(val_data[0])//batch_size, ncols=70):
                    v_loss, v_acc = self.sess.run([self.loss, self.acc],
                                                  {self.X:X_test_batch, self.Y:Y_test_batch,
                                                   self.keep_prob:1.0})
                    val_loss_list.append(v_loss)
                    val_acc_list.append(v_acc)
                val_loss, val_acc = self.list_avg(val_loss_list), self.list_avg(val_acc_list)

            # append to log
            log['loss'].append(loss)
            log['acc'].append(acc)
            if val_data is not None:
                log['val_loss'].append(val_loss)
                log['val_acc'].append(val_acc)
            # verbose
            if val_data is None:
                print ("Epoch %d/%d | train_loss: %.4f | train_acc: %.4f |" % (epoch+1, n_epoch, loss, acc),
                    "lr: %.4f" % (lr) )
            else:
                print ("Epoch %d/%d | train_loss: %.4f | train_acc: %.4f |" % (epoch+1, n_epoch, loss, acc),
                    "test_loss: %.4f | test_acc: %.4f |" % (val_loss, val_acc),
                    "lr: %.4f" % (lr) )
        # end "for epoch in range(n_epoch):"

        return log
    # end method fit


    def predict(self, X_test, batch_size=128):
        batch_pred_list = []
        for X_test_batch in tqdm(
            self.next_batch(X_test, batch_size), total=len(X_test)//batch_size, ncols=70):
            batch_pred = self.sess.run(self.logits, {self.X:X_test_batch, self.keep_prob:1.0})
            batch_pred_list.append(batch_pred)
        return np.argmax(np.vstack(batch_pred_list), 1)
    # end method predict


    def next_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i+batch_size]
    # end method gen_batch


    def decrease_lr(self, en_exp_decay, global_step, n_epoch, len_X, batch_size):
        if en_exp_decay:
            max_lr = 0.005
            min_lr = 0.001
            decay_rate = math.log(min_lr/max_lr) / (-n_epoch*len_X/batch_size)
            lr = max_lr*math.exp(-decay_rate*global_step)
        else:
            lr = 0.005
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


    def multihead_attn(self, inputs, masks):
        num_units = self.hidden_units
        num_heads = self.num_heads
        seq_len = self.seq_len
        T_q = T_k = inputs.get_shape().as_list()[1]           

        Q_K_V = tf.layers.dense(inputs, 3*num_units, tf.nn.relu)
        Q, K, V = tf.split(Q_K_V, 3, -1)
        
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)                         # (h*N, T_q, C/h) 
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)                         # (h*N, T_k, C/h) 
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)                         # (h*N, T_k, C/h)
        
        align = tf.matmul(Q_, tf.transpose(K_, [0,2,1]))                               # (h*N, T_q, T_k)
        align = align / np.sqrt(K_.get_shape().as_list()[-1])                          # scale
        
        paddings = tf.fill(tf.shape(align), float('-inf'))                             # exp(-large) -> 0
        if masks is not None:
            align = tf.where(tf.equal(masks, 0), paddings, align)                      # (h*N, T_q, T_k)

        align = tf.nn.softmax(align)                                                   # (h*N, T_q, T_k)

        # Weighted sum
        outputs = tf.matmul(align, V_)                                                 # (h*N, T_q, C/h)
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)              # (N, T_q, C)
        # Masking
        outputs *= tf.expand_dims(tf.to_float(tf.sign(self.X)), -1)
        # Residual connection
        outputs += inputs                                                              # (N, T_q, C)   
        # Normalize
        outputs = layer_norm(outputs)                                                  # (N, T_q, C)
        return outputs


    def global_pooling(self, x, fn):
        batch_size = tf.shape(self.X)[0]
        num_units = x.get_shape().as_list()[-1]
        x = fn(x, x.get_shape().as_list()[1], 1)
        x = tf.reshape(x, [batch_size, num_units])
        return x


# end class