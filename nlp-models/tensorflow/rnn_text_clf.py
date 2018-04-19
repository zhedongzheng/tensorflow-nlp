from base_text_clf import BaseTextClassifier

import tensorflow as tf
import sklearn
import numpy as np


class RNNTextClassifier(BaseTextClassifier):
    def __init__(self, vocab_size, n_out, embedding_dims=128, cell_size=128, grad_clip=5.0,
                 sess=tf.Session()):
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
        """
        self.vocab_size = vocab_size
        self.embedding_dims = embedding_dims
        self.cell_size = cell_size
        self.grad_clip = grad_clip
        self.n_out = n_out
        self.sess = sess
        self._pointer = None
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.add_input_layer()
        self.add_word_embedding_layer()
        self.add_dynamic_rnn()
        self.add_output_layer()
        self.add_backward_path()
    # end method build_graph


    def add_input_layer(self):
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int64, [None])
        self.X_seq_lens = tf.placeholder(tf.int32, [None])
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
        return tf.nn.rnn_cell.GRUCell(self.cell_size//2, kernel_initializer=tf.orthogonal_initializer())
    # end method lstm_cell


    def add_dynamic_rnn(self):       
        _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
            self.rnn_cell(), self.rnn_cell(), self._pointer,
            sequence_length=self.X_seq_lens, dtype=tf.float32)
        self._pointer = tf.concat([state_fw, state_bw], -1)
    # end method add_dynamic_rnn


    def add_output_layer(self):
        self.logits = tf.layers.dense(self._pointer, self.n_out)
    # end method add_output_layer


    def add_backward_path(self):
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits,1), self.Y), tf.float32))
        # gradient clipping
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
        self.train_op = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(clipped_gradients, params))
    # end method add_backward_path


    def fit(self, X, Y, val_data=None, n_epoch=10, batch_size=128, en_exp_decay=True, en_shuffle=True, 
            keep_prob=1.0):
        if val_data is None:
            print("Train %d samples" % len(X) )
        else:
            print("Train %d samples | Test %d samples" % (len(X), len(val_data[0])))
        log = {'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}
        global_step = 0

        self.sess.run(tf.global_variables_initializer()) # initialize all variables
        for epoch in range(n_epoch): # batch training
            if en_shuffle:
                X, Y = sklearn.utils.shuffle(X, Y)

            for local_step, ((X_batch, X_batch_lens), Y_batch) in enumerate(zip(self.next_batch(X, batch_size),
                                                                                self.gen_batch(Y, batch_size))):
                lr = self.decrease_lr(en_exp_decay, global_step, n_epoch, len(X), batch_size)           
                _, loss, acc = self.sess.run([self.train_op, self.loss, self.acc],
                                             {self.X :X_batch, self.Y: Y_batch,
                                              self.X_seq_lens: X_batch_lens,
                                              self.lr: lr,
                                              self.keep_prob: keep_prob})
                global_step += 1
                if local_step % 50 == 0:
                    print ("Epoch %d/%d | Step %d/%d | train_loss: %.4f | train_acc: %.4f | lr: %.4f"
                           %(epoch+1, n_epoch, local_step, int(len(X)/batch_size), loss, acc, lr))

            if val_data is not None: # go through testing data, average validation loss and ac 
                val_loss_list, val_acc_list = [], []
                for (X_test_batch, X_test_batch_lens), Y_test_batch in zip(self.next_batch(val_data[0], batch_size),
                                                                           self.gen_batch(val_data[1], batch_size)):
                    v_loss, v_acc = self.sess.run([self.loss, self.acc],
                                                  {self.X: X_test_batch, self.Y: Y_test_batch,
                                                   self.X_seq_lens: X_test_batch_lens,
                                                   self.keep_prob: 1.0})
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
                       "test_loss: %.4f | test_acc: %.4f |" % (val_loss, val_acc), "lr: %.4f" % (lr) )
        # end "for epoch in range(n_epoch)"

        return log
    # end method fit


    def predict(self, X_test, batch_size=128):
        batch_pred_list = []
        for (X_test_batch, X_test_batch_lens) in self.next_batch(X_test, batch_size):
            batch_pred = self.sess.run(self.logits,
                                      {self.X: X_test_batch,
                                       self.X_seq_lens: X_test_batch_lens,
                                       self.keep_prob: 1.0})
            batch_pred_list.append(batch_pred)
        return np.argmax(np.vstack(batch_pred_list), 1)
    # end method predict


    def pad_sentence_batch(self, sentence_batch, pad_int=0):
        max_seq_len = max([len(sentence) for sentence in sentence_batch])
        padded_seqs = []
        seq_lens = []
        for sentence in sentence_batch:
            padded_seqs.append(sentence + [pad_int] * (max_seq_len - len(sentence)))
            seq_lens.append(len(sentence))
        return padded_seqs, seq_lens
    # end method pad_sentence_batch


    def next_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            padded_seqs, seq_lens = self.pad_sentence_batch(arr[i : i+batch_size])
            yield padded_seqs, seq_lens
    # end method gen_batch
# end class