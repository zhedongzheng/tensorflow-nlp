import numpy as np
import tensorflow as tf
import sklearn
import math

from scipy.sparse import isspmatrix_csr


class LogisticRegression:
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

        self.logits = tf.layers.dense(self.X, self.n_out)

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.acc = tf.reduce_mean(tf.cast(tf.equal(
            tf.argmax(self.logits, 1), self.Y), tf.float32))
    # end method build_graph


    def fit(self, X, Y, val_data=None, n_epoch=10, batch_size=128,
            en_exp_decay=False, en_shuffle=True):
        if val_data is None:
            print("Train %s" % X.shape)
        else:
            print("Train %s | Test %s" % (X.shape, val_data[0].shape))

        log = {'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}
        global_step = 0

        self.sess.run(tf.global_variables_initializer())
        for epoch in range(n_epoch):
            if en_shuffle:
                X, Y = sklearn.utils.shuffle(X, Y)
            local_step = 1
            
            for X_batch, Y_batch in zip(self.gen_batch(X, batch_size),
                                        self.gen_batch(Y, batch_size)):
                lr = self.decrease_lr(en_exp_decay, global_step, n_epoch, X.shape[0], batch_size) 
                _, loss, acc = self.sess.run([self.train_op, self.loss, self.acc],
                    {self.X:X_batch, self.Y:Y_batch, self.lr:lr})
                local_step += 1
                global_step += 1
                if local_step % 50 == 0:
                    print ("Epoch %d/%d | Step %d/%d | train_loss: %.4f | train_acc: %.4f | lr: %.4f"
                        %(epoch+1, n_epoch, local_step, X.shape[0]//batch_size, loss, acc, lr))

            if val_data is not None:
                val_loss_list, val_acc_list = [], []
                for X_test_batch, Y_test_batch in zip(self.gen_batch(val_data[0], batch_size),
                                                      self.gen_batch(val_data[1], batch_size)):
                    v_loss, v_acc = self.sess.run([self.loss, self.acc],
                        {self.X:X_test_batch, self.Y:Y_test_batch})
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

        return log
    # end method fit


    def predict(self, X_test, batch_size=128):
        batch_pred_list = []
        for X_test_batch in self.gen_batch(X_test, batch_size):
            batch_pred = self.sess.run(self.logits,
                {self.X:X_test_batch})
            batch_pred_list.append(batch_pred)
        return np.argmax(np.vstack(batch_pred_list), 1)
    # end method predict


    def gen_batch(self, arr, batch_size):
        for i in range(0, arr.shape[0], batch_size):
            if isspmatrix_csr(arr):
                yield arr[i : i+batch_size].toarray()
            else:
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
# end class