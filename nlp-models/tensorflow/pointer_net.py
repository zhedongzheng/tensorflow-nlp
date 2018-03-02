import tensorflow as tf
import numpy as np


class PointerNetwork:
    def __init__(self, max_len, rnn_size, X_word2idx, embedding_dim,
                 sess=tf.Session(), grad_clip=5.0):
        self.max_len = max_len
        self.rnn_size = rnn_size
        self.grad_clip = grad_clip
        self.X_word2idx = X_word2idx
        self.embedding_dim = embedding_dim
        self.sess = sess
        self.register_symbols()
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.add_input_layer()
        self.add_encoder_layer()
        self.add_decoder_layer()
        self.add_backward_path()
    # end method build_graph


    def add_input_layer(self):
        self.X = tf.placeholder(tf.int32, [None, self.max_len])
        self.Y = tf.placeholder(tf.int32, [None, self.max_len])
        self.X_seq_len = tf.placeholder(tf.int32, [None])
        self.Y_seq_len = tf.placeholder(tf.int32, [None])
        self.batch_size = tf.shape(self.X)[0]
    # end method add_input_layer


    def rnn_cell(self, reuse=False):
        return tf.nn.rnn_cell.GRUCell(self.rnn_size, kernel_initializer=tf.orthogonal_initializer(), reuse=reuse)
    # end method lstm_cell


    def add_encoder_layer(self):
        with tf.variable_scope('encoder'):
            embedding = tf.get_variable('embedding', [len(self.X_word2idx), self.embedding_dim], tf.float32)
            self.enc_inp = tf.nn.embedding_lookup(embedding, self.X)
        self.enc_rnn_out, self.enc_state = tf.nn.dynamic_rnn(
            cell = self.rnn_cell(), 
            inputs = self.enc_inp,
            sequence_length = self.X_seq_len,
            dtype = tf.float32)
    # end method add_encoder_layer


    def add_decoder_layer(self):
        def loop_fn(state, reuse=None):
            query = tf.expand_dims(state, -1)
            keys = self.enc_rnn_out
            align = tf.squeeze(tf.matmul(keys, query), -1)
            return align

        def rnn_decoder(initial_state, cell, embedding):
            state = initial_state
            outputs = []
            starts = tf.fill([self.batch_size], self._x_go)
            inp = tf.nn.embedding_lookup(embedding, starts)
            for i in range(self.max_len):
                _, state = cell(inp, state)
                output = loop_fn(state, reuse=True) if i > 0 else loop_fn(state)
                outputs.append(output)
                idx = tf.argmax(output, -1)
                inp = point(idx)
            return outputs

        def point(idx):
            idx = tf.expand_dims(idx, 1)
            b = tf.range(self.batch_size)
            b = tf.expand_dims(b, 1)
            c = tf.concat((tf.to_int64(b), idx), 1)
            g = tf.gather_nd(self.enc_inp, c)
            return g

        with tf.variable_scope('encoder', reuse=True):
            embedding = tf.get_variable('embedding')
        outputs = rnn_decoder(self.enc_state, self.rnn_cell(), embedding)
        outputs = tf.stack(outputs, 1)

        self.training_logits = outputs
        self.predicting_ids = tf.argmax(outputs, -1)
    # end method add_decoder_layer


    def add_backward_path(self):
        masks = tf.sequence_mask(self.Y_seq_len, self.max_len, dtype=tf.float32)
        self.loss = tf.contrib.seq2seq.sequence_loss(logits = self.training_logits,
                                                     targets = self.Y,
                                                     weights = masks)

        # gradient clipping
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
        self.train_op = tf.train.AdamOptimizer().apply_gradients(zip(clipped_gradients, params))
    # end method add_backward_path


    def fit(self, X_train, X_train_len, Y_train, Y_train_len, val_data, n_epoch=50, display_step=50, batch_size=128):
        X_test_batch, X_test_batch_lens, Y_test_batch, Y_test_batch_lens = val_data
        self.sess.run(tf.global_variables_initializer())

        for epoch in range(1, n_epoch+1):
            for local_step, (X_train_batch, X_train_batch_lens, Y_train_batch, Y_train_batch_lens) in enumerate(
                zip(self.gen_batch(X_train, batch_size), self.gen_batch(X_train_len, batch_size),
                    self.gen_batch(Y_train, batch_size), self.gen_batch(Y_train_len, batch_size))):
                _, loss = self.sess.run([self.train_op, self.loss], {self.X: X_train_batch,
                                                                     self.Y: Y_train_batch,
                                                                     self.X_seq_len: X_train_batch_lens,
                                                                     self.Y_seq_len: Y_train_batch_lens})
                if local_step % display_step == 0:
                    val_loss = self.sess.run(self.loss, {self.X: X_test_batch,
                                                         self.Y: Y_test_batch,
                                                         self.X_seq_len: X_test_batch_lens,
                                                         self.Y_seq_len: Y_test_batch_lens})
                    print("Epoch %d/%d | Batch %d/%d | train_loss: %.3f | test_loss: %.3f"
                        % (epoch, n_epoch, local_step, len(X_train)//batch_size, loss, val_loss))
    # end method fit


    def infer(self, input_word, X_idx2word, batch_size=128):        
        source = [self.X_word2idx.get(char, self._x_unk) for char in input_word] + [self._x_eos]
        seq_len = len(source)
        _input = source + [self._x_pad]*(self.max_len-seq_len)
        out_indices = self.sess.run(self.predicting_ids, {
            self.X: [_input],
            self.X_seq_len: [seq_len]})[0]
        
        print('\nSource')
        #print('Word: {}'.format([i for i in source]))
        print('IN: {}'.format(' '.join([X_idx2word[i] for i in source])))
        
        print('\nTarget')
        #print('Word: {}'.format([i for i in out_indices]))
        idx = out_indices[:len(source)]
        output = np.array(source)[idx]
        print('OUT: {}'.format(' '.join([X_idx2word[i] for i in output])))
    # end method infer


    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i+batch_size]
    # end method gen_batch


    def register_symbols(self):
        self._x_go = self.X_word2idx['<GO>']
        self._x_eos = self.X_word2idx['<EOS>']
        self._x_pad = self.X_word2idx['<PAD>']
        self._x_unk = self.X_word2idx['<UNK>']
    # end method add_symbols
# end class