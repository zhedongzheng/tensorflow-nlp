import tensorflow as tf
import numpy as np


class RNNTextGen:
    def __init__(self, text, seq_len, embedding_dims=30, rnn_size=256, n_layers=2, grad_clip=5.,
                 beam_width=3, sess=tf.Session()):
        self.sess = sess
        self.text = text
        self.seq_len = seq_len
        self.embedding_dims = embedding_dims
        self.rnn_size = rnn_size
        self.n_layers = n_layers
        self.grad_clip = grad_clip
        self.beam_width = beam_width
        self.preprocessing()
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.add_input_layer()       
        self.add_decoder()
        self.add_backward_path()
    # end method


    def add_input_layer(self):
        self.sequence = tf.placeholder(tf.int32, [None, None])
        self.sequence_length = tf.placeholder(tf.int32, [None])
        self._batch_size = tf.shape(self.sequence)[0]
        self.gen_seq_length = tf.placeholder(tf.int32, [])
    # end method


    def add_decoder(self):
        with tf.variable_scope('decode'):
            decoder_embedding = tf.get_variable('decoder_embedding', [self.vocab_size, self.embedding_dims],
                                                 tf.float32, tf.random_uniform_initializer(-1.0, 1.0))
            decoder_cell = tf.nn.rnn_cell.MultiRNNCell([self._rnn_cell() for _ in range(self.n_layers)])

            helper = tf.contrib.seq2seq.TrainingHelper(
                inputs = tf.nn.embedding_lookup(decoder_embedding, self._decoder_input()),
                sequence_length = self.sequence_length+1)
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell = decoder_cell,
                helper = helper,
                initial_state = decoder_cell.zero_state(self._batch_size, tf.float32),
                output_layer = tf.layers.Dense(self.vocab_size))
            decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = decoder)
            self.logits = decoder_output.rnn_output
        
        with tf.variable_scope('decode', reuse=True):
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell = tf.nn.rnn_cell.MultiRNNCell(
                    [self._rnn_cell(reuse=True) for _ in range(self.n_layers)]),
                embedding = tf.get_variable('decoder_embedding'),
                start_tokens = tf.tile(tf.constant(
                    [self.char2idx['<start>']], dtype=tf.int32), [self._batch_size]),
                end_token = self.char2idx['<end>'],
                initial_state = tf.contrib.seq2seq.tile_batch(
                    decoder_cell.zero_state(self._batch_size,tf.float32), self.beam_width),
                beam_width = self.beam_width,
                output_layer = tf.layers.Dense(self.vocab_size, _reuse=True))
            decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = decoder,
                maximum_iterations = self.gen_seq_length)
            self.predicted_ids = decoder_output.predicted_ids[:, :, 0]
    # end method


    def add_backward_path(self):
        targets = self._decoder_output()
        self.loss = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(
            logits = self.logits,
            targets = targets,
            weights = tf.to_float(tf.ones_like(targets)),
            average_across_timesteps = True,
            average_across_batch = True))
        # gradient clipping
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
        self.train_op = tf.train.AdamOptimizer().apply_gradients(zip(clipped_gradients, params))
    # end method


    def preprocessing(self):
        text = self.text 
        chars = set(text)
        self.char2idx = {c: i+2 for i, c in enumerate(chars)}
        self.char2idx['<start>'] = 0
        self.char2idx['<end>'] = 1
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


    def fit(self, text_iter_step=25, n_epoch=1, batch_size=128):
        n_batch = (len(self.indexed) - self.seq_len*batch_size - 1) // text_iter_step
        self.sess.run(tf.global_variables_initializer()) # initialize all variables
        
        for epoch in range(n_epoch):
            for local_step, seq_batch in enumerate(self.next_batch(batch_size, text_iter_step)):
                _, train_loss = self.sess.run([self.train_op, self.loss],
                                              {self.sequence: seq_batch,
                                               self.sequence_length: [self.seq_len]*len(seq_batch)})
                if local_step % 10 == 0:
                    print ('Epoch %d/%d | Batch %d/%d | train loss: %.4f' %
                          (epoch+1, n_epoch, local_step, n_batch, train_loss))
                if local_step % 100 == 0:
                    self.decode()
    # end method


    def decode(self):
        predicted_ids = self.sess.run(self.predicted_ids,
                                     {self._batch_size: 1,
                                      self.gen_seq_length: self.seq_len})[0]
        print('D: '+''.join([self.idx2char[idx] for idx in predicted_ids]), end='\n\n')
    # end method


    def _decoder_output(self):
        _end = tf.fill([self._batch_size, 1], self.char2idx['<end>'])
        return tf.concat([self.sequence, _end], 1)
    # end method


    def _decoder_input(self):
        _start = tf.fill([self._batch_size, 1], self.char2idx['<start>'])
        return tf.concat([_start, self.sequence], 1)
    # end method


    def _rnn_cell(self, reuse=False):
        return tf.nn.rnn_cell.GRUCell(self.rnn_size,
            kernel_initializer=tf.orthogonal_initializer(), reuse=reuse)
    # end method
# end class
