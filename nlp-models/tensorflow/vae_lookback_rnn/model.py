from __future__ import print_function
from config import args
from modified_tf_classes import BasicDecoder, BeamSearchDecoder
from rnn_cell_impl import AttnGRUCell

import tensorflow as tf
import numpy as np
import sys


class VRAE:
    def __init__(self, params):
        self.params = params
        self._build_graph()


    def _build_graph(self):
        self._build_forward_graph()
        self._build_backward_graph()


    def _build_forward_graph(self):
        self._build_inputs()
        self._decode(self._reparam(*self._encode()))


    def _build_backward_graph(self):
        self.global_step = tf.Variable(0, trainable=False)

        self.nll_loss = self._nll_loss_fn()
        self.kl_w = self._kl_w_fn(args.anneal_max, args.anneal_bias, self.global_step)
        self.kl_loss = self._kl_loss_fn(self.z_mean, self.z_logvar)
        
        loss_op = self.nll_loss + self.kl_w * self.kl_loss
        
        clipped_gradients, params = self._gradient_clipping(loss_op)
        self.train_op = tf.train.AdamOptimizer().apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step)


    def _build_inputs(self):
        # placeholders
        self.enc_inp = tf.placeholder(tf.int32, [None, args.max_len])
        self.dec_inp = tf.placeholder(tf.int32, [None, args.max_len+1])
        self.dec_out = tf.placeholder(tf.int32, [None, args.max_len+1])
        # global helpers
        self._batch_size = tf.shape(self.enc_inp)[0]
        self.enc_seq_len = tf.count_nonzero(self.enc_inp, 1, dtype=tf.int32)
        self.dec_seq_len = self.enc_seq_len + 1
        
    
    def _encode(self):
        # the embedding is shared between encoder and decoder
        # since the source and the target for an autoencoder are the same
        with tf.variable_scope('encoder'):
            tied_embedding = tf.get_variable('tied_embedding',
                [self.params['vocab_size'], args.embedding_dim],
                tf.float32)
            _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = self._rnn_cell(args.rnn_size//2),
                cell_bw = self._rnn_cell(args.rnn_size//2), 
                inputs = tf.nn.embedding_lookup(tied_embedding, self.enc_inp),
                sequence_length = self.enc_seq_len,
                dtype = tf.float32)
        encoded_state = tf.concat((state_fw, state_bw), -1)
        self.z_mean = tf.layers.dense(encoded_state, args.latent_size)
        self.z_logvar = tf.layers.dense(encoded_state, args.latent_size)
        return self.z_mean, self.z_logvar


    def _reparam(self, z_mean, z_logvar):
        self.gaussian_noise = tf.truncated_normal(tf.shape(z_logvar))
        self.z = z_mean + tf.exp(0.5 * z_logvar) * self.gaussian_noise
        return self.z


    def _decode(self, z):
        self.training_rnn_out, self.training_logits = self._decoder_training(z)
        self.predicted_ids = self._decoder_inference(z)


    def _decoder_training(self, z):
        with tf.variable_scope('encoder', reuse=True):
            tied_embedding = tf.get_variable('tied_embedding', [self.params['vocab_size'], args.embedding_dim])

        with tf.variable_scope('decoding'):
            lookback_cell = self._lookback_rnn_cell()
            lookback_state = lookback_cell.zero_state(self._batch_size, tf.float32)
            lookback_state = (tf.layers.dense(self.z, args.rnn_size, tf.nn.elu),
                lookback_state[1])

            lin_proj = tf.layers.Dense(self.params['vocab_size'], _scope='decoder/dense')
            
            helper = tf.contrib.seq2seq.TrainingHelper(
                inputs = tf.nn.embedding_lookup(tied_embedding, self.dec_inp),
                sequence_length = self.dec_seq_len)
            decoder = BasicDecoder(
                cell = lookback_cell,
                helper = helper,
                initial_state = lookback_state,
                concat_z = self.z)
            decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = decoder)
        
        return decoder_output.rnn_output, lin_proj.apply(decoder_output.rnn_output)


    def _decoder_inference(self, z):
        tiled_z = tf.tile(tf.expand_dims(self.z, 1), [1, args.beam_width, 1])
        
        with tf.variable_scope('encoder', reuse=True):
            tied_embedding = tf.get_variable('tied_embedding', [self.params['vocab_size'], args.embedding_dim])

        with tf.variable_scope('decoding', reuse=True):
            lookback_cell = self._lookback_rnn_cell(reuse=True)
            lookback_state = lookback_cell.zero_state(self._batch_size, tf.float32)
            lookback_state = (tf.layers.dense(self.z, args.rnn_size, tf.nn.elu, reuse=True),
                lookback_state[1])

            decoder = BeamSearchDecoder(
                cell = lookback_cell,
                embedding = tied_embedding,
                start_tokens = tf.tile(tf.constant(
                    [self.params['word2idx']['<start>']], dtype=tf.int32), [self._batch_size]),
                end_token = self.params['word2idx']['<end>'],
                initial_state = tf.contrib.seq2seq.tile_batch(lookback_state, args.beam_width),
                beam_width = args.beam_width,
                output_layer = tf.layers.Dense(self.params['vocab_size'], _reuse=True),
                concat_z = tiled_z)
            decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = decoder,
                maximum_iterations = 2*tf.reduce_max(self.enc_seq_len))

        return decoder_output.predicted_ids[:, :, 0]


    def _rnn_cell(self, rnn_size=None, reuse=False):
        rnn_size = args.rnn_size if rnn_size is None else rnn_size
        return tf.nn.rnn_cell.GRUCell(
            rnn_size, kernel_initializer=tf.orthogonal_initializer(), reuse=reuse)

    
    def _lookback_rnn_cell(self, rnn_size=None, reuse=False):
        rnn_size = args.rnn_size if rnn_size is None else rnn_size
        return AttnGRUCell(
            rnn_size, args.max_len, kernel_initializer=tf.orthogonal_initializer(),
            reuse=reuse)


    def _nll_loss_fn(self):
        mask_fn = lambda l : tf.sequence_mask(l, tf.reduce_max(l), dtype=tf.float32)
        mask = mask_fn(self.dec_seq_len)
        if (args.num_sampled <= 0) or (args.num_sampled >= self.params['vocab_size']):
            return tf.reduce_sum(tf.contrib.seq2seq.sequence_loss(
                logits = self.training_logits,
                targets = self.dec_out,
                weights = mask,
                average_across_timesteps = False,
                average_across_batch = True))
        else:
            with tf.variable_scope('decoding/decoder/dense', reuse=True):
                mask = tf.reshape(mask, [-1])
                return tf.reduce_sum(mask * tf.nn.sampled_softmax_loss(
                    weights = tf.transpose(tf.get_variable('kernel')),
                    biases = tf.get_variable('bias'),
                    labels = tf.reshape(self.dec_out, [-1, 1]),
                    inputs = tf.reshape(self.training_rnn_out, [-1, args.rnn_size]),
                    num_sampled = args.num_sampled,
                    num_classes = self.params['vocab_size'],
                )) / tf.to_float(self._batch_size)


    def _kl_w_fn(self, anneal_max, anneal_bias, global_step):
        return anneal_max * tf.sigmoid((10 / anneal_bias) * (
            tf.cast(global_step, tf.float32) - tf.constant(anneal_bias / 2)))


    def _kl_loss_fn(self, mean, gamma):
        return 0.5 * tf.reduce_sum(
            tf.exp(gamma) + tf.square(mean) - 1 - gamma) / tf.to_float(self._batch_size)


    def train_session(self, sess, enc_inp, dec_inp, dec_out):
        _, nll_loss, kl_w, kl_loss, step = sess.run(
            [self.train_op, self.nll_loss, self.kl_w, self.kl_loss, self.global_step],
                {self.enc_inp: enc_inp, self.dec_inp: dec_inp, self.dec_out: dec_out})
        return {'nll_loss': nll_loss,
                'kl_w': kl_w,
                'kl_loss': kl_loss,
                'step': step}


    def reconstruct(self, sess, sentence, sentence_dropped):
        idx2word = self.params['idx2word']
        print('I: %s' % ' '.join([idx2word[idx] for idx in sentence]))
        print()
        print('D: %s' % ' '.join([idx2word[idx] for idx in sentence_dropped]))
        print()
        predicted_ids = sess.run(self.predicted_ids, {self.enc_inp: np.atleast_2d(sentence)})[0]
        print('O: %s' % ' '.join([idx2word[idx] for idx in predicted_ids]))
        print('-'*12)
    

    def customized_reconstruct(self, sess, sentence):
        idx2word = self.params['idx2word']
        sentence = [self.get_new_w(w) for w in sentence.split()][:args.max_len]
        print('I: %s' % ' '.join([idx2word[idx] for idx in sentence]))
        print()
        sentence = sentence + [self.params['word2idx']['<pad>']] * (args.max_len-len(sentence))
        predicted_ids = sess.run(self.predicted_ids, {self.enc_inp: np.atleast_2d(sentence)})[0]
        print('O: %s' % ' '.join([idx2word[idx] for idx in predicted_ids]))
        print('-'*12)


    def get_new_w(self, w):
        idx = self.params['word2idx'][w]
        return idx if idx < self.params['vocab_size'] else self.params['word2idx']['<unk>']


    def generate(self, sess):
        predicted_ids = sess.run(self.predicted_ids,
                                {self._batch_size: 1,
                                 self.z: np.random.randn(1, args.latent_size),
                                 self.enc_seq_len: [args.max_len]})[0]
        print('G: %s' % ' '.join([self.params['idx2word'][idx] for idx in predicted_ids]))
        print('-'*12)


    def _gradient_clipping(self, loss_op):
        params = tf.trainable_variables()
        gradients = tf.gradients(loss_op, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, args.clip_norm)
        return clipped_gradients, params
