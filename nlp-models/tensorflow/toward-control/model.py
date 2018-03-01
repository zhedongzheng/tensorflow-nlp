from __future__ import print_function
from config import args
from utils import gumbel_softmax_sample, inverse_sigmoid
from modified_tf_classes import BasicDecoder, BeamSearchDecoder

import numpy as np
import tensorflow as tf


class Model:
    def __init__(self, params):
        self.params = params
        self.scopes = {
            'E': 'Encoder',
            'D': 'Discriminator',
            'G': 'Generator'}
        self.build_placeholders()
        self.build_global_helpers()

         # initialize the base VAE by equation 4
        self.build_train_vae_graph()              
        print("[1/5] VAE Graph Built")

         # train the discriminator D by eqation 11
        self.build_train_discriminator_graph()    
        print("[2/5] Discriminator Graph Built")

        # train the generator by eqation 8 and encoder by equation 4
        self.build_train_generator_encoder_graph()
        print("[3/5] Generator and Encoder Graphs Built")

        self.build_prior_inference_graph()
        print("[4/5] Prior Inference Graph Built")
        
        self.build_posterior_inference_graph()
        print("[5/5] Posterior Inference Graph Built")

        self.saver = tf.train.Saver()
        self.model_path = './saved/model.ckpt'


    def build_train_vae_graph(self):
        z_mean, z_logvar = self.encoder(self.enc_inp)
        z = self.reparam(z_mean, z_logvar)
        latent_vec = tf.concat((z, self.draw_c_prior()), -1)
        outputs = self.generator(latent_vec)

        self.train_vae_nll_loss = self.seq_loss_fn(*outputs)
        self.train_vae_kl_w = self.kl_w_fn()
        self.train_vae_kl_loss = self.kl_loss_fn(z_mean, z_logvar)
        loss_op = self.train_vae_nll_loss + self.train_vae_kl_w * self.train_vae_kl_loss
        self.train_vae_op = self.optimizer.apply_gradients(
            self.gradient_clipped(loss_op), global_step=self.global_step)


    def build_train_discriminator_graph(self):
        logits_real = self.discriminator(self.enc_inp, is_training=True)
        self.train_clf_loss = self.sparse_cross_entropy_fn(logits_real, self.labels)
        self.train_clf_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(logits_real, 1), self.labels)))

        c_prior = self.draw_c_prior()
        latent_vec = tf.concat((self.draw_z_prior(), c_prior), -1)
        _, logits_gen = self.generator(latent_vec, reuse=True)
        ids_gen = tf.argmax(logits_gen[:, :-1, :], -1)
        logits_fake = self.discriminator(ids_gen, reuse=True, refeed=True)
        entropy = - tf.reduce_sum(tf.log(tf.nn.softmax(logits_fake)))
        self.L_u = self.cross_entropy_fn(logits_fake, c_prior) + args.beta * entropy

        loss_op = self.train_clf_loss + args.lambda_u * self.L_u
        self.train_clf_op = self.optimizer.minimize(
            loss_op, var_list=tf.trainable_variables(self.scopes['D']), global_step=self.global_step)


    def build_train_generator_encoder_graph(self):
        z_mean, z_logvar = self.encoder(self.enc_inp, reuse=True)
        z = self.reparam(z_mean, z_logvar)
        c = self.draw_c(self.discriminator(self.enc_inp, reuse=True))
        latent_vec = tf.concat((z, c), -1)
        outputs = self.generator(latent_vec, reuse=True)

        self.train_ge_vae_nll_loss = self.seq_loss_fn(*outputs)
        self.train_ge_vae_kl_w = self.kl_w_fn()
        self.train_ge_vae_kl_loss = self.kl_loss_fn(z_mean, z_logvar)
        vae_loss = self.train_ge_vae_nll_loss + self.train_ge_vae_kl_w * self.train_ge_vae_kl_loss

        z_prior = self.draw_z_prior()
        c_prior = self.draw_c_prior()
        latent_vec = tf.concat((z_prior, c_prior), -1)
        _, logits_gen = self.generator(latent_vec, reuse=True)
        self.temperature = self.temperature_fn()
        gumbel_softmax = gumbel_softmax_sample(logits_gen[:, :-1, :], self.temperature)

        c_logits = self.discriminator(gumbel_softmax, reuse=True, refeed=True, gumbel=True)
        self.l_attr_c = self.cross_entropy_fn(c_logits, c_prior)

        z_mean_gen, z_logvar_gen = self.encoder(gumbel_softmax, reuse=True, gumbel=True)
        self.l_attr_z = self.mutinfo_loss_fn(z_mean_gen, z_logvar_gen)

        generator_loss_op = vae_loss + (args.lambda_c*self.l_attr_c) + (args.lambda_z*self.l_attr_z)
        encoder_loss_op = vae_loss

        self.train_generator_op = self.optimizer.apply_gradients(
            self.gradient_clipped(generator_loss_op, scope=self.scopes['G']))
        self.train_encoder_op = self.optimizer.apply_gradients(
            self.gradient_clipped(encoder_loss_op, scope=self.scopes['E']))


    def build_prior_inference_graph(self):
        latent_vec = tf.concat((self.draw_z_prior(), self.draw_c_prior()), -1)
        self.prior_gen_ids = self.generator(latent_vec, inference=True)


    def build_posterior_inference_graph(self):
        z_mean, z_logvar = self.encoder(self.enc_inp, reuse=True)
        z = self.reparam(z_mean, z_logvar)
        c_logits = self.discriminator(self.enc_inp, reuse=True)
        latent_vec = tf.concat((z, self.draw_c(c_logits)), -1)
        self.post_gen_ids = self.generator(latent_vec, inference=True)
        
        c_reversed = self.draw_c(tf.log(1 - tf.nn.softmax(c_logits)))
        latent_vec = tf.concat((z, c_reversed), -1)
        self.post_reverse_gen_ids = self.generator(latent_vec, inference=True)

    
    def encoder(self, inputs, reuse=None, gumbel=False):
        with tf.variable_scope(self.scopes['E'], reuse=reuse):
            embedding = tf.get_variable(
                'embedding', [self.params['vocab_size'], args.embedding_dim], tf.float32)

            if not gumbel:
                x = tf.nn.embedding_lookup(embedding, inputs)
            else:
                x = tf.matmul(tf.reshape(inputs, [-1,self.params['vocab_size']]), embedding)
                x = tf.reshape(x, [self.batch_size, args.max_len, args.embedding_dim])
            
            _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = self.rnn_cell(args.rnn_size//2, reuse=reuse),
                cell_bw = self.rnn_cell(args.rnn_size//2, reuse=reuse), 
                inputs = x,
                sequence_length = self.enc_seq_len,
                dtype = tf.float32)
            birnn_state = tf.concat((state_fw, state_bw), -1)

            z_mean = tf.layers.dense(birnn_state, args.latent_size, reuse=reuse)
            z_logvar = tf.layers.dense(birnn_state, args.latent_size, reuse=reuse)
            return z_mean, z_logvar

    
    def discriminator(self, inputs, reuse=None, refeed=False, gumbel=False, is_training=False):
        with tf.variable_scope(self.scopes['D'], reuse=reuse):
            embedding = tf.get_variable(
                'embedding', [self.params['vocab_size'], args.embedding_dim], tf.float32)
            
            if not gumbel:
                x = tf.nn.embedding_lookup(embedding, inputs)
            else:
                x = tf.matmul(tf.reshape(inputs, [-1,self.params['vocab_size']]), embedding)
                x = tf.reshape(x, [self.batch_size, args.max_len, args.embedding_dim])
            x = tf.layers.dropout(x, args.cnn_dropout_rate, training=is_training)
            
            multi_kernels = []
            for i, k in enumerate([3, 4, 5]):
                _x = tf.layers.conv1d(x, args.cnn_filters, k,
                    activation=tf.nn.elu, reuse=reuse, name='conv%d'%i)
                seq_len = _x.get_shape().as_list()[1] if not refeed else (args.max_len - k + 1)
                _x = tf.layers.max_pooling1d(_x, seq_len, 1)
                _x = tf.reshape(_x, [self.batch_size, args.cnn_filters])
                multi_kernels.append(_x)
            x = tf.concat(multi_kernels, -1)
            
            logits = tf.layers.dense(x, args.num_class, reuse=reuse)
            return logits


    def generator(self, latent_vec, reuse=None, inference=False):
        with tf.variable_scope(self.scopes['E'], reuse=True):
            embedding = tf.get_variable(
                'embedding', [self.params['vocab_size'], args.embedding_dim])

        if not inference:
            with tf.variable_scope(self.scopes['G'], reuse=reuse):
                init_state = tf.layers.dense(latent_vec, args.rnn_size, tf.nn.elu, reuse=reuse)
                lin_proj = tf.layers.Dense(self.params['vocab_size'], _scope='decoder/dense', _reuse=reuse)

                helper = tf.contrib.seq2seq.TrainingHelper(
                    inputs = tf.nn.embedding_lookup(embedding, self.dec_inp),
                    sequence_length = self.dec_seq_len)
                decoder = BasicDecoder(
                    cell = self.rnn_cell(reuse=reuse),
                    helper = helper,
                    initial_state = init_state,
                    concat_z = latent_vec)
                decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder = decoder)
                return decoder_output.rnn_output, lin_proj.apply(decoder_output.rnn_output)
        else:
            with tf.variable_scope(self.scopes['G'], reuse=True):
                init_state = tf.layers.dense(latent_vec, args.rnn_size, tf.nn.elu, reuse=True)

                decoder = BeamSearchDecoder(
                    cell = self.rnn_cell(reuse=True),
                    embedding = embedding,
                    start_tokens = tf.tile(tf.constant([self.params['<start>']], dtype=tf.int32), [self.batch_size]),
                    end_token = self.params['<end>'],
                    initial_state = tf.contrib.seq2seq.tile_batch(init_state, args.beam_width),
                    beam_width = args.beam_width,
                    output_layer = tf.layers.Dense(self.params['vocab_size'], _reuse=True),
                    concat_z = tf.tile(tf.expand_dims(latent_vec, 1), [1, args.beam_width, 1]))
                decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder = decoder,
                    maximum_iterations = 2 * tf.reduce_max(self.enc_seq_len))
                return decoder_output.predicted_ids[:, :, 0]

    
    def build_placeholders(self):
        self.enc_inp = tf.placeholder(tf.int32, [None, args.max_len])
        self.dec_inp = tf.placeholder(tf.int32, [None, args.max_len+1])
        self.dec_out = tf.placeholder(tf.int32, [None, args.max_len+1])
        self.labels = tf.placeholder(tf.int64, [None])

    
    def build_global_helpers(self):
        self.batch_size = tf.shape(self.enc_inp)[0]
        self.enc_seq_len = tf.count_nonzero(self.enc_inp, 1, dtype=tf.int32)
        self.dec_seq_len = self.enc_seq_len + 1
        self.global_step = tf.Variable(0, trainable=False)
        self.optimizer = tf.train.AdamOptimizer()
        self.gaussian = tf.truncated_normal([self.batch_size, args.latent_size])


    def draw_c_prior(self):
        return tf.contrib.distributions.OneHotCategorical(
            logits=tf.ones([self.batch_size, args.num_class]), dtype=tf.float32).sample()


    def draw_c(self, logits):
        return tf.contrib.distributions.OneHotCategorical(logits=logits, dtype=tf.float32).sample()


    def draw_z_prior(self):
        return self.gaussian


    def reparam(self, z_mean, z_logvar):
        return z_mean + tf.exp(0.5 * z_logvar) * self.draw_z_prior()


    def gradient_clipped(self, loss_op, scope=None):
        params = tf.trainable_variables(scope=scope)
        gradients = tf.gradients(loss_op, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, args.clip_norm)
        return zip(clipped_gradients, params)


    def seq_loss_fn(self, training_rnn_out, training_logits):
        mask = tf.sequence_mask(
            self.dec_seq_len, tf.reduce_max(self.dec_seq_len), dtype=tf.float32)
        if args.num_sampled >= self.params['vocab_size']:
            return tf.reduce_sum(tf.contrib.seq2seq.sequence_loss(
                logits = training_logits,
                targets = self.dec_out,
                weights = mask,
                average_across_timesteps = False,
                average_across_batch = True))
        else:
            with tf.variable_scope('Generator/decoder/dense', reuse=True):
                weights = tf.transpose(tf.get_variable('kernel'))
                biases = tf.get_variable('bias')
            return tf.reduce_sum(tf.reshape(mask,[-1]) * tf.nn.sampled_softmax_loss(
                weights = weights,
                biases = biases,
                labels = tf.reshape(self.dec_out, [-1, 1]),
                inputs = tf.reshape(training_rnn_out, [-1, args.rnn_size]),
                num_sampled = args.num_sampled,
                num_classes = self.params['vocab_size'],
            )) / tf.to_float(self.batch_size)


    def kl_w_fn(self):
        return args.kl_anneal_max * tf.sigmoid((10 / args.kl_anneal_bias) * (
            tf.to_float(self.global_step) - tf.constant(args.kl_anneal_bias / 2)))


    def kl_loss_fn(self, mean, logvar):
        return 0.5 * tf.reduce_sum(
            tf.exp(logvar) + tf.square(mean) - 1 - logvar) / tf.to_float(self.batch_size)


    def sparse_cross_entropy_fn(self, logits, labels):
        return tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))

        
    def cross_entropy_fn(self, logits, labels):
        return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=labels))


    def temperature_fn(self):
        return args.temperature_anneal_max * inverse_sigmoid((10 / args.temperature_anneal_bias) * (
            tf.to_float(self.global_step - args.temperature_start_step) - tf.constant(
                args.temperature_anneal_bias / 2)))


    def mutinfo_loss_fn(self, z_mean_new, z_logvar_new, epsilon=tf.constant(1e-10)):
        distribution = tf.contrib.distributions.MultivariateNormalDiag(
            z_mean_new, tf.exp(z_logvar_new), validate_args=True)
        mutinfo_loss = -tf.log(tf.add(epsilon, distribution.prob(self.draw_z_prior())))
        return tf.reduce_sum(mutinfo_loss) / tf.to_float(self.batch_size)


    def mse_fn(self, x1, x2):
        return tf.reduce_sum(tf.squared_difference(x1, x2)) / tf.to_float(self.batch_size)


    def rnn_cell(self, rnn_size=None, reuse=False):
        rnn_size = args.rnn_size if rnn_size is None else rnn_size
        return tf.nn.rnn_cell.GRUCell(rnn_size, kernel_initializer=tf.orthogonal_initializer(), reuse=reuse)


    def get_new_w(self, w):
        idx = self.params['word2idx'][w]
        return idx if idx < self.params['vocab_size'] else self.params['word2idx']['<unk>']


    def train_vae_session(self, sess, enc_inp, dec_inp, dec_out):
        _, nll_loss, kl_w, kl_loss, step = sess.run(
            [self.train_vae_op, self.train_vae_nll_loss, self.train_vae_kl_w, self.train_vae_kl_loss,
             self.global_step],
            {self.enc_inp: enc_inp, self.dec_inp: dec_inp, self.dec_out: dec_out})
        return {'nll_loss': nll_loss, 'kl_w': kl_w, 'kl_loss': kl_loss, 'step': step}

    
    def train_discriminator_session(self, sess, enc_inp, dec_inp, dec_out, labels):
        _, loss, acc, L_u, step = sess.run(
            [self.train_clf_op, self.train_clf_loss, self.train_clf_acc, self.L_u,
             self.global_step],
            {self.enc_inp: enc_inp, self.dec_inp: dec_inp, self.dec_out: dec_out, self.labels: labels})
        return {'clf_loss': loss, 'clf_acc': acc, 'step': step,
                'L_u': (args.lambda_u * L_u)}

    
    def train_generator_session(self, sess, enc_inp, dec_inp, dec_out):
        _, nll_loss, kl_w, kl_loss, step, temperature, l_attr_z, l_attr_c = sess.run(
            [self.train_generator_op, self.train_ge_vae_nll_loss, self.train_ge_vae_kl_w,
             self.train_ge_vae_kl_loss, self.global_step, self.temperature, self.l_attr_z, self.l_attr_c],
            {self.enc_inp: enc_inp, self.dec_inp: dec_inp, self.dec_out: dec_out})
        return {'nll_loss': nll_loss, 'kl_w': kl_w, 'kl_loss': kl_loss, 'step': step,
                'temperature': temperature, 'l_attr_z': (args.lambda_z * l_attr_z),
                'l_attr_c': (args.lambda_c * l_attr_c)}

    
    def train_encoder_session(self, sess, enc_inp, dec_inp, dec_out):
        _, nll_loss, kl_w, kl_loss, step = sess.run(
            [self.train_encoder_op, self.train_ge_vae_nll_loss, self.train_ge_vae_kl_w,
             self.train_ge_vae_kl_loss, self.global_step],
            {self.enc_inp: enc_inp, self.dec_inp: dec_inp, self.dec_out: dec_out})
        return {'nll_loss': nll_loss, 'kl_w': kl_w, 'kl_loss': kl_loss, 'step': step}


    def prior_inference(self, sess):
        predicted_ids = sess.run(self.prior_gen_ids, {self.batch_size: 1, self.enc_seq_len: [15]})[0]
        print('G: %s' % ' '.join([self.params['idx2word'][idx] for idx in predicted_ids]))
        print('-'*12)

    
    def post_inference(self, sess, sentence):
        print('I: %s' % sentence)
        print()
        sentence = [self.get_new_w(w) for w in sentence.split()][:args.max_len]
        sentence = sentence + [self.params['word2idx']['<pad>']] * (args.max_len-len(sentence))

        predicted_ids = sess.run(self.post_gen_ids, {self.enc_inp: np.atleast_2d(sentence)})[0]
        print('O: %s' % ' '.join([self.params['idx2word'][idx] for idx in predicted_ids]))
        print()
        predicted_ids = sess.run(self.post_reverse_gen_ids, {self.enc_inp: np.atleast_2d(sentence)})[0]
        print('R: %s' % ' '.join([self.params['idx2word'][idx] for idx in predicted_ids]))
        print('-'*12)
