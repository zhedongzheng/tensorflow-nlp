from config import args

import tensorflow as tf
import numpy as np
import pprint


class MemoryNetwork:
    def __init__(self, params):
        self.params = params
        self.build_graph()


    def build_graph(self):
        self.build_placeholders()
        logits = self.forward()
        self.backward(logits)


    def build_placeholders(self):
        self.placeholders = {
            'inputs': tf.placeholder(
                tf.int64, [None, self.params['max_input_len'], self.params['max_sent_len']]),
            'questions': tf.placeholder(tf.int64, [None, self.params['max_quest_len']]),
            'answers': tf.placeholder(tf.int64, [None, self.params['max_answer_len']]),
            'inputs_len': tf.placeholder(tf.int32, [None]),
            'inputs_sent_len': tf.placeholder(tf.int32, [None, self.params['max_input_len']]),
            'questions_len': tf.placeholder(tf.int32, [None]),
            'answers_len': tf.placeholder(tf.int32, [None]),
            'is_training': tf.placeholder(tf.bool)}


    def forward(self):
        memory_o = self.input_pipe(self.placeholders['inputs'], 'memory_o')
        memory_i = self.input_pipe(self.placeholders['inputs'], 'memory_i')
        question = self.quest_pipe(self.placeholders['questions'])

        match = tf.matmul(question, tf.transpose(memory_i, [0,2,1]))
        match = tf.nn.softmax(match) # (batch, question_maxlen, story_maxlen)

        response = tf.matmul(match, memory_o)
        answer = tf.layers.flatten(tf.concat([response, question], -1))
        answer = tf.layers.dense(answer, args.hidden_dim)
        
        logits = self.answer_module(answer)
        return logits


    def quest_pipe(self, x):
        with tf.variable_scope('question'):
            x = self.embed_seq(x)
            x = tf.layers.dropout(x, args.dropout_rate,
                training=self.placeholders['is_training'],
                noise_shape=[tf.shape(x)[0], 1, args.hidden_dim])
            pos = self.position_encoding(
                self.params['max_quest_len'], args.hidden_dim)
        return (x * pos)


    def input_pipe(self, x, name):
        with tf.variable_scope('input_'+name):
            x = self.embed_seq(x)
            x = tf.layers.dropout(x, args.dropout_rate,
                training=self.placeholders['is_training'],
                noise_shape=[tf.shape(x)[0], 1, 1, args.hidden_dim])
            pos = self.position_encoding(
                self.params['max_sent_len'], args.hidden_dim)
            x = tf.reduce_sum(x * pos, 2)
        return x


    def backward(self, logits):
        targets = self.placeholders['answers']
        masks = tf.ones_like(targets, tf.float32)
        self.loss_op = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(
            logits=logits, targets=targets, weights=masks))
        self.acc_op = tf.reduce_mean(
            tf.to_float(tf.equal(tf.argmax(logits, -1), targets)))
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss_op)


    def answer_module(self, answer):
        init_state = tf.layers.dropout(
            answer, args.dropout_rate, training=self.placeholders['is_training'])
        answer_inputs = self.shift_right(self.placeholders['answers'])

        with tf.variable_scope('input_memory_o', reuse=True):
            embedding = tf.get_variable('lookup_table')
        
        with tf.variable_scope('answer_module'):
            helper = tf.contrib.seq2seq.TrainingHelper(
                inputs = tf.nn.embedding_lookup(embedding, answer_inputs),
                sequence_length = self.placeholders['answers_len'])
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell = self.GRU(),
                helper = helper,
                initial_state = init_state,
                output_layer = tf.layers.Dense(self.params['vocab_size']))
            decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = decoder)
        logits = decoder_output.rnn_output

        with tf.variable_scope('answer_module', reuse=True):
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding = embedding,
                start_tokens = tf.tile(
                    tf.constant([self.params['<start>']], dtype=tf.int32), [self.batch_size]),
                end_token = self.params['<end>'])
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell = self.GRU(reuse=True),
                helper = helper,
                initial_state = init_state,
                output_layer = tf.layers.Dense(self.params['vocab_size'], _reuse=True))
            decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = decoder,
                maximum_iterations = self.params['max_answer_len'])
        self.predicted_ids = decoder_output.sample_id

        return logits


    def shift_right(self, x):
        self.batch_size = tf.shape(self.placeholders['inputs'])[0]
        start = tf.to_int64(tf.fill([self.batch_size, 1], self.params['<start>']))
        return tf.concat([start, x[:, :-1]], 1)


    def GRU(self, reuse=None):
        return tf.nn.rnn_cell.GRUCell(
            args.hidden_dim, kernel_initializer=tf.orthogonal_initializer(), reuse=reuse)


    def position_encoding(self, sentence_size, embedding_size):
        encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
        ls = sentence_size + 1
        le = embedding_size + 1
        for i in range(1, le):
            for j in range(1, ls):
                encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
        encoding = 1 + 4 * encoding / embedding_size / sentence_size
        return np.transpose(encoding)

    def embed_seq(self, x, zero_pad=False):
        lookup_table = tf.get_variable('lookup_table',
            dtype=tf.float32, shape=[self.params['vocab_size'], args.hidden_dim])
        if zero_pad:
            lookup_table = tf.concat((tf.zeros([1, args.hidden_dim]), lookup_table[1:, :]), axis=0)
        return tf.nn.embedding_lookup(lookup_table, x)


    def train_session(self, sess, batch):
        (i, q, a, i_len, i_sent_len, q_len, a_len) = batch
        feed_dict = {
            self.placeholders['inputs']: i,
            self.placeholders['questions']: q,
            self.placeholders['answers']: a,
            self.placeholders['inputs_len']: i_len,
            self.placeholders['inputs_sent_len']: i_sent_len,
            self.placeholders['questions_len']: q_len,
            self.placeholders['answers_len']:a_len,
            self.placeholders['is_training']: True}
        _, loss, acc = sess.run([self.train_op, self.loss_op, self.acc_op], feed_dict)
        return loss, acc


    def predict_session(self, sess, batch):
        (i, q, _, i_len, i_sent_len, q_len, _) = batch
        feed_dict = {
            self.placeholders['inputs']: i,
            self.placeholders['questions']: q,
            self.placeholders['inputs_len']: i_len,
            self.placeholders['inputs_sent_len']: i_sent_len,
            self.placeholders['questions_len']: q_len,
            self.placeholders['is_training']: False}
        ids = sess.run(self.predicted_ids, feed_dict)
        return ids


    def demo_session(self, sess, i, q, i_len, i_sent_len, q_len, idx2word, demo, demo_idx):
        feed_dict = {
            self.placeholders['inputs']: np.expand_dims(i, 0),
            self.placeholders['questions']: np.atleast_2d(q),
            self.placeholders['inputs_len']: np.atleast_1d(i_len),
            self.placeholders['inputs_sent_len']: np.atleast_2d(i_sent_len),
            self.placeholders['questions_len']: np.atleast_1d(q_len),
            self.placeholders['is_training']: False}
        ids = sess.run(self.predicted_ids, feed_dict)[0]

        demo_i, demo_q, demo_a = demo
        print()
        pprint.pprint(demo_i[demo_idx])
        print()
        print('Question:', demo_q[demo_idx])
        print()
        print('Ground Truth:', demo_a[demo_idx])
        print()
        print('- '*12)
        print('Machine Answer:', [idx2word[id] for id in ids])
