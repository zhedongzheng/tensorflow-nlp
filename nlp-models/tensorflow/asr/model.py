import tensorflow as tf


class Model:
    def __init__(self, num_classes, rnn_size=50, num_features=13, clip_norm=5.0):
        self.num_classes = num_classes
        self.rnn_size = rnn_size
        self.num_features = num_features
        self.clip_norm = clip_norm
        self.build_graph()
        
    def build_graph(self):
        self.forward()
        self.backward()

    def forward(self):
        self.inputs = tf.placeholder(tf.float32, [None, None, self.num_features])
        self.seq_lens = tf.placeholder(tf.int32, [None])
        self.targets = tf.sparse_placeholder(tf.int32)

        outputs, _ = tf.nn.dynamic_rnn(
            self.rnn_cell(), self.inputs, self.seq_lens, dtype=tf.float32)
        self.logits = tf.layers.dense(outputs, self.num_classes)
    
    def backward(self):
        time_major_logits = tf.transpose(self.logits, [1, 0, 2])

        decoded, log_prob = tf.nn.ctc_greedy_decoder(time_major_logits, self.seq_lens)
        decoded = tf.to_int32(decoded[0])
        self.predictions = tf.sparse_tensor_to_dense(decoded)

        self.loss = tf.reduce_mean(tf.nn.ctc_loss(self.targets, time_major_logits, self.seq_lens))
        self.edit_dist = tf.reduce_mean(tf.edit_distance(decoded, self.targets))

        params = tf.trainable_variables()
        grads = tf.gradients(self.loss, params)
        clipped_grads, _ = tf.clip_by_global_norm(grads, self.clip_norm)
        self.train_op = tf.train.AdamOptimizer().apply_gradients(zip(clipped_grads, params))

    def rnn_cell(self):
        return tf.nn.rnn_cell.GRUCell(self.rnn_size,
            kernel_initializer=tf.orthogonal_initializer())

    def train_batch(self, sess, inputs, seq_lens, sparse_targets):
        loss, edit_dist, _ = sess.run([self.loss, self.edit_dist, self.train_op],
            {self.inputs: inputs,
             self.seq_lens: seq_lens,
             self.targets: sparse_targets})
        return loss, edit_dist

    def test_batch(self, sess, inputs, seq_lens):
        preds = sess.run(self.predictions,
            {self.inputs: inputs,
             self.seq_lens: seq_lens})
        return preds
    