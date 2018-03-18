import tensorflow as tf
import paddle.v2 as paddle


class Model:
    def __init__(self):
        self.build_graph()


    def build_graph(self):
        self.interfaces()
        self.forward()
        self.backward()


    def interfaces(self):
        self.placeholders = {
            'user_id': tf.placeholder(tf.int32, [None]),
            'gender_id': tf.placeholder(tf.int32, [None]),
            'age_id': tf.placeholder(tf.int32, [None]),
            'job_id': tf.placeholder(tf.int32, [None]),
            'movie_id': tf.placeholder(tf.int32, [None]),
            'category_ids': tf.placeholder(tf.float32, [None, 18]),
            'movie_title': tf.placeholder(tf.int32, [None, 10]),
            'score': tf.placeholder(tf.float32, [None])}
        self.ops = {
            'user_features': None,
            'movie_features': None,
            'predict': None,
            'loss': None,
            'train': None}


    def forward(self):
        with tf.variable_scope('user_id'):
            user_id_embed = tf.contrib.layers.embed_sequence(
                ids = self.placeholders['user_id'],
                vocab_size = paddle.dataset.movielens.max_user_id()+1,
                embed_dim = 32)
            user_id_fc = tf.layers.dense(user_id_embed, 32)

        with tf.variable_scope('gender_id'):
            gender_id_embed = tf.contrib.layers.embed_sequence(
                ids = self.placeholders['gender_id'],
                vocab_size = 2,
                embed_dim = 16)
            gender_id_fc = tf.layers.dense(gender_id_embed, 16)
        
        with tf.variable_scope('age_id'):
            age_id_embed = tf.contrib.layers.embed_sequence(
                ids = self.placeholders['age_id'],
                vocab_size = len(paddle.dataset.movielens.age_table),
                embed_dim = 16)
            age_id_fc = tf.layers.dense(age_id_embed, 16)

        with tf.variable_scope('job_id'):
            job_id_embed = tf.contrib.layers.embed_sequence(
                ids = self.placeholders['job_id'],
                vocab_size = paddle.dataset.movielens.max_job_id()+1,
                embed_dim = 16)
            job_id_fc = tf.layers.dense(job_id_embed, 16)

        user_features = tf.concat([user_id_fc, gender_id_fc, age_id_fc, job_id_fc], -1)
        self.ops['user_features'] = tf.layers.dense(user_features, 200, tf.tanh)

        with tf.variable_scope('movie_id'):
            movie_id_embed = tf.contrib.layers.embed_sequence(
                ids = self.placeholders['movie_id'],
                vocab_size = paddle.dataset.movielens.max_movie_id()+1,
                embed_dim = 32)
            movie_id_fc = tf.layers.dense(movie_id_embed, 32)

        with tf.variable_scope('category_ids'):
            category_fc = tf.layers.dense(self.placeholders['category_ids'], 32)

        with tf.variable_scope('movie_title'):
            movie_title_embed = tf.contrib.layers.embed_sequence(
                ids = self.placeholders['movie_title'],
                vocab_size = 5175,
                embed_dim = 32)
            movie_title_conv = tf.layers.conv1d(movie_title_embed, 32, 3)
            movie_title_fc = self.global_max_pooling(movie_title_conv)
        
        movie_features = tf.concat([movie_id_fc, category_fc, movie_title_fc], -1)
        self.ops['movie_features'] = tf.layers.dense(movie_features, 200, tf.tanh)


    def backward(self):
        user_norm = tf.nn.l2_normalize(self.ops['user_features'], -1)
        movie_norm = tf.nn.l2_normalize(self.ops['movie_features'], -1)
        cos_sim = tf.reduce_sum(tf.multiply(user_norm, movie_norm), -1)
        self.ops['predict'] = 5 * cos_sim
        self.ops['loss'] = tf.reduce_mean(tf.squared_difference(
            self.ops['predict'], self.placeholders['score']))
        self.ops['train'] = tf.train.AdamOptimizer(1e-4).minimize(self.ops['loss'])


    def global_max_pooling(self, x):
        batch_size = tf.shape(self.placeholders['user_id'])[0]
        num_units = x.get_shape().as_list()[-1]
        x = tf.layers.max_pooling1d(x, x.get_shape().as_list()[1], 1)
        x = tf.reshape(x, [batch_size, num_units])
        return x


    def train_batch(self, sess,
        user_id, gender_id, age_id, job_id,
        movie_id, category_ids, movie_title,
        score):
        loss, _ = sess.run([self.ops['loss'], self.ops['train']],
            {self.placeholders['user_id']: user_id,
             self.placeholders['gender_id']: gender_id,
             self.placeholders['age_id']: age_id,
             self.placeholders['job_id']: job_id,
             self.placeholders['movie_id']: movie_id,
             self.placeholders['category_ids']: category_ids,
             self.placeholders['movie_title']: movie_title,
             self.placeholders['score']: score})
        return loss


    def predict_batch(self, sess,
        user_id, gender_id, age_id, job_id,
        movie_id, category_ids, movie_title,
        score):
        pred, loss = sess.run([self.ops['predict'], self.ops['loss']],
            {self.placeholders['user_id']: user_id,
             self.placeholders['gender_id']: gender_id,
             self.placeholders['age_id']: age_id,
             self.placeholders['job_id']: job_id,
             self.placeholders['movie_id']: movie_id,
             self.placeholders['category_ids']: category_ids,
             self.placeholders['movie_title']: movie_title,
             self.placeholders['score']: score})
        return pred, loss
