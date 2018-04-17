import numpy as np
import tensorflow as tf


def layer_norm(inputs, epsilon=1e-8):
    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    normalized = (inputs - mean) / (tf.sqrt(variance + epsilon))

    params_shape = inputs.get_shape()[-1:]
    gamma = tf.get_variable('gamma', params_shape, tf.float32, tf.ones_initializer())
    beta = tf.get_variable('beta', params_shape, tf.float32, tf.zeros_initializer())
    
    outputs = gamma * normalized + beta
    return outputs


def embed_seq(inputs, vocab_size=None, embed_dim=None, zero_pad=False, scale=False):
    lookup_table = tf.get_variable('lookup_table', dtype=tf.float32, shape=[vocab_size, embed_dim])

    if zero_pad:
        lookup_table = tf.concat((tf.zeros([1, embed_dim]), lookup_table[1:, :]), axis=0)
    
    outputs = tf.nn.embedding_lookup(lookup_table, inputs)

    if scale:
        outputs = outputs * (embed_dim ** 0.5)
     
    return outputs


def pointwise_feedforward(inputs, num_units=[None, None], activation=None):
    # Inner layer
    outputs = tf.layers.conv1d(inputs, num_units[0], kernel_size=1, activation=activation)
    # Readout layer
    outputs = tf.layers.conv1d(outputs, num_units[1], kernel_size=1, activation=None)
    # Residual connection
    outputs += inputs
    # Normalize
    outputs = layer_norm(outputs)
    return outputs


def learned_positional_encoding(inputs, embed_dim, zero_pad=False, scale=False):
    T = inputs.get_shape().as_list()[1]
    outputs = tf.range(T)                                  # (T_q)
    outputs = tf.expand_dims(outputs, 0)                   # (1, T_q)
    outputs = tf.tile(outputs, [tf.shape(inputs)[0], 1])   # (N, T_q)
    return embed_seq(outputs, T, embed_dim, zero_pad=zero_pad, scale=scale)


def sinusoidal_positional_encoding(inputs, num_units, zero_pad=False, scale=False):
    T = inputs.get_shape().as_list()[1]
    position_idx = tf.tile(tf.expand_dims(tf.range(T), 0), [tf.shape(inputs)[0], 1])

    position_enc = np.array(
        [[pos / np.power(10000, 2.*i/num_units) for i in range(num_units)] for pos in range(T)])
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
    lookup_table = tf.convert_to_tensor(position_enc, tf.float32)

    if zero_pad:
        lookup_table = tf.concat([tf.zeros([1, num_units]), lookup_table[1:, :]], axis=0)

    outputs = tf.nn.embedding_lookup(lookup_table, position_idx)

    if scale:
        outputs = outputs * num_units ** 0.5

    return outputs


def label_smoothing(inputs, epsilon=0.1):
    C = inputs.get_shape().as_list()[-1] # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / C)


def label_smoothing_sequence_loss(logits,
                                  targets,
                                  weights,
                                  label_depth,
                                  average_across_timesteps=True,
                                  average_across_batch=True,
                                  name=None):
    if len(logits.get_shape()) != 3:
        raise ValueError("Logits must be a "
                        "[batch_size x sequence_length x logits] tensor")
    if len(targets.get_shape()) != 2:
        raise ValueError("Targets must be a [batch_size x sequence_length] "
                        "tensor")
    if len(weights.get_shape()) != 2:
        raise ValueError("Weights must be a [batch_size x sequence_length] "
                        "tensor")
    
    with tf.name_scope(name, "sequence_loss", [logits, targets, weights]):
        targets = label_smoothing(tf.one_hot(targets, depth=label_depth))
        crossent = tf.nn.softmax_cross_entropy_with_logits_v2(labels=targets, logits=logits)
        crossent = tf.reshape(crossent, [-1]) *  tf.reshape(weights, [-1])
        
        if average_across_timesteps and average_across_batch:
            crossent = tf.reduce_sum(crossent)
            total_size = tf.reduce_sum(weights)
            total_size += 1e-12  # to avoid division by 0 for all-0 weights
            crossent /= total_size
        else:
            batch_size = tf.shape(logits)[0]
            sequence_length = tf.shape(logits)[1]
            crossent = tf.reshape(crossent, [batch_size, sequence_length])
        if average_across_timesteps and not average_across_batch:
            crossent = tf.reduce_sum(crossent, axis=[1])
            total_size = tf.reduce_sum(weights, axis=[1])
            total_size += 1e-12  # to avoid division by 0 for all-0 weights
            crossent /= total_size
        if not average_across_timesteps and average_across_batch:
            crossent = tf.reduce_sum(crossent, axis=[0])
            total_size = tf.reduce_sum(weights, axis=[0])
            total_size += 1e-12  # to avoid division by 0 for all-0 weights
            crossent /= total_size
        return crossent
