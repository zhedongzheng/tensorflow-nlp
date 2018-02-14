from config import args

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
        outputs = outputs * np.sqrt(embed_dim)
    return outputs


def multihead_attn(queries, keys, q_masks, k_masks, num_units=None, num_heads=8,
        dropout_rate=args.dropout_rate, future_binding=False, reuse=False, activation=None):
    """
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q]
      keys: A 3d tensor with shape of [N, T_k, C_k]
    """
    if num_units is None:
        num_units = queries.get_shape().as_list[-1]
    T_q = queries.get_shape().as_list()[1]                                         # max time length of query
    T_k = keys.get_shape().as_list()[1]                                            # max time length of key

    Q = tf.layers.dense(queries, num_units, activation, reuse=reuse, name='Q')     # (N, T_q, C)
    K = tf.layers.dense(keys, num_units, activation, reuse=reuse, name='K')        # (N, T_k, C)
    V = tf.layers.dense(keys, num_units, activation, reuse=reuse, name='V')        # (N, T_k, C)

    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)                         # (h*N, T_q, C/h) 
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)                         # (h*N, T_k, C/h) 
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)                         # (h*N, T_k, C/h)

    # Scaled Dot-Product
    align = tf.matmul(Q_, tf.transpose(K_, [0,2,1]))                               # (h*N, T_q, T_k)
    align = align / np.sqrt(K_.get_shape().as_list()[-1])                         # scale

    # Key Masking
    paddings = tf.fill(tf.shape(align), float('-inf'))                             # exp(-large) -> 0

    key_masks = k_masks                                                            # (N, T_k)
    key_masks = tf.tile(key_masks, [num_heads, 1])                                 # (h*N, T_k)
    key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, T_q, 1])                 # (h*N, T_q, T_k)
    align = tf.where(tf.equal(key_masks, 0), paddings, align)                      # (h*N, T_q, T_k)

    if future_binding:
        lower_tri = tf.ones([T_q, T_k])                                            # (T_q, T_k)
        lower_tri = tf.contrib.linalg.LinearOperatorTriL(lower_tri).to_dense()     # (T_q, T_k)
        masks = tf.tile(tf.expand_dims(lower_tri,0), [tf.shape(align)[0], 1, 1])   # (h*N, T_q, T_k)
        align = tf.where(tf.equal(masks, 0), paddings, align)                      # (h*N, T_q, T_k)
    
    # Softmax
    align = tf.nn.softmax(align)                                                   # (h*N, T_q, T_k)

    # Query Masking
    query_masks = tf.to_float(q_masks)                                             # (N, T_q)
    query_masks = tf.tile(query_masks, [num_heads, 1])                             # (h*N, T_q)
    query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, T_k])            # (h*N, T_q, T_k)
    align *= query_masks                                                           # (h*N, T_q, T_k)

    align = tf.layers.dropout(align, dropout_rate, training=(not reuse))           # (h*N, T_q, T_k)

    # Weighted sum
    outputs = tf.matmul(align, V_)                                                 # (h*N, T_q, C/h)
    # Restore shape
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)              # (N, T_q, C)
    # Residual connection
    outputs += queries                                                             # (N, T_q, C)   
    # Normalize
    outputs = layer_norm(outputs)                                                  # (N, T_q, C)
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


def learned_position_encoding(inputs, mask, embed_dim):
    T = inputs.get_shape().as_list()[-1]
    outputs = tf.range(tf.shape(inputs)[1])                # (T_q)
    outputs = tf.expand_dims(outputs, 0)                   # (1, T_q)
    outputs = tf.tile(outputs, [tf.shape(inputs)[0], 1])   # (N, T_q)
    outputs = embed_seq(outputs, T, embed_dim, zero_pad=False, scale=False)
    return tf.expand_dims(tf.to_float(mask), -1) * outputs


def sinusoidal_position_encoding(inputs, mask, num_units):
    T = inputs.get_shape().as_list()[-1]
    position_idx = tf.tile(tf.expand_dims(tf.range(T), 0), [tf.shape(inputs)[0], 1])

    position_enc = np.array(
        [[pos / np.power(10000, 2.*i/num_units) for i in range(num_units)] for pos in range(T)])
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

    lookup_table = tf.convert_to_tensor(position_enc, tf.float32)
    outputs = tf.nn.embedding_lookup(lookup_table, position_idx)
    
    return tf.expand_dims(tf.to_float(mask), -1) * outputs


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
        crossent = tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=logits)
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
