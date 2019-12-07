import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


def avgpooling(inp):
    # 将(bacth_size, seq_len, em_dim) pooling至(bacth_size, em_dim)
    x, mask = inp
    if mask is not None:
        mask = tf.cast(mask, tf.float32)
        for _ in range(K.ndim(x) - K.ndim(mask)):
            # => (bacth_size, seq_len, 1)
            mask = mask[..., tf.newaxis]
        x = tf.math.reduce_sum(x, 1) * mask / (tf.math.reduce_sum(mask, 1) + 1e-6)
    else:
        x = tf.math.reduce_mean(x, 1)
    return x


def maxpooling(inp):
    # 将(bacth_size, seq_len, em_dim) pooling至(bacth_size, em_dim)
    x, mask = inp
    if mask is not None:
        mask = tf.cast(mask, tf.float32)
        for _ in range(K.ndim(x) - K.ndim(mask)):
            # => (bacth_size, seq_len, 1)
            mask = mask[..., tf.newaxis]
        x -= (1 - mask) * 1e9
    return tf.math.reduce_max(x, 1)


class AttentionPooling1D(keras.layers.Layer):
    """
    通过加性attention，将向量序列转化为句向量
    """
    def __init__(self, h_dim=None, **kwargs):
        super(AttentionPooling1D, self).__init__(**kwargs)
        self.h_dim = h_dim

    def build(self, input_shape):
        super(AttentionPooling1D, self).build(input_shape)
        if self.h_dim is None:
            self.h_dim = input_shape[0][-1]
        self.k_dense = keras.layers.Dense(
            self.h_dim,
            use_bias=False,
            activation='tanh'
        )
        self.o_dense = keras.layers.Dense(1, use_bias=False)

    def call(self, inputs):
        # x_init: [batch_size, seq_len_x, em_dim]
        x_init, mask = inputs
        # 一层线性变换
        x = self.k_dense(x_init)
        # [batch_size, seq_len, 1]
        x = self.o_dense(x)
        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            for _ in range(K.ndim(x_init) - K.ndim(mask)):
                # mask: [batch_size, seq_len_x] => [batch_size, seq_len_x, 1, 1]
                mask = mask[..., tf.newaxis]
            x = x - (1 - mask) * 1e9
        # softmax计算每个词向量的权重
        x = K.softmax(x, 1)
        return K.sum(x * x_init, 1)  # [batch_size, em_dim]

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][-1])
