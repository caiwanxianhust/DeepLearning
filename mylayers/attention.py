import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras


def scaled_dot_product_attention(q, k, v, mask=None, return_weights=False):
    # q:(batch_size,..., seq_len_q, dim), k:(batch_size,..., seq_len_k, dim)
    # => (batch_size,..., seq_len_q, seq_len_k)
    scores = tf.matmul(q, k, transpose_b=True)
    # 缩放因子
    dk = tf.cast(tf.shape(q)[-1], tf.float32)
    # scores[:, i, j] means the simility of the q[j] with k[j]
    scores = scores / tf.math.sqrt(dk)
    if mask is not None:
        mask = tf.cast(mask, tf.float32)
        for _ in range(K.ndim(scores) - K.ndim(mask)):
            mask = mask[..., tf.newaxis]
        scores -= (1 - mask) * 1e9
    scores = tf.math.softmax(scores, -1)
    out = tf.matmul(scores, v)
    if return_weights:
        return out, scores
    return out


class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, latent_dim, heads, mask_right=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.heads = heads
        self.mask_right = mask_right
        assert self.latent_dim % heads == 0, "the latent_dim: {} must can be divisible by heads!"
        self.depth = self.latent_dim // heads
        self.q_dense = keras.layers.Dense(self.latent_dim, use_bias=False)
        self.k_dense = keras.layers.Dense(self.latent_dim, use_bias=False)
        self.v_dense = keras.layers.Dense(self.latent_dim, use_bias=False)

    def call(self, inp):
        q, k, v = inp[:3]
        v_mask, q_mask = None, None
        if len(inp) > 3:
            # x_mask: [batch_size, seq_len_x]
            v_mask = inp[3]
            if len(inp) > 4:
                q_mask = inp[4]
        wq = self.q_dense(q)
        wk = self.k_dense(k)
        wv = self.v_dense(v)
        # (batch_size, seq_len, latent_dim) =>(batch_size, seq_len, heads, depth)
        wq = tf.reshape(wq, (tf.shape(wq)[0], tf.shape(wq)[1], self.heads, self.depth))
        wk = tf.reshape(wk, (tf.shape(wk)[0], tf.shape(wk)[1], self.heads, self.depth))
        wv = tf.reshape(wv, (tf.shape(wv)[0], tf.shape(wv)[1], self.heads, self.depth))
        # (batch_size, seq_len, heads, depth) => (batch_size, heads, seq_len, depth)
        wq = tf.transpose(wq, perm=(0, 2, 1, 3))
        wk = tf.transpose(wk, perm=(0, 2, 1, 3))
        wv = tf.transpose(wv, perm=(0, 2, 1, 3))
        # => (batch_size, heads, seq_len_q, seq_len_k)
        scores = tf.matmul(wq, wk, transpose_b=True)
        # 缩放因子
        dk = tf.cast(self.depth, tf.float32)
        # scores[:, i, j] means the simility of the q[j] with k[j]
        scores = scores / tf.math.sqrt(dk)

        if v_mask is not None:
            # v_mask:(batch_size, seq_len_k)
            v_mask = tf.cast(v_mask, tf.float32)
            # (batch_size, seq_len_k) => (batch_size, 1, 1, seq_len_k)
            for _ in range(K.ndim(scores) - K.ndim(v_mask)):
                v_mask = tf.expand_dims(v_mask, 1)
            scores -= (1 - v_mask) * 1e9
        # 解码端，自注意力时使用。预测第三个词仅使用前两个词
        if self.mask_right:
            # [1,1,seq_len_q,seq_len_k]
            ones = tf.ones_like(scores[:1, :1])
            # 不包含对角线的上三角矩阵，每个元素是1e9
            mask_ahead = (ones - tf.linalg.band_part(ones, -1, 0)) * 1e9
            # 遮掉所有未预测的词
            scores = scores - mask_ahead
        scores = tf.math.softmax(scores, -1)
        # (batch_size, heads, seq_len_q, seq_len_k) => (batch_size, heads, seq_len_q, depth)
        out = tf.matmul(scores, wv)
        # (batch_size, heads, seq_len_q, depth) => (batch_size, seq_len_q, heads, depth)
        out = tf.transpose(out, perm=(0, 2, 1, 3))
        # (batch_size, seq_len_q, heads, depth) => (batch_size, seq_len_q, latent_dim)
        out = tf.reshape(out, (tf.shape(out)[0], tf.shape(out)[1], self.latent_dim))
        if q_mask:
            # q_mask:(batch_size, seq_len_q)
            q_mask = tf.cast(q_mask, tf.float32)
            # (batch_size, seq_len_q) => (batch_size, seq_len_q, 1)
            for _ in range(K.ndim(out) - K.ndim(q_mask)):
                q_mask = q_mask[..., tf.newaxis]
            out *= q_mask
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.latent_dim)
