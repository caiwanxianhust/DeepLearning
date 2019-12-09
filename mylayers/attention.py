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


def extract_seq_patches(x, kernel_size, rate):
    """x.shape = [None, seq_len, seq_dim]
    滑动地把每个窗口的x取出来，为做局部attention作准备。
    """
    seq_dim = K.int_shape(x)[-1]
    seq_len = K.shape(x)[1]
    k_size = kernel_size + (rate - 1) * (kernel_size - 1)
    p_right = (k_size - 1) // 2
    p_left = k_size - 1 - p_right
    x = K.temporal_padding(x, (p_left, p_right))
    xs = [x[:, i: i + seq_len] for i in range(0, k_size, rate)]
    x = K.concatenate(xs, 2)
    return K.reshape(x, (-1, seq_len, kernel_size, seq_dim))


def to_mask(x, mask, mode='mul'):
    """通用mask函数
    这里的mask.shape=[batch_size, seq_len]或[batch_size, seq_len, 1]
    """
    if mask is None:
        return x
    else:
        for _ in range(K.ndim(x) - K.ndim(mask)):
            mask = K.expand_dims(mask, K.ndim(mask))
        if mode == 'mul':
            return x * mask
        else:
            return x - (1 - mask) * 1e10


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
        if (self.mask_right is not False) or (self.mask_right is not None):
            if self.mask_right:
                # [1,1,seq_len_q,seq_len_k]
                ones = tf.ones_like(scores[:1, :1])
                # 不包含对角线的上三角矩阵，每个元素是1e9
                mask_ahead = (ones - tf.linalg.band_part(ones, -1, 0)) * 1e9
                # 遮掉所有未预测的词
                scores = scores - mask_ahead
            else:
                # 这种情况下，mask_right是外部传入的0/1矩阵，shape=[q_len, k_len]
                mask_ahead = (1 - K.constant(self.mask_right)) * 1e9
                mask_ahead = K.expand_dims(K.expand_dims(mask_ahead, 0), 0)
                self.mask_ahead = mask_ahead
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


class AtrousSelfAttention(keras.layers.Layer):
    """空洞多头自注意力机制
    说明：每个元素只跟相对距离为rate的倍数的元素有关联。
    """

    def __init__(self, latent_dim, heads, mask_right=False, rate=1, **kwargs):
        super(AtrousSelfAttention, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.heads = heads
        self.mask_right = mask_right
        assert self.latent_dim % heads == 0, "the latent_dim: {} must can be divisible by heads!"
        self.rate = rate

    def build(self, input_shape):
        super(AtrousSelfAttention, self).build(input_shape)
        self.attention = MultiHeadAttention(
            latent_dim=self.latent_dim,
            heads=self.heads,
            mask_right=self.mask_right
        )

    def call(self, inputs):
        # 如果inputs为列表
        if isinstance(inputs, list):
            x, x_mask = inputs
        else:
            x, x_mask = inputs, None
        # x:[bacth_size, seq_len, latent_dim)
        seq_dim = K.int_shape(x)[-1]
        # 补足长度，保证可以reshape
        seq_len = K.shape(x)[1]
        pad_len = self.rate - seq_len % self.rate
        # 向一个3维tensor的中间维度，左右padding指定长度
        x = K.temporal_padding(x, (0, pad_len))
        if x_mask is not None:
            x_mask = K.temporal_padding(x_mask, (0, pad_len))
        new_seq_len = K.shape(x)[1]
        # 变换shape
        x = K.reshape(x, (-1, new_seq_len // self.rate, self.rate, seq_dim))
        x = K.permute_dimensions(x, (0, 2, 1, 3))
        x = K.reshape(x, (-1, new_seq_len // self.rate, seq_dim))
        if x_mask is not None:
            x_mask = K.reshape(x_mask, (-1, new_seq_len // self.rate, self.rate, 1))
            x_mask = K.permute_dimensions(x_mask, (0, 2, 1, 3))
            x_mask = K.reshape(x_mask, (-1, new_seq_len // self.rate, 1))
        # 做attention
        if x_mask is not None:
            x = self.attention([x, x, x, x_mask, x_mask])
        else:
            x = self.attention([x, x, x])
        # 恢复shape
        x = K.reshape(x, (-1, self.rate, new_seq_len // self.rate, self.latent_dim))
        x = K.permute_dimensions(x, (0, 2, 1, 3))
        x = K.reshape(x, (-1, new_seq_len, self.latent_dim))
        x = x[:, : - pad_len]
        return x

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return (input_shape[0][0], input_shape[0][1], self.latent_dim)
        else:
            return (input_shape[0], input_shape[1], self.latent_dim)


class LocalSelfAttention(keras.layers.Layer):
    """局部多头自注意力机制
    说明：每个元素只跟相对距离不超过neighbors的元素有关联，这里的rate
    是真正的膨胀率（跟膨胀卷积一样），如果不了解可以忽略，默认为1就好。
    """

    def __init__(self, latent_dim, heads, mask_right=False, neighbors=1, rate=1, **kwargs):
        super(LocalSelfAttention, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.heads = heads
        self.mask_right = mask_right
        assert self.latent_dim % heads == 0, "the latent_dim: {} must can be divisible by heads!"
        self.rate = rate
        self.neighbors = neighbors

    def build(self, input_shape):
        super(LocalSelfAttention, self).build(input_shape)
        if self.mask_right:
            mask_right = np.ones((1, 1 + 2 * self.neighbors))
            mask_right[:, - self.neighbors:] = 0
        else:
            mask_right = self.mask_right
        self.attention = MultiHeadAttention(
            latent_dim=self.latent_dim,
            heads=self.heads,
            mask_right=mask_right
        )

    def call(self, inputs):
        if isinstance(inputs, list):
            x, x_mask = inputs
        else:
            x, x_mask = inputs, None
        # 提取局部特征
        kernel_size = 1 + 2 * self.neighbors
        # xp:[bacth_size, seq_len, kernel_size, latent_dim]
        xp = extract_seq_patches(x, kernel_size, self.rate)
        if x_mask is not None:
            xp_mask = extract_seq_patches(x_mask, kernel_size, self.rate)
        # 变换shape
        seq_len = K.shape(x)[1]
        seq_dim = K.int_shape(x)[-1]
        x = K.reshape(x, (-1, 1, seq_dim))
        xp = K.reshape(xp, (-1, kernel_size, seq_dim))
        if x_mask is not None:
            xp_mask = K.reshape(xp_mask, (-1, kernel_size, 1))
        # 做attention
        if x_mask is not None:
            # x:[batch_size * seq_len, 1, latent_dim]
            # xp:[bacth_size * seq_len, kernel_size, latent_dim]
            x = self.attention([x, xp, xp, xp_mask])
        else:
            x = self.attention([x, xp, xp])
        # 恢复shape
        x = K.reshape(x, (-1, seq_len, self.latent_dim))
        x = to_mask(x, x_mask, 'mul')
        return x

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return (input_shape[0][0], input_shape[0][1], self.latent_dim)
        else:
            return (input_shape[0], input_shape[1], self.latent_dim)


class SparseSelfAttention(keras.layers.Layer):
    """稀疏多头自注意力机制
    来自文章《Generating Long Sequences with Sparse Transformers》
    说明：每个元素只跟相对距离为rate的倍数的元素、以及相对距离不超过rate的元素有关联。
    """

    def __init__(self, latent_dim, heads, mask_right=False, rate=1, **kwargs):
        super(SparseSelfAttention, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.heads = heads
        assert latent_dim % heads == 0, "the latent_dim: {} must can be divisible by heads!"
        self.depth = self.latent_dim // heads
        assert rate != 1, 'if rate=1, please use SelfAttention directly'
        self.rate = rate
        self.neighbors = rate - 1
        self.mask_right = mask_right

    def build(self, input_shape):
        super(SparseSelfAttention, self).build(input_shape)
        self.q_dense = keras.layers.Dense(self.latent_dim, use_bias=False)
        self.k_dense = keras.layers.Dense(self.latent_dim, use_bias=False)
        self.v_dense = keras.layers.Dense(self.latent_dim, use_bias=False)

    def call(self, inputs):
        if isinstance(inputs, list):
            x, x_mask = inputs
        else:
            x, x_mask = inputs, None
        seq_dim = K.int_shape(x)[-1]
        # 补足长度，保证可以reshape
        seq_len = K.shape(x)[1]
        pad_len = self.rate - seq_len % self.rate
        x = K.temporal_padding(x, (0, pad_len))
        if x_mask is not None:
            x_mask = K.temporal_padding(x_mask, (0, pad_len))
        new_seq_len = K.shape(x)[1]
        x = K.reshape(x, (-1, new_seq_len, seq_dim))  # 经过padding后shape可能变为None，所以重新声明一下shape
        # 线性变换
        qw = self.q_dense(x)
        kw = self.k_dense(x)
        vw = self.v_dense(x)
        # 提取局部特征
        kernel_size = 1 + 2 * self.neighbors
        kwp = extract_seq_patches(kw, kernel_size, self.rate)  # shape=[None, seq_len, kernel_size, out_dim]
        vwp = extract_seq_patches(vw, kernel_size, self.rate)  # shape=[None, seq_len, kernel_size, out_dim]
        if x_mask is not None:
            xp_mask = extract_seq_patches(x_mask, kernel_size, self.rate)
        # 形状变换
        qw = K.reshape(qw, (-1, new_seq_len // self.rate, self.rate, self.heads, self.depth))
        kw = K.reshape(kw, (-1, new_seq_len // self.rate, self.rate, self.heads, self.depth))
        vw = K.reshape(vw, (-1, new_seq_len // self.rate, self.rate, self.heads, self.depth))
        kwp = K.reshape(kwp, (-1, new_seq_len // self.rate, self.rate, kernel_size, self.heads, self.depth))
        vwp = K.reshape(vwp, (-1, new_seq_len // self.rate, self.rate, kernel_size, self.heads, self.depth))
        if x_mask is not None:
            x_mask = K.reshape(x_mask, (-1, new_seq_len // self.rate, self.rate, 1, 1))
            xp_mask = K.reshape(xp_mask, (-1, new_seq_len // self.rate, self.rate, kernel_size, 1, 1))
        # 维度置换
        qw = K.permute_dimensions(qw, (0, 3, 2, 1, 4))  # shape=[None, heads, r, seq_len // r, size]
        kw = K.permute_dimensions(kw, (0, 3, 2, 1, 4))
        vw = K.permute_dimensions(vw, (0, 3, 2, 1, 4))
        qwp = K.expand_dims(qw, 4)
        kwp = K.permute_dimensions(kwp,
                                   (0, 4, 2, 1, 3, 5))  # shape=[None, heads, r, seq_len // r, kernel_size, out_dim]
        vwp = K.permute_dimensions(vwp, (0, 4, 2, 1, 3, 5))
        if x_mask is not None:
            x_mask = K.permute_dimensions(x_mask, (0, 3, 2, 1, 4))
            xp_mask = K.permute_dimensions(xp_mask, (0, 4, 2, 1, 3, 5))
        # Attention1
        a = K.batch_dot(qw, kw, [4, 4]) / self.depth ** 0.5
        a = K.permute_dimensions(a, (0, 1, 2, 4, 3))
        a = to_mask(a, x_mask, 'add')
        a = K.permute_dimensions(a, (0, 1, 2, 4, 3))
        if self.mask_right:
            ones = K.ones_like(a[: 1, : 1, : 1])
            mask = (ones - K.tf.matrix_band_part(ones, -1, 0)) * 1e10
            a = a - mask
        # Attention2
        ap = K.batch_dot(qwp, kwp, [5, 5]) / self.depth ** 0.5
        ap = K.permute_dimensions(ap, (0, 1, 2, 3, 5, 4))
        if x_mask is not None:
            ap = to_mask(ap, xp_mask, 'add')
        ap = K.permute_dimensions(ap, (0, 1, 2, 3, 5, 4))
        if self.mask_right:
            mask = np.ones((1, kernel_size))
            mask[:, - self.neighbors:] = 0
            mask = (1 - K.constant(mask)) * 1e10
            for _ in range(4):
                mask = K.expand_dims(mask, 0)
            ap = ap - mask
        ap = ap[..., 0, :]
        # 合并两个Attention
        A = K.concatenate([a, ap], -1)
        A = K.softmax(A)
        a, ap = A[..., : K.shape(a)[-1]], A[..., K.shape(a)[-1]:]
        # 完成输出1
        o1 = K.batch_dot(a, vw, [4, 3])
        # 完成输出2
        ap = K.expand_dims(ap, -2)
        o2 = K.batch_dot(ap, vwp, [5, 4])
        o2 = o2[..., 0, :]
        # 完成输出
        o = o1 + o2
        o = to_mask(o, x_mask, 'mul')
        o = K.permute_dimensions(o, (0, 3, 2, 1, 4))
        o = K.reshape(o, (-1, new_seq_len, self.latent_dim))
        o = o[:, : - pad_len]
        return o

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return (input_shape[0][0], input_shape[0][1], self.latent_dim)
        else:
            return (input_shape[0], input_shape[1], self.latent_dim)
