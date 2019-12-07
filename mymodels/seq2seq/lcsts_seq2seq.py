import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


def to_one_hot(x, vocab_size=7020):
    """输出一个词表大小的向量，来标记该词是否在文章出现过
    """
    # x: [batch_size, seq_length]
    # x_mask, x_mask, vocab_size: [batch_size, seq_length, 1]
    x, x_mask = x
    x_mask = tf.cast(x_mask[:, :, tf.newaxis], tf.float32)
    # 将x转换为整型用于onehot输入
    x = tf.cast(x, 'int32')
    # 产生一个[batch_size, seq_length, vocab_size]的张量。每个词用长度为vocab_size的onehot向量表示
    x = tf.one_hot(x, vocab_size)
    # x_mask * x : 得到[batch_size, seq_length, vocab_size]的张量，把未出现的词mask掉
    x = x_mask * x
    # 在axis=1轴上求和，相当于把每个词向量相加，词频统计，产生[batch_size, 1, vocab_size]张量
    x = tf.math.reduce_sum(x_mask * x, 1, keepdims=True)
    # 词频大于0设为1，否则为0
    x = tf.cast(tf.math.greater(x, 0.5), 'float32')
    return x


class ScaleShift(keras.layers.Layer):
    """缩放平移变换层（Scale and shift）
    """

    def __init__(self, **kwargs):
        super(ScaleShift, self).__init__(**kwargs)

    def build(self, input_shape):
        # [1,1,vocab_size]
        kernel_shape = (1,) * (len(input_shape) - 1) + (input_shape[-1],)
        self.log_scale = self.add_weight(name='log_scale',
                                         shape=kernel_shape,
                                         initializer='zeros')
        self.shift = self.add_weight(name='shift',
                                     shape=kernel_shape,
                                     initializer='zeros')

    def call(self, inputs):
        x_outs = tf.math.exp(self.log_scale) * inputs + self.shift
        return x_outs


def seq_avgpool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做avgpooling。
    """
    seq, mask = x
    # [None, seq_len] => [None, seq_len, 1]
    mask = tf.cast(mask[:, :, tf.newaxis], tf.float32)
    return tf.math.reduce_sum(seq * mask, 1) / (tf.math.reduce_sum(mask, 1) + 1e-6)


def seq_maxpool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做maxpooling。
    return: [None, s_size]
    """
    seq, mask = x
    mask = tf.cast(mask[:, :, tf.newaxis], tf.float32)
    seq -= (1 - mask) * 1e9
    return K.max(seq, 1)


class SelfModulatedLayerNormalization(keras.layers.Layer):
    """模仿Self-Modulated Batch Normalization，
    只不过将Batch Normalization改为Layer Normalization
    """

    def __init__(self, num_hidden, **kwargs):
        super(SelfModulatedLayerNormalization, self).__init__(**kwargs)
        self.num_hidden = num_hidden

    def build(self, input_shape):
        super(SelfModulatedLayerNormalization, self).build(input_shape)
        # embedding_dim
        output_dim = input_shape[0][-1]
        self.layernorm = keras.layers.LayerNormalization(center=False, scale=False)
        self.beta_dense_1 = keras.layers.Dense(self.num_hidden, activation='relu')
        self.beta_dense_2 = keras.layers.Dense(output_dim)
        self.gamma_dense_1 = keras.layers.Dense(self.num_hidden, activation='relu')
        self.gamma_dense_2 = keras.layers.Dense(output_dim)

    def call(self, inputs):
        # y: [batch_size, seq_len_y, embedding_dim]
        # x_max:[batch_size, z_dim]
        # inputs:y, cond: x_max
        inputs, cond = inputs
        inputs = self.layernorm(inputs)
        # [batch_size, z_dim] => [batch_size, num_hidden]
        beta = self.beta_dense_1(cond)
        # [batch_size, num_hidden] => [batch_size, embedding_dim]
        beta = self.beta_dense_2(beta)
        # [batch_size, z_dim] => [batch_size, num_hidden]
        gamma = self.gamma_dense_1(cond)
        # [batch_size, num_hidden] => [batch_size, embedding_dim]
        gamma = self.gamma_dense_2(gamma)
        for _ in range(K.ndim(inputs) - K.ndim(cond)):
            # [batch_size, embedding_dim] => [batch_size, 1, embedding_dim]
            beta = tf.expand_dims(beta, 1)
            gamma = tf.expand_dims(gamma, 1)
        # [batch_size, seq_len_y, embedding_dim]
        return inputs * (gamma + 1) + beta

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class MultiHeadAttention(keras.layers.Layer):
    """多头注意力机制
    """

    def __init__(self, heads, size_per_head, key_size=None,
                 mask_right=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        # 头数
        self.heads = heads
        # 每个头的维度
        self.size_per_head = size_per_head
        self.out_dim = heads * size_per_head
        self.key_size = key_size if key_size else size_per_head
        self.mask_right = mask_right

    def build(self, input_shape):
        super(MultiHeadAttention, self).build(input_shape)
        self.q_dense = keras.layers.Dense(self.key_size * self.heads, use_bias=False)
        self.k_dense = keras.layers.Dense(self.key_size * self.heads, use_bias=False)
        self.v_dense = keras.layers.Dense(self.out_dim, use_bias=False)

    def mask(self, x, mask, mode='mul'):
        # x:[batch_size,seq_len_x,seq_len_y,heads]
        if mask is None:
            return x
        else:
            for _ in range(K.ndim(x) - K.ndim(mask)):
                # mask: [batch_size, seq_len_x] => [batch_size, seq_len_x, 1, 1]
                mask = tf.cast(mask[..., tf.newaxis], tf.float32)
            if mode == 'mul':
                return x * mask
            else:
                return x - (1 - mask) * 1e9

    def call(self, inputs):
        # q:y; k:x; v:x
        q, k, v = inputs[:3]
        v_mask, q_mask = None, None
        if len(inputs) > 3:
            # x_mask: [batch_size, seq_len_x]
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]
        # 线性变换
        # [batch_size, seq_len_y, z_dim] => [batch_size, seq_len_y, size_per_head*heads]
        qw = self.q_dense(q)
        # [batch_size, seq_len_x, z_dim] => [batch_size, seq_len_x, size_per_head*heads]
        kw = self.k_dense(k)
        # [batch_size, seq_len_x, z_dim] => [batch_size, seq_len_x, size_per_head*heads]
        vw = self.v_dense(v)
        # 形状变换
        # [batch_size, seq_length, size_per_head*heads] => [batch_size,seq_length,heads,size_per_head]
        qw = tf.reshape(qw, (-1, tf.shape(qw)[1], self.heads, self.key_size))
        kw = tf.reshape(kw, (-1, tf.shape(kw)[1], self.heads, self.key_size))
        vw = tf.reshape(vw, (-1, tf.shape(vw)[1], self.heads, self.size_per_head))
        # 维度置换
        # [batch_size,seq_len,heads,size_per_head] => [batch_size,heads,seq_len,size_per_head]
        qw = tf.transpose(qw, perm=(0, 2, 1, 3))
        kw = tf.transpose(kw, perm=(0, 2, 1, 3))
        vw = tf.transpose(vw, perm=(0, 2, 1, 3))
        # Attention
        # out: [batch_size,heads,seq_len_y,seq_len_x]
        a = tf.matmul(qw, kw, transpose_b=True) / tf.math.sqrt(tf.cast(self.key_size, tf.float32))
        # [batch_size,heads,seq_len_y,seq_len_x] => [batch_size,seq_len_x,seq_len_y,heads]
        a = tf.transpose(a, perm=(0, 3, 2, 1))
        a = self.mask(a, v_mask, 'add')
        # [batch_size,seq_len_x,seq_len_y,heads] => [batch_size,heads,seq_len_y,seq_len_x]
        a = tf.transpose(a, perm=(0, 3, 2, 1))
        if self.mask_right:
            # [1,1,seq_len_y,seq_len_x]
            ones = tf.ones_like(a[:1, :1])
            # 不包含对角线的上三角矩阵，每个元素是1e9
            mask = (ones - tf.linalg.band_part(ones, -1, 0)) * 1e9
            # 遮掉所有未预测的词
            a = a - mask
        a = tf.nn.softmax(a, axis=-1)
        # 完成输出
        # a:[batch_size,heads,seq_len_y,seq_len_x]; vw:[batch_size,heads,seq_len_x,size_per_head]
        # out:[batch_size,heads,seq_len_y,size_per_head]
        o = tf.matmul(a, vw)
        # [batch_size,heads,seq_len_y,size_per_head] => [batch_size,seq_len_y,heads,size_per_head]
        o = tf.transpose(o, perm=(0, 2, 1, 3))
        # [batch_size,seq_len_y,heads,size_per_head] =>  [batch_size,seq_len_y, heads * size_per_headd]
        o = tf.reshape(o, (-1, tf.shape(o)[1], self.out_dim))
        o = self.mask(o, q_mask, 'mul')
        return o

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)


def selfcross_entropy(y_true, y_pred, from_logits=False, axis=-1):
    y_mask = tf.cast(tf.math.greater(y_true, 0), tf.float32)
    cross_entropy = K.sparse_categorical_crossentropy(y_true[:, 1:], y_pred[:, :-1],
                                                      from_logits=from_logits, axis=axis)
    cross_entropy = K.sum(cross_entropy * tf.cast(y_mask[:, 1:], tf.float32)) / K.sum(
        tf.cast(y_mask[:, 1:], tf.float32))
    return cross_entropy


class MySeq2seq(object):
    def __init__(self, vocab_size, z_dim, heads=8, embedding_dim=128, is_training=True):
        self.vocab_size = vocab_size
        self.z_dim = z_dim
        self.heads = heads
        self.embedding_dim = embedding_dim
        self.is_training = is_training
        self.embedding_layer = keras.layers.Embedding(self.vocab_size, self.embedding_dim, mask_zero=True)
        self.get_mask = keras.layers.Lambda(lambda x: x._keras_mask, name="get_mask")
        self.to_one_hot = keras.layers.Lambda(to_one_hot, name="toonehot")
        self.scaleShift = ScaleShift()
        self.LN_layers = [keras.layers.LayerNormalization() for i in range(2)]
        self.Bi_LSTM_layers = [keras.layers.Bidirectional(keras.layers.LSTM(z_dim // 2, return_sequences=True)) for i in
                               range(2)]
        self.seq_maxpool = keras.layers.Lambda(seq_maxpool, name="maxpool")
        self.modulated_LN_layers = [SelfModulatedLayerNormalization(self.z_dim // 4, name="selfMLN_{}".format(i)) for i
                                    in range(3)]
        self.LSTM_layers = [keras.layers.LSTM(z_dim, return_sequences=True) for i in range(2)]
        self.MA = MultiHeadAttention(self.heads, self.z_dim // self.heads)
        self.concate = keras.layers.Concatenate()
        self.Dense_layers = [keras.layers.Dense(self.vocab_size) for i in range(2)]
        self.leakyrelu = keras.layers.LeakyReLU(0.2)
        self.combine = keras.layers.Lambda(lambda x: (x[0] + x[1]) / 2, name="com")
        self.softmax = keras.layers.Activation('softmax')
        self._model_init()
        if self.is_training:
            self._encoder_model()
            self._decoder_model()

    def _model_init(self):
        self.x_in = keras.Input(shape=(None,))
        self.y_in = keras.Input(shape=(None,))
        x, y = self.x_in, self.y_in
        x = self.embedding_layer(x)
        y = self.embedding_layer(y)
        self.x_mask = self.get_mask(x)
        x_one_hot = self.to_one_hot([self.x_in, self.x_mask])
        self.x_prior = self.scaleShift(x_one_hot)
        x = self.LN_layers[0](x)
        x = self.Bi_LSTM_layers[0](x)
        x = self.LN_layers[1](x)
        self.x_out = self.Bi_LSTM_layers[1](x)
        self.x_max = self.seq_maxpool([self.x_out, self.x_mask])

        y = self.modulated_LN_layers[0]([y, self.x_max])
        y = self.LSTM_layers[0](y)
        y = self.modulated_LN_layers[1]([y, self.x_max])
        y = self.LSTM_layers[1](y)
        y = self.modulated_LN_layers[2]([y, self.x_max])

        xy = self.MA([y, self.x_out, self.x_out, self.x_mask])
        xy = self.concate([y, xy])
        xy = self.Dense_layers[0](xy)
        xy = self.leakyrelu(xy)
        xy = self.Dense_layers[1](xy)
        xy = self.combine([xy, self.x_prior])
        self.out = self.softmax(xy)

        self.model = keras.Model([self.x_in, self.y_in], self.out, name="train_model")
        self.model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=selfcross_entropy)

    def _encoder_model(self):
        self.encoder_model = keras.Model(inputs=self.x_in, outputs=[self.x_out, self.x_mask, self.x_max, self.x_prior],
                                         name="encoder_model")

    def _decoder_model(self):
        enc_out = keras.Input(shape=(None, self.z_dim))
        out_mask = keras.Input(shape=(None,))
        out_max = keras.Input(shape=(self.z_dim,))
        out_prior = keras.Input(shape=(1, self.vocab_size))

        dy = self.embedding_layer(self.y_in)
        dy = self.modulated_LN_layers[0]([dy, out_max])
        dy = self.LSTM_layers[0](dy)
        dy = self.modulated_LN_layers[1]([dy, out_max])
        dy = self.LSTM_layers[1](dy)
        dy = self.modulated_LN_layers[2]([dy, out_max])

        dxy = self.MA([dy, enc_out, enc_out, out_mask])
        dxy = self.concate([dy, dxy])
        dxy = self.Dense_layers[0](dxy)
        dxy = self.leakyrelu(dxy)
        dxy = self.Dense_layers[1](dxy)
        dxy = self.combine([dxy, out_prior])
        self.dout = self.softmax(dxy)

        self.decoder_model = keras.Model(inputs=[self.y_in, enc_out, out_mask, out_max, out_prior], outputs=self.dout,
                                         name="decoder_model")
