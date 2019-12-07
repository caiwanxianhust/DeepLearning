import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
import numpy as np
from sys import path

path.append(r'..')  # 将存放module的路径添加进来
from mylayers.attention import MultiHeadAttention


class PositionalEncoding(keras.layers.Layer):
    def __init__(self, maximum_position, latent_dim, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.maximum_position = maximum_position
        self.latent_dim = latent_dim
        position = np.arange(self.maximum_position).reshape((self.maximum_position, 1))
        d_model = np.arange(self.latent_dim).reshape((1, self.latent_dim))
        angle_rates = 1 / np.power(10000, (2 * (d_model // 2)) / np.float32(self.latent_dim))
        self.angle_rads = position * angle_rates
        # 将 sin 应用于数组中的偶数索引（indices）；2i
        self.angle_rads[:, 0::2] = np.sin(self.angle_rads[:, 0::2])
        # 将 cos 应用于数组中的奇数索引；2i+1
        self.angle_rads[:, 1::2] = np.cos(self.angle_rads[:, 1::2])
        # (1, maximum_position, latent_dim)
        self.pos_encoding = tf.cast(self.angle_rads[np.newaxis, ...], dtype=tf.float32)

    def build(self, input_shape):
        super(PositionalEncoding, self).build(input_shape)
        # [max_len, 1]
        self.seq_len = input_shape[1]

    def call(self, x):
        return self.pos_encoding[:, :self.seq_len, :]

    def compute_output_shape(self, input_shape):
        return (1, input_shape[1], input_shape[2])


def point_wise_feed_forward_network(d_model, dff=512):
    return tf.keras.Sequential([
        keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class FeedForward(keras.layers.Layer):
    def __init__(self, d_model, dff=512, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.d_model = d_model
        self.dense_1 = keras.layers.Dense(dff, activation='relu')
        self.dense_2 = keras.layers.Dense(d_model)

    def call(self, inp):
        out = self.dense_1(inp)
        out = self.dense_2(out)
        return out

    def compute_output_shape(self, input_shape):
        return input_shape[:2] + (self.d_model,)


class Encoderlayer(keras.layers.Layer):
    def __init__(self, latent_dim, heads, rate=0.1, training=True, **kwargs):
        super(Encoderlayer, self).__init__(**kwargs)
        self.training = training
        self.dropout_1 = keras.layers.Dropout(rate)
        self.dropout_2 = keras.layers.Dropout(rate)
        self.mha = MultiHeadAttention(latent_dim, heads)
        self.ffn = FeedForward(latent_dim)
        self.layernorm_1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_2 = keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inp):
        x, mask = inp
        attn_out = self.mha([x, x, x, mask])
        attn_out = self.dropout_1(attn_out, training=self.training)
        res_1 = keras.layers.Lambda(lambda x: x[0] + x[1])([x, attn_out])
        out1 = self.layernorm_1(res_1)
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout_2(ffn_output, training=self.training)
        res_2 = keras.layers.Lambda(lambda x: x[0] + x[1])([out1, ffn_output])
        out2 = self.layernorm_2(res_2)
        return out2

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, latent_dim, heads, rate=0.1, training=True, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.training = training
        self.mha_1 = MultiHeadAttention(latent_dim, heads, mask_right=True)
        self.mha_2 = MultiHeadAttention(latent_dim, heads)
        self.layernorm_1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_3 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = FeedForward(latent_dim)
        self.dropout_1 = keras.layers.Dropout(rate)
        self.dropout_2 = keras.layers.Dropout(rate)
        self.dropout_3 = keras.layers.Dropout(rate)

    def call(self, inp):
        x, enc_out, mask = inp
        attn_out_1 = self.mha_1([x, x, x, mask])
        attn_out_1 = self.dropout_1(attn_out_1, training=self.training)
        res_1 = keras.layers.Lambda(lambda x: x[0] + x[1])([x, attn_out_1])
        out1 = self.layernorm_1(res_1)

        attn_out_2 = self.mha_2([out1, enc_out, enc_out])
        attn_out_2 = self.dropout_2(attn_out_2, training=self.training)
        res_2 = keras.layers.Lambda(lambda x: x[0] + x[1])([out1, attn_out_2])
        out2 = self.layernorm_2(res_2)

        ffn_out = self.ffn(out2)
        ffn_out = self.dropout_3(ffn_out)
        res_3 = keras.layers.Lambda(lambda x: x[0] + x[1])([out2, ffn_out])
        out3 = self.layernorm_3(res_3)
        return out3

    def compute_output_shape(self, input_shape):
        return input_shape[0]


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(real, pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    return loss


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class Transformer(object):
    def __init__(self, inp_vocab_size, tar_vocab_size, em_dim, maximum_position, num_layers, heads, rate=0.1,
                 training=True):
        self.num_layers = num_layers
        self.training = training
        self.rate = rate
        self.em_dim = em_dim
        self.embedding_x = keras.layers.Embedding(inp_vocab_size, em_dim,
                                                  mask_zero=True, name="em_x")
        self.embedding_y = keras.layers.Embedding(tar_vocab_size, em_dim,
                                                  mask_zero=True, name="em_y")
        self.pos_encoding = PositionalEncoding(maximum_position, em_dim)
        self.res_layer = keras.layers.Lambda(lambda x: x[0] + x[1], name="res")
        self.get_mask = keras.layers.Lambda(lambda x: x._keras_mask, name="get_mask")
        self.dropout_1 = keras.layers.Dropout(rate)
        self.enc_layers = [Encoderlayer(em_dim, heads, training=self.training) for _ in range(self.num_layers)]
        self.dropout_2 = keras.layers.Dropout(rate)
        self.dec_layers = [DecoderLayer(em_dim, heads, training=self.training) for _ in range(self.num_layers)]
        self.final_layer = keras.layers.Dense(tar_vocab_size)

        self._model_init()
        self._encoder()
        self._decoder()

    def _model_init(self):
        self.x_in = keras.Input(shape=(None,), name="x_in")
        self.y_in = keras.Input(shape=(None,), name="y_in")

        x = self.embedding_x(self.x_in)
        x_mask = self.get_mask(x)
        x_pos_encoding = self.pos_encoding(x)
        x = self.res_layer([x, x_pos_encoding])
        self.enc_out = self.dropout_1(x, training=self.training)
        # Encorder
        for layer in self.enc_layers:
            self.enc_out = layer([self.enc_out, x_mask])

        y = self.embedding_y(self.y_in)
        y_mask = self.get_mask(y)
        y_pos_encoding = self.pos_encoding(y)
        y = self.res_layer([y, y_pos_encoding])
        self.dec_out = self.dropout_2(y, training=self.training)
        # Decoder
        for layer in self.dec_layers:
            self.dec_out = layer([self.dec_out, self.enc_out, y_mask])
        self.dec_out = self.final_layer(self.dec_out)

        self.transformer = keras.Model([self.x_in, self.y_in], self.dec_out, name="transformer")
        learning_rate = CustomSchedule(self.em_dim)
        self.transformer.compile(loss=loss_function,
                                 optimizer=keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9),
                                 metrics=[keras.metrics.SparseCategoricalAccuracy(name='train_accuracy'), ])

    def _encoder(self):
        self.encoder_model = keras.Model(self.x_in, self.enc_out, name="encoder")

    def _decoder(self):
        self.enc_out_inp = keras.Input(shape=(None, None), name="enc_out_inp")
        y = self.embedding_y(self.y_in)
        y_mask = self.get_mask(y)
        y_pos_encoding = self.pos_encoding(y)
        y = self.res_layer([y, y_pos_encoding])
        self.dec_out = self.dropout_2(y, training=self.training)
        # Decoder
        for layer in self.dec_layers:
            self.dec_out = layer([self.dec_out, self.enc_out_inp, y_mask])
        self.dec_out = self.final_layer(self.dec_out)
        self.decoder_model = keras.Model([self.enc_out_inp, self.y_in], self.dec_out, name="decoder")
