import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np


class GCNN(keras.layers.Layer):
    def __init__(self, output_dim=None, k_size=3, residual=False, **kwargs):
        super(GCNN, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.residual = residual
        self.k_size = k_size

    def build(self, input_shape):
        # 如果不指定输出维度，则不变换维度
        if self.output_dim == None:
            self.output_dim = input_shape[-1]
        self.conv1d = keras.layers.Conv1D(self.output_dim * 2,
                                          self.k_size,
                                          padding='same',
                                          kernel_initializer='glorot_uniform')
        if self.residual and self.output_dim != input_shape[-1]:
            # 如果需要残差层同时输出维度和输入维度不一致，则需要对输入x进行线性变换，这里使用一维卷积
            self.conv1d_1x1 = keras.layers.Conv1D(self.o_dim, 1)

    def call(self, inputs):
        init_x = inputs
        x = self.conv1d(init_x)
        x = x[:, :, :self.output_dim] * K.sigmoid(x[:, :, self.output_dim:])
        if self.residual:
            if self.output_dim != K.int_shape(init_x)[-1]:
                init_x = self.conv1d_1x1(init_x)
            return init_x + x
        else:
            return x

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.o_dim,)


def sampling(args):
    z_mean, z_log_var = args
    latent_dim = K.int_shape(z_mean)[-1]
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0, stddev=1)
    return z_mean + K.exp(z_log_var / 2) * epsilon


def vae_loss(x):
    input_sentence, output, z_log_var, z_mean = x
    # [batch_size, 1]
    xent_loss = K.sum(K.sparse_categorical_crossentropy(input_sentence, output), 1)
    # [batch_size, 1]
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = xent_loss + kl_loss
    return vae_loss


class CnnVae(object):
    def __init__(self, n, vocab_size, latent_dim=64, hidden_dim=64, num_gcnn=2):
        # n言诗
        self.n = n
        # 字典大小
        self.vocab_size = vocab_size
        # 隐变量维度
        self.latent_dim = latent_dim
        # 隐层节点数
        self.hidden_dim = hidden_dim
        self.embedding_layer = keras.layers.Embedding(self.vocab_size, self.hidden_dim)
        self.gcnnlayers = [GCNN(residual=True, name="en_gcnn_{}".format(_)) for _ in range(num_gcnn)]
        self.pooling = keras.layers.GlobalAveragePooling1D()
        self.mean_dense = keras.layers.Dense(self.latent_dim, name="mean_dense")
        self.var_dense = keras.layers.Dense(self.latent_dim, name="var_dense")
        self.sample_layer = keras.layers.Lambda(sampling, name="sample_layer")

        self.decoder_hidden = keras.layers.Dense(hidden_dim * 2 * n)
        self.decoder_cnn = GCNN(residual=True, name="de_gcnn")
        self.decoder_dense = keras.layers.Dense(self.vocab_size, activation='softmax', name="out")

        self.loss_layer = keras.layers.Lambda(vae_loss)
        self._init_model()
        self._gen_model()

    def _init_model(self):
        # [batch_size, seq_len]
        self.input_sentence = keras.Input(shape=(2*self.n,), name="inp_seq")
        # [bacth_size, seq_len, hidden_dim]
        input_vec = self.embedding_layer(self.input_sentence)  # id转向量

        # [bacth_size, seq_len, hidden_dim]
        h = self.gcnnlayers[0](input_vec)  # GCNN层
        h = self.gcnnlayers[1](h)  # GCNN层
        # [bacth_size, hidden_dim]
        h = self.pooling(h)  # 池化
        # 算均值方差
        # [bacth_size, latent_dim]
        z_mean = self.mean_dense(h)
        z_log_var = self.var_dense(h)
        # [bacth_size, latent_dim]
        z = self.sample_layer([z_mean, z_log_var])
        # [bacth_size, hidden_dim * 2 * n]
        h = self.decoder_hidden(z)
        # [bacth_size, 2 * n, hidden_dim]
        h = keras.layers.Reshape((2 * self.n, self.hidden_dim), name="reshape")(h)
        # [bacth_size, 2 * n, hidden_dim]
        h = self.decoder_cnn(h)
        # [bacth_size, 2 * n, self.vocab_size]
        output = self.decoder_dense(h)
        self.loss = self.loss_layer([self.input_sentence, output, z_log_var, z_mean])
        # 建立模型
        self.vae = keras.Model(self.input_sentence, self.loss)
        self.vae.compile(loss=lambda y_true, loss: loss,
                         optimizer='adam')

    def _gen_model(self):
        decoder_input = keras.Input(shape=(self.latent_dim,),name="dec_inp")
        _ = self.decoder_hidden(decoder_input)
        _ = keras.layers.Reshape((2 * self.n, self.hidden_dim), name="reshape")(_)
        _ = self.decoder_cnn(_)
        _output = self.decoder_dense(_)
        self.genmodel = keras.Model(decoder_input, _output)


# 利用生成模型随机生成一首诗
def gen(gen_model, latent_dim, id2char, n):
    # [bacth_size, 2 * n, self.vocab_size]
    r = gen_model.predict(np.random.randn(1, latent_dim))[0]
    # [2 * n]
    r = r.argmax(axis=-1)
    return ''.join([id2char[i] for i in r[:n]]) + '，' + ''.join([id2char[i] for i in r[n:]])


# 回调器，方便在训练过程中输出
class Evaluate(keras.callbacks.Callback):
    def __init__(self, gen_model, latent_dim, id2char, n):
        self.log = []
        self.gen_model = gen_model
        self.latent_dim = latent_dim
        self.id2char = id2char
        self.n = n

    def on_epoch_end(self, epoch, logs=None):
        self.log.append(gen(self.gen_model, self.latent_dim, self.id2char, self.n))
        print("epoch: {}, sent: {}".format(epoch, self.log[-1]))
