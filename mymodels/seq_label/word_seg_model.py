import tensorflow as tf
from tensorflow import keras
from sys import path

path.append(r'..')  # 将存放module的路径添加进来
from mylayers.crf import CRF
from mylayers.cnn import DGateReConv1D
import tensorflow.keras.backend as K
# from keras_contrib.layers import CRF

class CnnCrf(object):
    def __init__(self, vocab_size, em_size, nlabels, num_layers=3):
        self.embedding_layer = keras.layers.Embedding(vocab_size, em_size, mask_zero=True, name="embedding")
        self.get_mask = keras.layers.Lambda(lambda x: x._keras_mask, name="get_mask")
        self.dgcnn_layers = [DGateReConv1D(name="dgcnn_{}".format(_)) for _ in range(num_layers)]
        self.dense = keras.layers.Dense(nlabels)
        self.crf = CRF(True)  # 定义crf层，参数为True，自动mask掉最后一个标签
        self._model_init()

    def _model_init(self):
        x_in = keras.Input(shape=(None,), name="input")  # 建立输入层，输入长度设为None
        x = self.embedding_layer(x_in)
        x_mask = self.get_mask(x)
        for layer in self.dgcnn_layers:
            x = layer([x, x_mask])
        out = self.dense(x)  # 变成了5分类，第五个标签用来mask掉
        out = self.crf(out)  # 包装一下原来的tag_score
        self.model = keras.Model(inputs=x_in, outputs=out)
        self.model.compile(loss=self.crf.loss,  # 用crf自带的loss
                           optimizer='adam',
                           metrics=[self.crf.accuracy]  # 用crf自带的accuracy
                           )


class BiLSTMCRF(object):
    def __init__(self, vocab_size, em_size, nlabels, num_layers=3):
        self.embedding_layer = keras.layers.Embedding(vocab_size, em_size,
                                                      # mask_zero=True,
                                                      name="embedding")
        # self.get_mask = keras.layers.Lambda(lambda x: x._keras_mask, name="get_mask")
        self.bi_lstm_layers = [keras.layers.Bidirectional(keras.layers.LSTM(em_size, return_sequences=True)) for _ in
                               range(num_layers)]
        self.dense = keras.layers.Dense(nlabels)
        self.crf = CRF(True)  # 定义crf层，参数为True，自动mask掉最后一个标签
        # self.crf = CRF(nlabels, sparse_target=False)
        self._model_init()

    def _model_init(self):
        x_in = keras.Input(shape=(None,), name="input")  # 建立输入层，输入长度设为None
        x = self.embedding_layer(x_in)
        for layer in self.bi_lstm_layers:
            x = layer(x)
        out = self.dense(x)  # 变成了5分类，第五个标签用来mask掉
        out = self.crf(out)  # 包装一下原来的tag_score
        self.model = keras.Model(inputs=x_in, outputs=out)
        self.model.compile(loss=self.crf.loss,  # 用crf自带的loss
                           optimizer='adam',
                           metrics=[self.crf.accuracy]  # 用crf自带的accuracy
                           )
