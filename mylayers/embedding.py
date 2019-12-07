import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


class MixEmbedding(keras.layers.Layer):
    def __init__(self, char_vocab_size, word_vocab_size, out_dim, word_vec_mat=None, **kwargs):
        super(MixEmbedding, self).__init__(**kwargs)
        self.out_dim = out_dim
        self.word_vec_mat = word_vec_mat
        self.char_vocab_size = char_vocab_size
        self.word_vocab_size = word_vocab_size

    def build(self, input_shape):
        super(MixEmbedding, self).build(input_shape)
        self.char_embedding = keras.layers.Embedding(self.char_vocab_size, self.out_dim)
        if self.word_vec_mat is None:
            self.word_embedding = keras.layers.Embedding(self.word_vocab_size, self.out_dim)
        else:
            assert self.word_vocab_size == self.word_vec_mat.shape[0], "word_vocab_size is incompatible with word_vec_mat"
            self.word_embedding = keras.layers.Embedding(self.word_vocab_size, self.word_vec_mat.shape[-1],
                                                         weights=[self.word_vec_mat], trainable=False)
        self.dense = keras.layers.Dense(self.out_dim, use_bias=False)

    def call(self, inp):
        char_id, word_id = inp
        char_em = self.char_embedding(char_id)
        word_em = self.word_embedding(word_id)
        word_em = self.dense(word_em)
        mix_em = char_em + word_em
        return mix_em

    def compute_output_shape(self, input_shape):
        return input_shape[0] + (self.out_dim,)
