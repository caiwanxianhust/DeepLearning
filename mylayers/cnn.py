import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

class DGateReConv1D(keras.layers.Layer):
    def __init__(self, rate=1, kernel_size=3, out_dim=None, residual=True, **kwargs):
        super(DGateReConv1D, self).__init__(**kwargs)
        self.out_dim = out_dim
        self.residual = residual
        self.rate = rate
        self.k_size = kernel_size

    def build(self, input_shape):
        if self.out_dim is None:
            self.out_dim = input_shape[0][-1]
        self.conv1d_1 = keras.layers.Conv1D(self.out_dim, self.k_size,
                                            padding="same", dilation_rate=self.rate)
        self.conv1d_2 = keras.layers.Conv1D(self.out_dim, self.k_size, dilation_rate=self.rate,
                                            padding="same", activation="sigmoid")
        if self.residual and self.out_dim != input_shape[0][-1]:
            self.conv1d_1_x = keras.layers.Conv1D(self.out_dim, 1, padding="same")


    def call(self, inp):
        x_init, mask = inp
        x = self.conv1d_1(x_init)
        g = self.conv1d_2(x_init)
        if self.residual:
            if self.out_dim != K.int_shape(x_init)[-1]:
                x_init = self.conv1d_1_x(x_init)
            out = x_init * (1-g) + x * g
        else:
            out = x * g
        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            for _ in range(K.ndim(x_init) - K.ndim(mask)):
                mask = mask[..., tf.newaxis]
            out = out * mask
        return out

    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (self.out_dim, )
