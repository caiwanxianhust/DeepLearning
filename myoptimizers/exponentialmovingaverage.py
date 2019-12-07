import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


class ExponentialMovingAverage(object):
    """
    原文链接：https://spaces.ac.cn/archives/6575
    对模型权重进行指数滑动平均。
    用法：在model.compile之后、第一次训练之前使用；
    先初始化对象，然后执行inject方法。
    模型训练前：
        EMAer = ExponentialMovingAverage(model) # 在模型compile之后执行
        EMAer.inject() # 在模型compile之后执行
        model.fit(x_train, y_train) # 训练模型
    训练完成后：
        EMAer.apply_ema_weights() # 将EMA的权重应用到模型中
        model.predict(x_test) # 进行预测、验证、保存等操作
        EMAer.reset_old_weights() # 继续训练之前，要恢复模型旧权重。还是那句话，EMA不影响模型的优化轨迹。
        model.fit(x_train, y_train) # 继续训练

    现在翻看实现过程，可以发现主要的一点是引入了K.moving_average_update操作，并且插入到model.metrics_updates中，
    在训练过程中，模型会读取并执行model.metrics_updates的所有算子，从而完成了滑动平均。
    """

    def __init__(self, model, momentum=0.9999):
        self.momentum = momentum
        self.model = model
        self.ema_weights = [K.zeros(K.shape(w)) for w in model.weights]

    def initialize(self):
        """ema_weights初始化跟原模型初始化一致。
        """
        self.old_weights = K.batch_get_value(self.model.weights)
        K.batch_set_value(zip(self.ema_weights, self.old_weights))

    def inject(self):
        """添加更新算子到model.metrics_updates。
        """
        self.initialize()
        for w1, w2 in zip(self.ema_weights, self.model.weights):
            op = K.moving_average_update(w1, w2, self.momentum)
            self.model.metrics_updates.append(op)

    def apply_ema_weights(self):
        """备份原模型权重，然后将平均权重应用到模型上去。
        """
        self.old_weights = K.batch_get_value(self.model.weights)
        ema_weights = K.batch_get_value(self.ema_weights)
        K.batch_set_value(zip(self.model.weights, ema_weights))

    def reset_old_weights(self):
        """恢复模型到旧权重。
        """
        K.batch_set_value(zip(self.model.weights, self.old_weights))
