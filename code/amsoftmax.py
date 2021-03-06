import tensorflow as tf
import math
from keras.initializers import RandomNormal
from keras import backend as K
from keras.engine import InputSpec
from keras.layers import activations, initializers, regularizers, constraints, Lambda
from keras.engine.topology import Layer


# def recallll(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall
#
#
# def precisionnnn(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision


class AMSoftmax(Layer):
    def __init__(self, units=50, s=30, m=0.35,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs
                 ):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(AMSoftmax, self).__init__(**kwargs)
        self.units = units
        self.s = s
        self.m = m
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.bias = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, **kwargs):
        inputs = tf.nn.l2_normalize(inputs, dim=-1)
        self.kernel = tf.nn.l2_normalize(self.kernel, dim=(0, 1))  # W归一化

        dis_cosin = K.dot(inputs, self.kernel)
        psi = dis_cosin - self.m

        e_costheta = K.exp(self.s * dis_cosin)
        e_psi = K.exp(self.s * psi)
        sum_x = K.sum(e_costheta, axis=-1, keepdims=True)

        temp = e_psi - e_costheta
        temp = temp + sum_x

        output = e_psi / temp
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units

    def get_config(self):
        config = super(AMSoftmax, self).get_config()
        config.update({"units": self.units})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def amsoftmax_loss(y_true, y_pred):
    d1 = K.sum(y_true * y_pred, axis=-1)
    d1 = K.log(K.clip(d1, K.epsilon(), None))
    loss = -K.mean(d1, axis=-1)
    return loss
