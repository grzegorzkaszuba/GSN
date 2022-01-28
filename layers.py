import numpy as np
import tensorflow as tf
from keras.losses import Loss
import math
from keras.layers import Layer
from tensorflow.keras.regularizers import l1_l2 as l1andl2
from tensorflow_probability import distributions as tfd


class GraphConvolutionLayer(Layer):
    def __init__(self, attributes_out, **kwargs):
        self.attributes_out = attributes_out
        super(GraphConvolutionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.conv_weights = self.add_weight(name='weights',
                                       shape=[input_shape[0][-1], self.attributes_out],
                                       initializer='random_normal',
                                        regularizer=l1andl2(l1=1e-5, l2=1e-5))
        super(GraphConvolutionLayer, self).build(input_shape)

    def call(self, inputs, dm12):
        act_in, adjacency = inputs[0], inputs[1]
        return tf.nn.leaky_relu(adjacency @ act_in @ self.conv_weights, alpha=0.2)


class StackLayer(Layer):
    def __init__(self, **kwargs):
        super(StackLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(StackLayer, self).build(input_shape)

    def call(self, t):
        return tf.stack(t, axis=-1)


class CapsuleLayer(Layer):
    def __init__(self, capsules_out, caps_shape_out, transPOSE=True, n_iters=3, lamb=0.2, shape_ignorant=False,
                 **kwargs):
        self.capsules_out = capsules_out
        self.caps_shape_out = caps_shape_out
        self.transPOSE = transPOSE
        self.iters = n_iters
        self.lamb = lamb
        self.shape_ignorant = shape_ignorant
        super(CapsuleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.shape_ignorant:
            self.capsules_in = 1
        else:
            self.capsules_in = input_shape[0][-1]
        if self.shape_ignorant:
            if self.transPOSE:
                shape = [self.capsules_out, input_shape[1][-2], self.caps_shape_out]
            else:
                shape = [self.capsules_out, input_shape[1][-1], self.caps_shape_out]
        else:
            if self.transPOSE:
                shape = [self.capsules_in, self.capsules_out, input_shape[1][-2], self.caps_shape_out]
            else:
                shape = [self.capsules_in, self.capsules_out, input_shape[1][-1], self.caps_shape_out]
        self.transitions = self.add_weight(name='transitions',
                                           shape=shape,
                                           initializer='random_normal',
                                           regularizer=l1andl2(l1=3e-5, l2=3e-5))

        self.bias_a = self.add_weight(name='bias_a',
                                      shape=[self.capsules_out],
                                      initializer='random_normal',
                                      regularizer=l1andl2(l1=3e-5, l2=3e-5))

        self.bias_b = self.add_weight(name='bias_b',
                                      shape=[self.capsules_out],
                                      initializer='random_normal',
                                      regularizer=l1andl2(l1=3e-5, l2=3e-5))
        super(CapsuleLayer, self).build(input_shape)

    def call(self, inputs):
        act = inputs[0]
        pose = inputs[1]
        if self.transPOSE:
            pose = tf.linalg.matrix_transpose(pose)
        R = tf.constant(np.ones((1, self.capsules_in, self.capsules_out)) / self.capsules_out)
        V = pose[:, :, tf.newaxis, :, :] @ self.transitions
        #y_true = K.print_tensor(V.shape, message='V = ')
        #t_shape = K.print_tensor(self.transitions.shape, message='transition shape = ')
        act_out, mu, sigma = self.M(act, R, V)
        for i in range(self.iters-1):
            R = self.E(act_out, V, mu, sigma)
            act_out, mu, sigma = self.M(act, R, V)
        return act_out, mu

    def E(self, act_out, V, mu, sigma):
        distribution = tfd.Normal(mu[:, np.newaxis], sigma)
        prob = distribution.prob(V)
        raw_R = act_out[:, tf.newaxis, :, tf.newaxis, tf.newaxis] * prob
        return tf.reduce_sum(tf.math.divide_no_nan(raw_R, tf.linalg.norm(raw_R, ord=1, keepdims=True)), axis=[-2, -1])

    def M(self, act_in, R, V):
        odds = act_in[:, :, tf.newaxis] * R
        r = tf.math.divide_no_nan(odds, tf.reduce_sum(odds, axis=1, keepdims=True))[:, :, :, np.newaxis, np.newaxis]
        mu = tf.math.reduce_sum(r * V, axis=1, keepdims=True)
        sigma = tf.math.sqrt(tf.reduce_sum(r * (V - mu) ** 2, axis=1, keepdims=True))
        cost = tf.math.reduce_sum(r * tf.math.log(sigma) + (1 / 2 + tf.math.log(2 * math.pi) / 2),
                                  axis=1)
        act_out = tf.math.softmax(self.lamb * (self.bias_a - self.bias_b *
                                               tf.math.reduce_sum(r, axis=[1, -2, -1]) -
                                               tf.math.reduce_sum(cost, axis=[2, 3])), axis=-1)
        return act_out, tf.reduce_sum(mu, axis=1), sigma


class SpreadLoss(Loss):

    def __init__(self, margin=0.2):
        super().__init__(self)
        self.margin = margin

    def call(self, y_true, y_pred):
        return tf.math.maximum(0, self.margin - (y_true - y_pred))**2

    def adjust_margin(self, margin):
        self.margin = margin




# EM routing

# B - batch size
# p - # of capsules in previous layer, n - # of capsules in current layer
# c1, c2, c3 - capsule shapes of previous (1, 2) and next (1, 3) layer
# likely c2 = c3; alternatively c3 > c2 if n << p: if so, c2 has to be the dimension that's extended, so ones has to transpose the pose

# variable      dimensionality      description                                                 source
# in_caps       1                   # of capsules passed from previous layer                    architecture
# out_caps      1                   # of capsules in the current layer                          architecture
# n_iters       1                   # of iterations                                             architecture
# transPOSE     1; bool             # switch the dimensions of the pose matrix (pun intended)   archutecture (required to extend the right dimension)
# act_in        B x p               activation of caps_in                                       feed forward
# X             B x p x c1 x c2     caps_in pose matrices                                       feed forward
# transitions   p x n x c2 x c3     transitions of pose matrices from one layer to another      learned
# bias_a        p                   bias                                                        learned
# bias_b        p                   bias                                                        learned
# lamb          1                   gaussian mixture parameter                                  hyperparameter
# R             B x p x n           posterior likelihood a caps_in corresponds to a caps_out    E-step: calculated from gaussian distribution based on assumed pose of a single caps_out, initialized as uniform distribution before first estimating those poses
# V             B x p x n x c1 x c3 pose matrices of caps_out if it corresponds to caps_in      calculated by vote method
# r             B x p x n           weight of caps_in pose for inferring caps_out pose          M-step
# ...r                              that means normalized prior likelihood that a caps_in exists AND corresponds to a caps_out
# mu            B x n  x c1 x c3     expected pose of caps_out                                  M-step
# sigma         B x n x c1 x c3     standard deviation of pose of caps_out                      M-step
# act_out       B x n               activation of caps_out                                      M-step

# cost          B x p * c1 * c3     cost function of the gaussian distribution                  M-step

# R is inferred from pose matrices
# r is inferred from pose matrices and activation

