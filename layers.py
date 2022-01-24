import numpy as np
from scipy import *
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.python.ops.numpy_ops import np_config
from sklearn.utils import shuffle
import math
from tensorflow_probability import distributions as tfd


class GraphConvolutionLayer(keras.Layer):
    def __init__(self, attributes_out, **kwargs):
        self.attributes_out = attributes_out
        super(GraphConvolutionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.weights = self.add_weight(name='weights',
                                       shape=[input_shape[0][-1], self.attributes_out],
                                       initializer='random_normal')
        super(GraphConvolutionLayer, self).build(input_shape)

    def call(self, act_in, adjacency, dm12):
        return tf.nn.leaky_relu(dm12 * adjacency * dm12 @ act_in @ self.weights, alpha=0.2)


class StackLayer(keras.Layer):
    def __init__(self, **kwargs):
        super(StackLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(StackLayer, self).build(input_shape)

    def call(self, t):
        return tf.stack(t, axis=2)


class CapsuleLayer(keras.Layer):
    def __init__(self, capsules_out, caps_shape_out, transPOSE=False, n_iters=3, lamb=1, shape_ignorant=False,
                 **kwargs):
        self.capsules_out = capsules_out
        self.caps_shape_out = caps_shape_out
        self.transPOSE = transPOSE
        self.n_iters = n_iters
        self.lamb = lamb
        self.shape_ignorant = shape_ignorant
        super(CapsuleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.shape_ignorant:
            self.capsules_in = 1
        else:
            self.capsules_in = input_shape[0][0]
        if self.transPOSE:
            input_shape[1][-2, -1] = input_shape[1][-1, -2]
        self.transitions = self.add_weight(name='transitions',
                                           shape=[self.capsules_in, input_shape[1][-1], self.attributes_out],
                                           initializer='random_normal')

        self.bias_a = self.add_weight(name='bias_a',
                                      shape=self.capsules_out,
                                      initializer='random_normal')

        self.bias_b = self.add_weight(name='bias_b',
                                      shape=self.capsules_out,
                                      initializer='random_normal')


    def call(self, act, pose):
        return self.EM_routing(act, pose)

    def EM_routing(self, act_in, X):
        if self.transPOSE:
            X = X.transpose([0, 1, 3, 2])
        c1, c2 = X.shape[2], X.shape[3]
        c3 = self.transitions.shape[3]
        # this estimate of R is an initial E-step
        R = tf.Variable(np.ones((1, self.n_capsules_in, self.n_capsules_out)) / self.n_capsules_out)
        V = self.voting(X)
        act_out, mu, sigma = self.M(act_in, R, V)
        for _ in range(self.n_iters - 1):
            R = self.E(act_out, V, mu, sigma)
            act_out, mu, sigma = self.M(act_in, R, V)
        return act_out, mu.reshape(X.shape[0], self.n_capsules_out, c1, c3)

    def voting(self, X):
        return tf.matmul(tf.cast(X[:, :, tf.newaxis, :, :], tf.float32), self.transitions)

    def E(self, act_out, V, mu, sigma):
        distribution = tfd.Normal(mu, sigma)
        prob = distribution.prob(V)
        raw_R = act_out[:, tf.newaxis, :, tf.newaxis, tf.newaxis] * prob
        return tf.reduce_sum(tf.math.divide_no_nan(raw_R, tf.linalg.norm(raw_R, ord=1, keepdims=True)), axis=[-2, -1])

    def M(self, act_in, R, V):
        odds = act_in[:, :, tf.newaxis] * R
        r = tf.math.divide_no_nan(odds, np.sum(odds, axis=1, keepdims=True))[:, :, :, np.newaxis, np.newaxis]
        mu = tf.math.reduce_sum(r * V, axis=1, keepdims=True)
        sigma = tf.math.sqrt(tf.reduce_sum(r * (V - mu) ** 2, axis=1, keepdims=True))
        cost1 = r * tf.math.log(sigma)
        cost2 = tf.constant(1 / 2 + tf.math.log(2 * math.pi) / 2)
        cost = tf.math.reduce_sum(cost1 + cost2, axis=1)
        '''cost = tf.math.reduce_sum(
            r * tf.math.log(sigma) + tf.constant(1 / 2 + tf.math.log(2 * math.pi) / 2),
            axis=1)'''
        ac1 = self.lamb
        ac2 = tf.math.reduce_sum(r, axis=[1, -2, -1])
        ac3 = tf.math.reduce_sum(cost, axis=[2, 3])
        ac4 = self.bias_a - self.bias_b * ac2 - ac3
        act_out = tf.nn.softmax(ac1 * ac4)
        # act_out = tf.math.sigmoid(ac1 * ac4)
        '''
        act_out = tf.math.sigmoid(self.lamb * (self.bias_a - self.bias_b *
                                          tf.math.reduce_sum(r, axis=1) -
                                          tf.math.reduce_sum(cost, axis=[2, 3])))
        '''
        return act_out, mu, sigma


# dynamic routing

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

