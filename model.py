import numpy as np
from scipy import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops.numpy_ops import np_config
from sklearn.utils import shuffle
import pickle
from layers import *


np_config.enable_numpy_behavior()


class SpreadLoss(keras.Loss):

    def __init__(self, margin=0.2):
        super().__init__(self)
        self.margin = margin

    def call(self, y_true, y_pred):
        return tf.math.maximum(0, self.margin - (y_true - y_pred))**2

    def adjust_margin(self, margin):
        self.margin = margin


class CapsNet(keras.Model):

    def __init__(self):
        super().__init__()
        self.graphconv1 = GraphConvolutionLayer(8)
        self.graphconv2 = GraphConvolutionLayer(8)
        self.graphconv3 = GraphConvolutionLayer(8)
        self.graphconv4 = GraphConvolutionLayer(8)
        self.stack = StackLayer()
        self.caps1 = CapsuleLayer(10, 4, transPOSE=True, shape_ignorant=True)
        self.caps2 = CapsuleLayer(8, 6)
        self.caps3 = CapsuleLayer(6, 8)
        self.optimizer = tf.optimizers.Adam(learning_rate=0.01)
        self.margin = 0.3


    def call(self, X):
        d = tf.reduce_sum(adjacency, axis=-1)
        act1 = d / tf.reduce_sum(d, axis=-1, keepdims=True)
        dm12 = tf.repeat(tf.reduce_sum(adjacency, axis=-1, keepdims=True), adjacency.shape[-1], axis=-1) ** tf.constant(
            -1 / 2)
        X1 = self.layers[0].call(X, adjacency, dm12)
        X2 = self.layers[1].call(X1, adjacency, dm12)
        # X3 = self.layers[2].call(X2, adjacency, dm12)
        # X4 = self.layers[3].call(X3, adjacency, dm12)
        caps1 = self.layers[2].call([X1, X2])
        act2, caps2 = self.layers[3].call(act1, caps1)
        act3, caps3 = self.layers[4].call(act2, caps2)
        # act4, caps4 = self.layers[7].call(act3, caps3)
        # print('shapes: ', act4.shape, caps4.shape)
        # print('xd')

        return act3


    def predict(self, X, adjacency):
        probas = self.predict_proba(X, adjacency)
        outputs = tf.one_hot(tf.argmax(probas, axis=-1), 2)
        return outputs







model = Model()
model.create_layers()

# with open('labels.pickle', 'rb') as f:
#     labels = pickle.load(f)
#
# with open('attributes.pickle', 'rb') as f:
#     attributes = pickle.load(f)
#
# with open('adjacency.pickle', 'rb') as f:
#     adjacency = pickle.load(f)

from colouring import get_graph_coloring_data
(train_X, train_adj, train_Y), test = get_graph_coloring_data()

model.fit(train_X, train_adj, train_Y, epochs=100, valid_data=test)
#print(model.predict(attributes, adjacency))

