from model import *

class CapsNetB(keras.Model):

    def __init__(self):
        super().__init__()
        self.graphconv1 = GraphConvolutionLayer(4)
        self.graphconv2 = GraphConvolutionLayer(4)
        self.graphconv3 = GraphConvolutionLayer(4)
        self.graphconv4 = GraphConvolutionLayer(4)
        self.stack = StackLayer()
        self.caps1 = CapsuleLayer(20, 4, shape_ignorant=True)
        self.caps2 = CapsuleLayer(10, 4)
        self.caps3 = CapsuleLayer(4, 6)
        self.caps4 = CapsuleLayer(1, 6)
        self.batch_norm = BatchNormalization()
        self.flatten = Flatten()
        self.dense1 = Dense(60, kernel_regularizer=l1andl2(5e-6, 5e-6))
        self.dense1 = Dense(40, kernel_regularizer=l1andl2(5e-6, 5e-6))
        self.dense2 = Dense(6, activation='softmax', kernel_regularizer=l1andl2(5e-6, 5e-6))


    def call(self, inputs):
        attributes = inputs[0]
        adjacency = inputs[1]
        d = tf.reduce_sum(adjacency, axis=-1)
        dm12 = tf.repeat(tf.reduce_sum(adjacency, axis=-1, keepdims=True), adjacency.shape[-1], axis=-1)
        X1 = self.graphconv1([attributes, adjacency], dm12)
        X2 = self.graphconv2([X1, adjacency], dm12)
        X3 = self.graphconv3([X2, adjacency], dm12)
        X4 = self.graphconv4([X3, adjacency], dm12)
        act1 = d / tf.reduce_sum(d, axis=-1, keepdims=True)
        caps1 = self.stack([X1, X2, X3, X4])
        act2, caps2 = self.caps1([act1, caps1])
        act3, caps3 = self.caps2([act2, caps2])
        act4, caps4 = self.caps3([act3, caps3])
        act5, caps5 = self.caps4([act4, caps4])
        caps_out = self.flatten(caps5)
        caps_out = self.batch_norm(caps_out)
        dense1 = self.dense1(caps_out)
        dense2 = self.dense2(dense1)
        return dense2
