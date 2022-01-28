from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
from keras.layers import Dense, Flatten, BatchNormalization
from tensorflow.python.ops.numpy_ops import np_config

from layers import *


np_config.enable_numpy_behavior()


class CapsNet(keras.Model):

    def __init__(self):
        super().__init__()
        self.graphconv1 = GraphConvolutionLayer(8)
        self.graphconv2 = GraphConvolutionLayer(8)
        self.graphconv3 = GraphConvolutionLayer(8)
        self.graphconv4 = GraphConvolutionLayer(8)
        self.stack = StackLayer()
        self.caps1 = CapsuleLayer(20, 4, shape_ignorant=True)
        self.caps2 = CapsuleLayer(12, 6)
        self.caps3 = CapsuleLayer(6, 8)


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
        #pred = K.print_tensor(act4, message='y_pred = ')
        return act4




with open('labels.pickle', 'rb') as f:
    labels = pickle.load(f)

with open('attributes.pickle', 'rb') as f:
    attributes = tf.stack(pickle.load(f), axis=0)

with open('adjacency.pickle', 'rb') as f:
    adjacency = tf.stack(pickle.load(f), axis=0)


scaler = MinMaxScaler()
attributes = attributes.reshape(-1, 18)
attributes = scaler.fit_transform(attributes)
attributes = attributes.reshape(-1, 126, 18)


X_train, X_test, adj_train, adj_test, Y_train, Y_test = train_test_split(attributes, adjacency, labels, train_size=0.8, random_state=1)

X_data = {"attributes": X_train,
        "adjacency": adj_train}

Y_data = {"labels": Y_train}


if __name__ == '__main__':
    model = CapsNet()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=4*1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy',
                           'mse',
                           'categorical_crossentropy'])


    earlyStopping = EarlyStopping(monitor='val_categorical_crossentropy', patience=200, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('.saved_model', save_format="tf", save_best_only=True, monitor='val_accuracy', mode='max')
    callbacks = [earlyStopping, mcp_save, mcp_save2]
    model.fit([X_train, adj_train], Y_train, batch_size=32, epochs=2000, validation_split=0.2, callbacks=callbacks)
    model.summary()

