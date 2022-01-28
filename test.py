from model import *


print('Test set results: ')
model = keras.models.load_model('model A/.mdl.model')
model.evaluate([X_test, adj_test], Y_test)

