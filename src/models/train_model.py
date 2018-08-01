from keras.models import Sequential
from keras.layers import Dense, LSTM

import numpy as np


def _reshape_data(data):
    y, X = data[:,:,0], data[:,:,1:][:,:,::-1]
    X = np.reshape(X, (X.shape[0],1,np.prod(X.shape[1:])))
    return X, y

def lstm(train, validate, targets):

    X, y = _reshape_data(train)
    Xval, yval = _reshape_data(validate)

    model = Sequential()

    model.add(LSTM(40, input_shape=(None, X.shape[-1])))
    model.add(Dense(4))
    model.compile(loss='mean_absolute_error', optimizer='adam')

    # x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
    model.fit(X, y, epochs=30, batch_size=20,
              validation_data=(Xval,yval))

    model.save(targets[0], overwrite=True)

