import keras

import numpy as np
from collections import deque
from doit_xtended.linkedtasks import creates_files

@creates_files('prediction.npy')
def prediction(test, file_lstm, targets):

    model = keras.models.load_model(file_lstm)

    X = test[0,:,1:][:,::-1]

    data = deque(X.transpose(),maxlen=X.shape[1])
    ypreds = []
    for i in range(test.shape[0]):
        X = np.reshape(data, (1,1,np.prod(X.shape)))
        ypred = model.predict(X)
        ypreds.append(ypred[0])
        data.appendleft(ypred[0])

    test[:,:,0] = ypreds
    np.save(targets[0],test)