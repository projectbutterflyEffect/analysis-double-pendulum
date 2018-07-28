from tensorflow.contrib import learn
from .lstm import _lstm_model

def lstm(train, validate):

    y = train.iloc[:,0]
    X = train.iloc[:,1:]

    LOG_DIR = './ops_logs'
    TIMESTEPS = 5
    RNN_LAYERS = [{'steps': TIMESTEPS}]
    DENSE_LAYERS = [10, 10]
    TRAINING_STEPS = 100000
    BATCH_SIZE = 100
    PRINT_STEPS = TRAINING_STEPS / 100

    regressor = learn.Estimator(model_fn=_lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS))

    validation_monitor = learn.monitors.ValidationMonitor(validate.iloc[:,0], validate.iloc[:,1:],
                                                          every_n_steps=PRINT_STEPS,
                                                          early_stopping_rounds=1000)

    regressor.fit(X, y, monitors=[validation_monitor], #logdir=LOG_DIR,
                                          # n_classes=0,
                                          # verbose=1,
                                          steps=TRAINING_STEPS,
                                          # optimizer='Adagrad',
                                          # learning_rate=0.03,
                                          batch_size=BATCH_SIZE)
