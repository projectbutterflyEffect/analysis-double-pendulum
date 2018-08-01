from tostools.numpy import rolling_window
from doit_xtended.linkedtasks import creates_files
import pathlib
import numpy as np

def _continuous_split(data, val_ratio=0.1, test_ratio=0.1):

    train_ratio = 1. - test_ratio - val_ratio
    assert train_ratio > 0.

    n = data.shape[0]

    n_val = int( n *val_ratio)
    n_test = int( n *test_ratio)
    n_train = n - n_val - n_test

    assert n > 0

    return dict(train = data.iloc[:n_train,:],
                validate = data.iloc[n_train:n_train+n_val,:],
                test = data.iloc[n_train+n_val:,:])

def _transpose_dataset(data, n_window=200):

    for k, val in data.items():
        data[k] = rolling_window(val.values, n_window)[:,:,::-1]

    return data

@creates_files('train.npy','validate.npy','test.npy')
def prepare_data(series1, targets):

    data = _continuous_split(series1.iloc[:,1:],test_ratio=0.15)
    data = _transpose_dataset(data, 40)

    for fname in targets:

        fname = pathlib.Path(fname)
        np.save(fname,data[fname.stem])



