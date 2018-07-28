from tostools.numpy import rolling_window
from doit_xtended.linkedtasks import creates_files
import pathlib
import pandas as pd

def _continuous_split(data, val_ratio=0.1, test_ratio=0.1):

    train_ratio = 1. - test_ratio - val_ratio
    assert train_ratio > 0.

    n = data.shape[0]

    n_val = int( n *val_ratio)
    n_test = int( n *test_ratio)
    n_train = n - n_val - n_test

    assert n > 0

    return dict(train = data.iloc[:n_train],
                validate = data.iloc[n_train:n_train+n_val],
                test = data.iloc[n_train+n_val:])

def _transpose_dataset(data, n_window=200):

    for k, val in data.items():
        data[k] = pd.DataFrame(rolling_window(val.values, n_window))

    return data

@creates_files('train.pkl','validate.pkl','test.pkl')
def prepare_data(series1, targets):

    data = _continuous_split(series1['w1'])
    data = _transpose_dataset(data, 50)

    for fname in targets:

        fname = pathlib.Path(fname)
        data[fname.stem].to_pickle(fname)



