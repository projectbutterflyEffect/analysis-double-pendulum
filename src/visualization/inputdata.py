from quickplot import QuickPlot
import numpy as np

def inputdata(train, test, validate, targets):

    n = len(train) + len(validate) + len(test)

    t = np.arange(n)
    with QuickPlot() as qp:

        qp.plot(t[:len(train)],train.iloc[:,0])
        qp.plot(t[len(train):len(train)+len(validate)],validate.iloc[:,0])
        qp.plot(t[len(train)+len(validate):],test.iloc[:,0])

    qp.fig.save(targets[0], dpi=300)