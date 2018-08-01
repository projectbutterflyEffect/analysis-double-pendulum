from quickplot import QuickPlot
import numpy as np

def inputdata(train, test, validate, targets):

    n = len(train) + len(validate) + len(test)

    t = np.arange(n)
    with QuickPlot() as qp:
        qp.plot(t[:len(train)],train[:,:,0])
        qp.plot(t[len(train):len(train)+len(validate)],validate[:,:,0])
        qp.plot(t[len(train)+len(validate):],test[:,:,0])

    qp.fig.savefig(targets[0], dpi=120)

def modelresult(train, test, validate, prediction, targets):

    n = len(train) + len(validate) + len(test)

    t = np.arange(n)
    with QuickPlot() as qp:
        qp.plot(t[:len(train)],train[:,:,0])
        qp.plot(t[len(train):len(train)+len(validate)],validate[:,:,0])
        qp.plot(t[len(train)+len(validate):],test[:,:,0])
        qp.plot(t[len(train)+len(validate):],prediction[:,:,0], marker='o')

    qp.fig.savefig(targets[0], dpi=120)