from .doublependulum import DoublePendulum
import numpy as np
import pandas as pd
from doit_xtended.linkedtasks import creates_files

@creates_files('series1.pkl')
def create_timeseries(targets):

    pend = DoublePendulum(600, 1, 1, 3, 3, gravity=0.81, q1_0=np.pi/5, q2_0=np.pi/7, max_step=0.02)

    values = []
    while pend.status == 'running':

        values.append([pend.t]+pend.y.tolist())
        pend.step()

    data = pd.DataFrame(values, columns=['time','omega1', 'omega2', 'w1', 'w2'])

    data.to_pickle(targets[0])


