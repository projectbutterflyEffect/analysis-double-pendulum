import scipy as sp
import scipy.integrate
import sympy as sym
from sympy.utilities.autowrap import ufuncify
import numpy as np

from .formulas import _create_symbols


class DoublePendulum(sp.integrate.RK45):
    def __init__(self,
                 t_bound,
                 length1,
                 length2,
                 mass1,
                 mass2,
                 gravity=10,
                 q1_0=0, q2_0=0, dq1dt_0=0, dq2dt_0=0,
                 max_step=1):

        x1, x2, y1, y2, m1, m2, l1, l2, t1, q1, t2, q2, eom, H, p1, p2, g, B = _create_symbols()

        # reduce with symmetric length and mass and set the parameters

        # coordinates
        self.coords_expr = sym.Matrix([x1, y1, x2, y2]).subs(
            {m1: mass1, m2: mass2, l1: length1, l2: length2, t1: q1, t2: q2})
        self.coords_num = [ufuncify([q1, q2], e) for e in self.coords_expr]

        # self.H_expr = H.subs({m1:mass1,m2:mass2,l1:length1,l2:length2,g:gravity}).simplify()
        # self.B_expr = B.subs({m1:mass1,m2:mass2,l1:length1,l2:length2})

        self.eom = eom.subs({m1: mass1, m2: mass2, l1: length1, l2: length2, g: gravity})
        self.H = H.subs({m1: mass1, m2: mass2, l1: length1, l2: length2, g: gravity})

        self.eom_num = [ufuncify([q1, q2, p1, p2], e) for e in self.eom]
        self.H_num = ufuncify([q1, q2, p1, p2], self.H)

        # calculate initial generalized momentum

        self.PQ = B.subs({m1: mass1, m2: mass2, l1: length1, l2: length2, g: gravity})
        self.invPQ = self.PQ.inv() * sym.Matrix([[p1], [p2]])
        self.invPQ_num = [ufuncify([q1, q2, p1, p2], pq) for pq in self.invPQ]

        p0 = self.PQ * sym.Matrix([[dq1dt_0], [dq2dt_0]])

        sp.integrate.RK45.__init__(self,
                                   self.grad,
                                   0, [q1_0, q2_0, p0[0], p0[1]],
                                   max_step=max_step,
                                   t_bound=t_bound,
                                   vectorized=True)
        self.y_old_old = None

    def w(self):
        return self.invPQ_num[0](*self.y), self.invPQ_num[1](*self.y)

    def energy(self):
        return self.H_num(*self.y)

    def coords(self):
        return [c(self.y[0], self.y[1]) for c in self.coords_num]

    def grad(self, t, y):
        """ q1, q2, p1, p2 """
        return np.vstack(e(y[0, :], y[1, :], y[2, :], y[3, :]) for e in self.eom_num)

    def sos(self):
        if (self.y_old is None) and (self.y[2] == 0):
            return True
        return (abs(self.y[2]) < 2) and (abs(self.y_old[2]) < abs(self.y[2])) and (
                    abs(self.y_old_old[2]) > abs(self.y_old[2]))

    def step(self):
        if self.y_old is not None:
            self.y_old_old = np.copy(self.y_old)
        else:
            self.y_old_old = np.copy(self.y)
        return sp.integrate.RK45.step(self)
