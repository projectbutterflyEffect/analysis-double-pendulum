
import sympy as sym

def _create_symbols():
    l1, l2, m1, m2, t1, t2 = sym.symbols("l_1, l_2, m_1, m_2, theta_1, theta_2")

    w1, w2 = sym.symbols("omega_1, omega_2")

    x1 = l1*sym.cos(t1)
    y1 = l1*sym.sin(t1)
    x2 = x1 + l2*sym.cos(t2)
    y2 = y1 + l2*sym.sin(t2)

    r1 = sym.Matrix([[x1],[y1]])
    r2 = sym.Matrix([[x2],[y2]])

    dr1dt = r1.diff(t1)*w1
    dr2dt = (r2.diff(t1)*w1+r2.diff(t2)*w2)

    T = ((m1*(dr1dt[0]**2+dr1dt[1]**2)/2+m2*(dr2dt[0]**2+dr2dt[1]**2)/2)).simplify()

    g = sym.symbols("g")
    V = (m1 + m2)*g*l1*sym.sin(t1) + m2*g*l2*sym.sin(t2)

    L = T-V
    q1, q2 = sym.symbols("q_1 q_2")
    L = L.subs({t1:q1,t2:q2})

    B = sym.Matrix([[l1**2*(m1+m2), l1*l2*m2*sym.cos(q1-q2)],
               [l1*l2*m2*sym.cos(q1-q2), l2**2*m2]])

    p1, p2 = sym.symbols('p_1 p_2')
    w = B.inv()*sym.Matrix([[p1],[p2]])

    H = p1*w[0]+p2*w[1]-L.subs({w1:w[0],w2:w[1]}).simplify()

    dq1dt = sym.Derivative(H,p1).doit().simplify()

    dq2dt = sym.Derivative(H,p2).doit().simplify()

    dp1dt = -sym.Derivative(H,q1).doit().simplify()

    dp2dt = -sym.Derivative(H,q2).doit().simplify()

    eom = sym.Matrix([dq1dt,dq2dt,dp1dt,dp2dt])

    return x1, x2, y1, y2, m1, m2, l1, l2, t1, q1, t2, q2, eom, H, p1, p2, g, B