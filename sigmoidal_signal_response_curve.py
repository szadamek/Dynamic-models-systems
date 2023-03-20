import numpy as np
from sympy import Function, symbols, dsolve, Eq, Derivative, plot, init_printing


k1 = 1
k2 = 1
Rt = 1
km1 = 0.05
km2 = 0.05
S = 1

t = symbols('t')
Rp = Function('Rp')

eq = Eq(Rp(t).diff(t), (k1*S*(Rt-Rp(t))/(km1+Rt-Rp(t))) - k2*Rp(t)/(km2+Rp(t)))
sol = dsolve(eq, ics={Rp(0): 1})
print(sol.simplify())
plot(sol.rhs, (t, 0, 20))


from scipy.integrate import odeint
import matplotlib.pyplot as plt

def model(Rp,t,S):
    k1 = 1
    k2 = 1
    Rt = 1
    km1 = 0.05
    km2 = 0.05
    dRpdt = (k1*S*(Rt-Rp)/(km1+Rt-Rp)) - k2*Rp/(km2+Rp)
    return dRpdt

S = 1
Rp0 = [0, 0.3, 1]
dt = 0.1
t = np.linspace(0, 20, 200)
result = odeint(model, Rp0, t, args=(S,))

fig, ax = plt.subplots()
ax.plot(t, result[:, 0], label='R0=0')
ax.plot(t, result[:, 1], label='R0=0.3')
ax.plot(t, result[:, 2], label='R0=1')
ax.legend()
ax.set_xlabel('Czas')
ax.set_ylabel('Rp')
plt.show()
