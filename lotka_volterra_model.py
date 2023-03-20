import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


a = 1.2
b = 0.6
c = 0.3
d = 0.8
x0 = 2
y0 = 1

dt = 0.01
tmax = 25
Nt = int(tmax/dt)
t = np.linspace(0, tmax, Nt)
X0 = [x0, y0]

def derivative(X, t, a, b, c, d):
    """Rozniczkowanie"""
    x, y = X
    dotx = x * (a - b * y)
    doty = y * (-d + c * x)
    return np.array([dotx, doty])

def Euler(func, X0, t, a, b, c, d):
    """Ofiary i Drapiezniki."""

    dt = t[1] - t[0]
    nt = len(t)
    X  = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        X[i+1] = X[i] + func(X[i], t[i], a,  b, c, d) * dt
    return X

Xe = Euler(derivative, X0, t, a, b, c, d)
plt.figure()
plt.title(f"Metoda Eulera dt = {dt}")
plt.plot(t, Xe[:, 0], label="Ofiary")
plt.plot(t, Xe[:, 1], label="Drapiezniki")
plt.grid()
plt.xlabel("Czas")
plt.ylabel("Populacja")
plt.legend()
plt.show()

res = odeint(derivative, X0, t, args=(a, b, c, d))

x, y = res.T

plt.figure()
plt.grid()
plt.title(f"Metoda odeint dla dt = {dt}")
plt.plot(t, x, label='Ofiary')
plt.plot(t, y, label="Drapiezniki")
plt.xlabel('Czas')
plt.ylabel('Populacja')
plt.legend()
plt.show()
print(f"Sredni Blad Aproksymacji dla dt = {dt} wynosi {abs(np.mean(Xe-res))}")
