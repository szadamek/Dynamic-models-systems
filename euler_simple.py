import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def lorenz(x, y, z, s=10, r=28, b=2.667):
    """
    Rozniczkowanie
    """
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot


dt = 0.002
tmax = 25
Nt = int(tmax/dt)
t = np.linspace(0, tmax, Nt)
nt = len(t)
# Need one more for the initial values
xs = np.empty(Nt + 1)
ys = np.empty(Nt + 1)
zs = np.empty(Nt + 1)

# Set initial values
xs[0], ys[0], zs[0] = (1, 1, 1)

# Step through "time", calculating the partial derivatives at the current point
# and using them to estimate the next point
for i in range(nt-1):
    x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
    xs[i + 1] = xs[i] + (x_dot * dt)
    ys[i + 1] = ys[i] + (y_dot * dt)
    zs[i + 1] = zs[i] + (z_dot * dt)


# Plot
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle(f'Lorenz Euler dt = {dt}')


ax1.plot(xs, ys)
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_title("z(x)")

ax2.plot(xs, zs)
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_title("z(x)")

ax3.plot(ys, zs)
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_title("z(y)")

plt.show()



def lorenz(state, t, sigma, beta, rho):
    """Metoda odient"""

    x, y, z = state

    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z

    return [dx, dy, dz]


sigma = 10.0
beta = 8.0 / 3.0
rho = 28.0

p = (sigma, beta, rho)

y0 = [1.0, 1.0, 1.0]
t = np.arange(0.0, 25.0, 0.002)

result = odeint(lorenz, y0, t, p)
x = result[:, 0]
y = result[:, 1]
z = result[:, 2]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle(f'Lorenz Odient dt = {dt}')


ax1.plot(x, y)
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_title("z(x)")

ax2.plot(x, z)
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_title("z(x)")

ax3.plot(y, z)
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_title("z(y)")

plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(xs, ys, zs)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(x, y, z)

plt.show()
