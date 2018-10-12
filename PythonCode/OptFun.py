import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

t_day = 0.7  # day/day
unit0 = 24*3600*t_day  # 1/s -> 1/d

gamma = 0.01  # m/day
vpd = 0.015  # mol/mol
k = 0.05*unit0  # mol/m2/day
z_r = 0.3  # m
ca = 350e-6  # mol/mol
a = 1.6
n = 0.5  # m3/m3
lai = 1.5
m_w = 0.018  # kg/mol

rhow = 1000  # kg/m3
nu = lai*m_w*(t_day)/rhow  # m3/mol
c = 1
beta = gamma / (n * z_r)  # 1/day
alpha = nu * a / (n * z_r)  # m2/mol

unit1 = 10**3*nu/(n*z_r)  # mol/m2 -> mmol/mol

def fun(t,y):
    """
    y[0] is lambda(t) and y[1] is x(t)
    t is the time
    """

    g = k * (np.sqrt(ca/(alpha*vpd*y[0]))-1)  # mol/m2/day

    losses = beta*y[1]**c  # 1/day
    evap_trans = alpha*g*vpd  # 1/day
    f = -(losses + evap_trans)  # 1/day

    dlamdt = y[0]*beta*c*y[1]**(c-1)  #mol/m2/day
    dxdt = f  # 1/day

    return np.vstack((dlamdt, dxdt))

def bc(ya, yb):
    x0 = 0.8
    return np.array([ya[1] - x0, yb[1]])

def bc_wus(ya,yb):
    x0 = 0.8
    wus_coeff = 800e-6*t_day*unit0  # mol/m2
    return np.array([ya[1] - x0, yb[0] - wus_coeff])


t = np.linspace(0,20,1000)

lam_guess = 1*np.ones((1, t.size))
x_guess = 0.8*np.ones((1, t.size))

y_guess = np.vstack((lam_guess, x_guess))

res = solve_bvp(fun, bc_wus, t, y_guess)

lam_plot = res.sol(t)[0]*unit1
soilM_plot = res.sol(t)[1]
plt.subplot(311)
plt.plot(t, lam_plot)
#plt.xlabel("days")
plt.ylabel("$\lambda (t), mmol.mol^{-1}$")

plt.subplot(312)
plt.plot(t, (k/unit0) * (np.sqrt(ca/(alpha*vpd*res.sol(t)[0]))-1))
plt.ylabel("$g(t), mol.m^{-2}.s^{-1}$")

plt.subplot(313)
plt.plot(t, soilM_plot)
plt.xlabel("time, days")
plt.ylabel("$x(t)$")

plt.show()