import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

def fun(x,y):
    UNIT0 = 3600*24 #1/s -> 1/d

    gamma = 0.01 #m/day
    VPD = 0.015 #mol/mol
    k = 0.05*UNIT0 #/m2/day
    Zr = 0.3 #m
    ca = 350e-6 #mol/mol
    a = 1.6
    n = 0.5 #m3/m3
    LAI = 5
    Mw = 0.018 #kg/mol
    Tday = 12 #h/day
    rhow = 1000 #kg/m3
    nu = LAI*Mw*(Tday/24)/rhow #m3/mol
    
    g = k*(np.sqrt(ca*n*Zr*(a*VPD*y[0]*nu)**(-1)))

    return np.vstack((np.zeros(y[0].shape),-(gamma+nu*a*g*VPD)/(n*Zr)))

def bc(ya, yb):
    x0 = 0.8
    return np.array([ya[1] - x0, yb[1]])

x = np.linspace(0,20,1000)

lam_guess = 1e-3*np.ones((1,x.size))
x_guess = 0.8*np.ones((1,x.size))

y_guess = np.vstack((lam_guess,x_guess))

res = solve_bvp(fun, bc, x, y_guess)

lam_plot = res.sol(x)[0]
soilM_plot = res.sol(x)[1]
plt.subplot(211)
plt.plot(x,lam_plot)
plt.xlabel("days")
plt.ylabel("$\lambda (t)$")
plt.subplot(212)
plt.plot(x,soilM_plot)
plt.xlabel("days")
plt.ylabel("$x(t)$")