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
unit2 = 18*1e-6  # mol H2O/m2/s ->  m/s
unit3 = 1e6  # Pa -> MPa
unit4 = 273.15  # Degree C -> K


def A(a1, a2, ca, cp, g):
    """

    :param a1: Parameter corresponding to either light- or Rubisco-limited photosynthesis
    :param a2: Parameter corresponding to either light- or Rubisco-limited photosynthesis
    :param ca: ambient CO2 mole fraction in the air in mmol/mol
    :param cp: CO2 concentration at which assimilation is zero or compensation point in mmol/mol
    :param g: stomatal conductance in umol/m2/s
    :return: value of A at a particular value of g
    """
    gamma = np.sqrt((a2 + ca + a1/g) ** 2 + 4 * a1 * (cp - ca) / g)

    return 0.5 * (a1 + g * (ca + a2) + g * gamma)


def dAdg(a1, a2, ca, cp, g):
    """

    :param a1: Parameter corresponding to either light- or Rubisco-limited photosynthesis
    :param a2: Parameter corresponding to either light- or Rubisco-limited photosynthesis
    :param ca: ambient CO2 mole fraction in the air in mmol/mol
    :param cp: CO2 concentration at which assimilation is zero or compensation point in mmol/mol
    :param g: stomatal conductance in umol/m2/s
    :return: value of the partial derivative of A with respect to g at a particular value of g
    """
    ca *= 1e3
    cp *= 1e3
    gamma = np.sqrt((a2 + ca + a1 / g) ** 2 + 4 * a1 * (cp - ca) / g)

    return (1 / (2 * g * gamma)) * (g * (a2 + ca) ** 2 + a1 * (a2 - ca) + (a2 + ca) * g * gamma)


def a_light(alpha_p, em, par, cp):
    """

    :param alpha_p: absorptivity of the leaf for PAR
    :param em: maximum quantum efficiency in mol/mol
    :param par: PAR photon flux density incident on the leaf in umol/m2/s
    :param cp: CO2 concentration at which assimilation is zero or compensation point in mmol/mol
    :return: a1 in umol/m2/s and a2 in umol/mol
    """
    a1 = alpha_p * em * par
    a2 = 2 * cp * 1e3
    return a1, a2


def a_rubisco(vm, kc, coa, ko):
    """

    :param vm: the rubisco capacity in umol/m2/s
    :param kc: Michaelis constant for CO2 in umol/mol
    :param coa: oxygen mole fraction in the air in mmol/mol
    :param ko: inhibition constant for O2 in mmol/mol
    :return: a1 in umol/m2/s and a2 in umol/mol
    """
    a1 = vm
    a2 = kc * (1 + coa / ko)
    return a1, a2

def vm_calc(vm_25, Tl):
    """

    :param vm_25:  value of the rubisco capacity at leaf temperature Tl of 25C in umol/m2/s
    :param Tl: leaf temperature in C
    :return: the rubisco capacity Vm in umol/m2/s
    """
    # The optimal temperature is considered 25C
    # The cutoff temperature is assumed 41C
    # For reference, check Norman and Campbell page 239-241
    return vm_25 * np.exp(0.088 * (Tl - 25)) / (1 + np.exp(0.29 * (Tl - 41)))

def dydt(t, y):
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


def bc(ya, yb):  # boundary imposed on x at t=T
    x0 = 0.8
    return np.array([ya[1] - x0, yb[1]])


def bc_wus(ya,yb):  # Water use strategy
    x0 = 0.8
    wus_coeff = 800e-6*t_day*unit0  # mol/m2
    return np.array([ya[1] - x0, yb[0] - wus_coeff])


t = np.linspace(0, 20, 1000)

lam_guess = 1*np.ones((1, t.size))
x_guess = 0.8*np.ones((1, t.size))

y_guess = np.vstack((lam_guess, x_guess))

res = solve_bvp(dydt, bc, t, y_guess)

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