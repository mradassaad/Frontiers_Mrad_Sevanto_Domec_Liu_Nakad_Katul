import numpy as np
import numpy.ma as ma
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.optimize import root
from scipy import optimize
import warnings
from Plant_Env_Props import*
from Useful_Funcs import*
from scipy.optimize import minimize
import operator
from datetime import datetime
import glob

np.seterr(divide='warn')


class ConvergenceError(Error):
    """Exception raised for setting lambda to a value that makes finding
    a reasonable psi_l given the set vulnerability curve.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message



def dydt(t, y):
    """

    :param t: time in 30 min intervals
    :param y: y[0] is lambda(t) in mol/m2, y[1] is x(t) in mol/mol
    :return:
    """

    if np.any(y[0] < 0):
        raise GuessError('y[0] < 0')
    #
    if np.any(y[0] > (ca - cp_interp(t))/(VPDinterp(t) * alpha)):
        raise GuessError('y[0] too large')
    # ----------------- stomatal conductance based on current values -------------------
    gpart11 = (ca + k2_interp(t) - 2 * alpha * y[0] * VPDinterp(t)) *\
                    np.sqrt(alpha * y[0] * VPDinterp(t) * (ca - cp_interp(t)) * (cp_interp(t) + k2_interp(t)) *
                    ((ca + k2_interp(t)) - alpha * y[0] * VPDinterp(t)))  # mol/m2/d


    gpart12 = alpha * y[0] * VPDinterp(t) * ((ca + k2_interp(t)) - alpha * y[0] * VPDinterp(t))  # mol/mol

    gpart21 = gpart11 / gpart12  # mol/m2/d

    gpart22 = (ca - (2 * cp_interp(t) + k2_interp(t)))  # mol/m2/d

    gpart3 = gpart21 + gpart22  # mol/m2/d

    gpart4 = (ca + k2_interp(t))**2  # mol2/mol2

    zeta = gpart3 / gpart4  # unitless

    zeta_mask = ma.masked_less(zeta, 0)
    # zeta[zeta_mask.mask] = 0
    if np.any(zeta_mask.mask):
        print('Stop')

    gl = k1_interp(t) * zeta  # mol/m2/d
    # gl_mask = ma.masked_less(gl, 0)

    psi_x = psi_sat * y[1] ** (-b)  # Soil water potential, MPa

    # ----------------------------------- Find conduction maximum of plant ------------------------
    with warnings.catch_warnings():
        warnings.filterwarnings('error', category=RuntimeWarning)
        try:
            res_fsolve = root(psil_val, psi_x+1, args=(psi_x, psi_63, w_exp, Kmax, gl, lai, VPDinterp, t), method='broyden1')
        except RuntimeWarning:
            print('stomatal conductance exceeds transpiration capacity: increase lam guess or no sols')
            import sys
            sys.exit()

    psi_l = res_fsolve.x

    # psi_l_mask = ma.masked_less(psi_l, psi_x)
    # # psi_l[psi_l_mask.mask] = psi_x[psi_l_mask.mask]
    # psi_l[psi_l_mask.mask] = 999999
    # gl[psi_l_mask.mask] = 0
    # psi_l = 2 * psi_63 *\
    #         (np.log(Kmax / (gl[~gl_mask.mask] * VPDinterp(t[~gl_mask.mask])))) **\
    #         (1 / w_exp) - psi_x[~gl_mask.mask]  # leaf water pot, MPa

    psi_p = (psi_x + psi_l) / 2  # plant water potential, MPa

    dpsi_xdx = psi_sat * (-b) * y[1] ** (-b - 1)
    # dgldx = - Kmax / lai / VPDinterp(t) * dpsi_xdx * np.exp(-(psi_p / psi_63) ** w_exp) * \
    #         (0.5 * w_exp / psi_63 * (psi_p / psi_63) ** (w_exp - 1) * (psi_l - psi_x) + 1)  # mol/m2/d
    dEdx = - alpha / lai * Kmax * dpsi_xdx * np.exp(-(psi_p / psi_63) ** w_exp) *\
           (0.5 * (w_exp / psi_63) * (psi_p / psi_63) ** (w_exp-1) * (psi_l - psi_x) + 1)  # mol/m2/d
    # dEdx = alpha * VPDinterp(t) * dgldx
    # --------------- cost of sucking water through the plant stem, dAdx ---------------------
    dAdx = 0

    # dgldx = - Kmax / lai / VPDinterp(t) * dpsi_xdx * np.exp(-(psi_p / psi_63) ** w_exp) * \
    #                 (0.5 * w_exp / psi_63 * (psi_p / psi_63) ** (w_exp - 1) * (psi_l - psi_x) + 1)  # mol/m2/d

    # ddeltadx_part1 = - 2 * (ca - cp_interp(t)) / ((k2_interp(t) + ca) * zeta + 1)
    # ddeltadx_part2 = (ca + k2_interp(t))
    # ddeltadx_part3 = np.sqrt(1 - 4 * (ca - cp_interp(t)) * zeta / ((ca + k2_interp(t)) * zeta + 1) ** 2)
    #
    # ddeltadx = (ddeltadx_part1 + ddeltadx_part2) * dgldx / ddeltadx_part3  # mol/m2/d
    #
    # dAdx = 0.5 * ((ca + k2_interp(t)) * dgldx - ddeltadx)  # mol/m2/d

    # -------------- uncontrolled losses and evapo-trans ------------------------
    losses = 0
    dlossesdx = 0
    # Comment out following 2 lines if only plant hydraulic effects are sought
    losses = beta * y[1] ** c  # 1/d
    dlossesdx = beta * c * y[1] ** (c - 1)

    evap_trans = alpha * gl * VPDinterp(t)  # 1/d
    f = - (losses + evap_trans)  # 1/d

    dfdx = - (dlossesdx + dEdx)

    dlamdt = - (dAdx + y[0] * dfdx)  # mol/m2/d
    dxdt = f  # 1/d

    return np.vstack((dlamdt, dxdt))



# def Weibull(Xylem, psi_s, psi_l):
#
#     return Xylem.gpmax * (np.exp(((psi_s - psi_l) / Xylem.psi_63) ** Xylem.c))


# ------------------------OPT Boundary Conditions----------------

def bc(ya, yb):  # boundary imposed on x at t=T
    x0 = 0.42
    return np.array([ya[1] - x0, yb[1] - 0.28])


def bc_wus(ya, yb):  # Water use strategy
    x0 = 0.5
    wus_coeff = Lambda  # mol/m2
    return np.array([ya[1] - x0, yb[0] - wus_coeff])


# t = np.linspace(0, days, 2000)
maxLam = 763e-6*unit0
Lambda = maxLam*0.8 # mol/m2
# lam_guess = 5*np.ones((1, t.size)) + np.cumsum(np.ones(t.shape)*(50 - 2.67) / t.size)
lam_guess = 5*np.ones((1, t.size))# mol/m2
x_guess = 0.45*np.ones((1, t.size))

y_guess = np.vstack((lam_guess, x_guess))
try:
    res = solve_bvp(dydt, bc, t, y_guess, tol=1e-3, verbose=2, max_nodes=10000)
except OverflowError:
    print('Try reducing initial guess for lambdba')
    import sys
    sys.exit()

lam_plot = res.y[0]*unit1
soilM_plot = res.y[1]

gl = g_val(res.x, res.y[0], ca, alpha, VPDinterp, k1_interp, k2_interp, cp_interp)  # stomatal conductance, mol/m2/d

A_val = A(res.x, gl, ca, k1_interp, k2_interp, cp_interp)  # mol/m2/d

psi_x = psi_sat * soilM_plot ** (-b)  # Soil water potential, MPa


ci = ca - A_val / gl  # internal carbon concentration, mol/mol

psi_l_res = root(psil_val, psi_x+1, args=(psi_x, psi_63, w_exp, Kmax, gl, lai, VPDinterp, res.x), method='broyden1')
psi_l = psi_l_res.x
psi_p = 0.5 * (psi_x + psi_l)
PLC = 100 * (1 - np.exp(- (psi_p / psi_63) ** w_exp))

E = transpiration(psi_l, psi_x, psi_63, w_exp, Kmax)
f = - (beta * soilM_plot + alpha * E / lai)
objective_term_1 = np.sum(np.diff(res.x) * (A_val[1:] + res.y[0][1:] * f[1:]))  # mol/m2
objective_term_2 = Lambda * soilM_plot[-1]  # mol/m2
theta = objective_term_2 / (objective_term_1 + objective_term_2)

H = A_val + res.y[0]*f  # mol/m2/d

# --- for debugging and insight

plt.figure()
plt.subplot(331)
plt.plot(res.x, lam_plot)
#plt.xlabel("days")
plt.ylabel("$\lambda (t), mmol.mol^{-1}$")

plt.subplot(332)
plt.plot(res.x, soilM_plot)
# plt.xlabel("time, days")
plt.ylabel("$x(t)$")

plt.subplot(333)
plt.plot(res.x, gl / unit0)
plt.xlabel("time, days")
plt.ylabel("$g(t), mol.m^{-2}.s^{-1}$")


plt.subplot(334)
plt.plot(res.x, A_val * 1e6 / unit0)
# plt.xlabel("time, days")
plt.ylabel("$A, \mu mol.m^{-2}.s^{-1}$")

plt.subplot(335)
plt.plot(res.x, (E * alpha * n * z_r / lai))
# plt.xlabel("time, days")
plt.ylabel("$E, m.d^{-1}$")

plt.subplot(336)
plt.plot(res.x, psi_x)
# plt.xlabel("time, days")
plt.ylabel("$\psi_x, MPa$")

plt.subplot(337)
plt.plot(res.x, psi_l)
plt.xlabel("time, days")
plt.ylabel("$\psi_l, MPa$")

plt.subplot(338)
plt.plot(res.x, psi_p)
plt.xlabel("time, days")
plt.ylabel("$\psi_p, MPa$")

plt.subplot(339)
plt.plot(res.x, PLC)
plt.xlabel("time, days")
plt.ylabel("PLC, %")

# --- Fig1b

timeOfDay = 14.5/24

env_data = np.array([cp_interp(timeOfDay), VPDinterp(timeOfDay),
                     k1_interp(timeOfDay), k2_interp(timeOfDay)])
xvals = soilM_plot
psi_x_vals = psi_sat * xvals ** -b
psi_l_vals = np.zeros(xvals.shape)
trans_vals = np.zeros(xvals.shape)

i = 0
for x in xvals:
    OptRes = minimize(trans_opt, psi_x_vals[i], args=(psi_x_vals[i], psi_63, w_exp, Kmax))
    psi_l_vals[i] = OptRes.x
    trans_vals[i] = transpiration(OptRes.x, psi_x_vals[i], psi_63, w_exp, Kmax)
    i += 1


def lam(trans, ca, alpha, cp, VPD, k1, k2):
    gl_vals = trans / (VPD * lai)
    part11 = ca ** 2 * gl_vals + 2 * cp * k1 - ca * (k1 - 2 * gl_vals * k2) + k2 * (k1 + gl_vals * k2)
    part12 = np.sqrt(4 * (cp - ca) * gl_vals * k1 + (k1 + gl_vals * (ca + k2)) ** 2)
    part1 = part11 / part12

    part2 = ca + k2

    part3 = 2 * VPD * alpha
    return (part2 - part1) / part3


lam_low = lam(trans_vals, ca, alpha, env_data[0], env_data[1], env_data[2], env_data[3])

timeOfDay = 12/24
env_data = np.array([cp_interp(timeOfDay), VPDinterp(timeOfDay),
                     k1_interp(timeOfDay), k2_interp(timeOfDay)])
lam_up = np.ones(lam_low.shape) * (ca - env_data[0]) / env_data[1] / alpha

fig, ax = plt.subplots()
lam_line = ax.plot(res.x, lam_plot)
lam_low_line = ax.plot(res.x, lam_low*unit1, 'r:')
lam_high_line = ax.plot(res.x, lam_up*unit1, 'r:')