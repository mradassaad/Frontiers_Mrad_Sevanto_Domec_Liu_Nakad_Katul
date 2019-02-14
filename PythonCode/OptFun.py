import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
import warnings
from Useful_Funcs import*
from scipy.interpolate import interp1d

# from Plant_Env_Props import*
from pickle_extract import*

np.seterr(divide='raise')
np.seterr(divide='raise')


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

    if np.any(y[0] > (ca - cp_interp(t))/(VPDinterp(t) * 1.6)):
        raise GuessError('y[0] too large')

    elif np.any(y[0] < 0):

        raise GuessError('y[0] < 0')

    # ----------------- stomatal conductance based on current values -------------------
    gl = g_val(t, y[0], ca,
               VPDinterp, k1_interp, k2_interp, cp_interp)  # mol/m2/d per unit LEAF area

    psi_x = psi_sat * y[1] ** -b
    trans_max = trans_max_interp(psi_x)  # mol/m2/d per unit LEAF area

    ok = np.less_equal(1.6 * gl * VPDinterp(t), trans_max)
    Nok = ~ok

    psi_r = np.zeros(t.shape); psi_l = np.zeros(t.shape)
    dEdx = np.zeros(t.shape); evap_trans = np.zeros(t.shape)

    psi_r[ok] = psi_r_val(y[1][ok], psi_sat, gamma, b, d_r, z_r, RAI, gl[ok], lai, VPDinterp, t[ok])
    evap_trans[ok] = 1.6 * gl[ok] * VPDinterp(t[ok])  # mol m-2 d-1 per unit leaf area per rooting depth
    dEdx[ok] = 0
    # ----------------------------------- Find psi_l ------------------------
    with warnings.catch_warnings():
        warnings.filterwarnings('error', category=RuntimeWarning)
        try:
            res_fsolve = root(psil_val_integral, psi_r[ok] + 0.1,
                                args=(psi_r[ok], psi_63, w_exp, Kmax, gl[ok], VPDinterp, t[ok], reversible),
                                method='hybr')
        except RuntimeWarning:
            print('stomatal conductance exceeds transpiration capacity: increase lam guess or no sols')
            import sys
            sys.exit()

    psi_l[ok] = res_fsolve.x

    # ----------------------------- When  transpiration limit is reached --------------
    if np.any(np.equal(Nok, True)):
        evap_trans[Nok] = trans_max_interp(psi_x[Nok])
        psi_r[Nok] = psi_r_interp(psi_x[Nok])
        psi_l[Nok] = psi_l_interp(psi_x[Nok])

    # -------------- uncontrolled losses and evapo-trans ------------------------
    losses = 0
    dlossesdx = 0

    # Comment out following 2 lines if only plant hydraulic effects are sought
    # losses = beta * y[1] ** c + 0.1 * y[1] ** 2  # 1/d
    # dlossesdx = beta * c * y[1] ** (c - 1) + 0.2 * y[1]  # 1/d

    f = - (losses + evap_trans)  # mol m-2 d-1 per unit leaf area per rooting depth
    dfdx = - (dlossesdx + dEdx)  # mol m-2 d-1
    dlamdt = - y[0] * dfdx  # mol/m2/d
    dxdt = f * nu / n / z_r  # 1/d
    return np.vstack((dlamdt, dxdt))


# ------------------------OPT Boundary Conditions----------------

def bc(ya, yb):  # boundary imposed on x at t=T
    x0 = 0.22
    return np.array([ya[1] - x0, yb[1] - 0.16])


def bc_wus(ya, yb):  # Water use strategy
    x0 = 0.2
    wus_coeff = Lambda  # mol/m2
    return np.array([ya[1] - x0, yb[0] - wus_coeff])

# t = np.linspace(0, days, 2000)
# maxLam = 763e-6*unit0
Lambda = 5 * 1e-3  # mol/mol
# lam_guess = 5*np.ones((1, t.size)) + np.cumsum(np.ones(t.shape)*(50 - 2.67) / t.size)
lam_guess = 2 * 1e-3 * np.ones((1, t.size))  # mol/mol
x_guess = 0.22*np.ones((1, t.size))

y_guess = np.vstack((lam_guess, x_guess))

# ---------------- SOLVER - SOLVER - SOLVER - SOLVER - SOLVER --------------------
try:
    res = solve_bvp(dydt, bc, t, y_guess, tol=1e-3, verbose=2, max_nodes=10000)
except OverflowError:
    print('Try reducing initial guess for lambda')
    import sys
    sys.exit()

# ---------------- PLOT - PLOT - PLOT - PLOT - PLOT --------------------

lam_plot = res.y[0] * 1e3
soilM_plot = res.y[1]

gl = g_val(res.x, res.y[0], ca, VPDinterp, k1_interp, k2_interp, cp_interp)  # stomatal conductance, mol/m2/d
# per unit LEAF area
gl[gl < 0] = 0

A_val = A(res.x, gl, ca, k1_interp, k2_interp, cp_interp)  # mol/m2/d

psi_x = psi_sat * soilM_plot ** (-b)  # Soil water potential, MPa

ci = np.zeros(res.x.shape)
notZero = gl != 0
ci[notZero] = ca - A_val[notZero] / gl[notZero]  # internal carbon concentration, mol/mol

trans_max = trans_max_interp(psi_x)  # mol m-2 d-1 per unit LEAF area
ok = np.less_equal(1.6 * gl * VPDinterp(res.x), trans_max)
Nok = ~ok

psi_r = np.zeros(res.x.shape)
psi_l = np.zeros(res.x.shape)
E = np.zeros(res.x.shape)

psi_r[ok] = psi_r_val(res.y[1][ok], psi_sat, gamma, b, d_r, z_r, RAI, gl[ok], lai, VPDinterp, res.x[ok])
# psi_l_res = root(psil_val, psi_r[ok]+0.1, args=(psi_r[ok], psi_63, w_exp, Kmax, gl[ok], VPDinterp, res.x[ok], reversible),
#                  method='hybr')
psi_l_res = root(psil_val_integral, psi_r[ok]+0.1, args=(psi_r[ok], psi_63, w_exp, Kmax, gl[ok], VPDinterp, res.x[ok], reversible),
                 method='hybr')
psi_l[ok] = psi_l_res.x
E[ok] = a * gl[ok] * VPDinterp(res.x[ok])  # mol m-2 d-1 per unit leaf area

E[Nok] = trans_max_interp(psi_x[Nok])
gl[Nok] = E[Nok] / a / VPDinterp(res.x[Nok])  # mol m-2 d-1 per unit leaf area
A_val[Nok] = A(res.x[Nok], gl[Nok], ca, k1_interp, k2_interp, cp_interp)
ci[Nok] = ca - A_val[Nok] / gl[Nok]
psi_r[Nok] = psi_r_interp(psi_x[Nok])
psi_l[Nok] = psi_l_interp(psi_x[Nok])

psi_p = (psi_l + psi_r) / 2
# PLC = 100*(1 - plant_cond(psi_r, psi_l, psi_63, w_exp, 1, reversible))
PLC_time = res.x[E > 0.01]
PLC = 100 * \
      np.maximum.accumulate(1 - E[E > 0.01] / (psi_l[E > 0.01] - psi_r[E > 0.01]) / Kmax)

# f = - (gamma / lai * soilM_plot ** c + 1.6 * E / lai + 0.1 * soilM_plot ** 2)  # mol m-2 day-1 per unit LEAF area
f = - (E)  # mol m-2 day-1 per unit LEAF area
objective_term_1 = np.sum(np.diff(res.x) * (A_val[1:] + res.y[0][1:] * f[1:])) * lai  # mol/m2 per GROUND area
objective_term_2 = Lambda * soilM_plot[-1]  # mol/mol
theta = objective_term_2 / (objective_term_1 + objective_term_2)

# ------------ profit ---------
E_crit = np.zeros(psi_x.shape)
A_crit = np.zeros(psi_x.shape)
P_crit = np.zeros(psi_x.shape)
P_opt = np.zeros(psi_x.shape)
psi_r_opt = np.zeros(psi_x.shape)
A_opt = np.zeros(psi_x.shape)
E_opt = np.zeros(psi_x.shape)
crits = np.array((psi_l_interp(psi_x), trans_max_interp(psi_x), k_crit_interp(psi_x), k_max_interp(psi_x)))

for i in range(psi_x.shape[0]):
    print(i)
    E_crit[i], A_crit[i], P_crit[i], E_opt[i], A_opt[i], P_opt[i], psi_r_opt[i] = \
         profit_max(psi_x[i], psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI, lai, ca,
                    k1_interp(res.x[i]), k2_interp(res.x[i]), cp_interp(res.x[i]), VPDinterp(res.x[i]),
                    0.01, crits[:, i])


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
plt.plot(res.x, E * 1e3 / unit0)  # mmol m-2 s-1
# plt.xlabel("time, days")
plt.ylabel("$E, mmol m^{-2} s^{-1}$")

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
plt.plot(PLC_time, PLC)
plt.xlabel("time, days")
plt.ylabel("PLC, %")

# --- Fig1b

timeOfDay = res.x

env_data = np.array([cp_interp(timeOfDay), VPDinterp(timeOfDay),
                     k1_interp(timeOfDay), k2_interp(timeOfDay)])


lam_low = lam_from_trans(trans_max_interp(psi_x), ca,
                         env_data[0], env_data[1], env_data[2], env_data[3])

env_data = np.array([cp_interp(timeOfDay), VPDinterp(timeOfDay),
                     k1_interp(timeOfDay), k2_interp(timeOfDay)])
lam_up = np.ones(lam_low.shape) * (ca - env_data[0]) / env_data[1] / 1.6  # mol mol-1
#
fig, ax = plt.subplots()
lam_line = ax.plot(res.x, lam_plot, 'r')
lam_low_line = ax.plot(res.x, lam_low * 1e3, 'r:')
lam_high_line = ax.plot(res.x, lam_up * 1e3, 'r:')


# --- Fig 2

mid_day = np.arange(0.5, 0.5 + (days - 1), 1)
gl_interp = interp1d(res.x, gl, kind='cubic')
psil_sim_interp = interp1d(res.x, psi_l, kind='cubic')
psix_sim_interp = interp1d(res.x, psi_x, kind='cubic')
gl_mid_day = gl_interp(mid_day)
psil_mid_day = psil_sim_interp(mid_day)
psix_mid_day = psix_sim_interp(mid_day)


# -----save----

H = np.zeros(A_val.shape)
oklam = np.greater_equal(res.y[0], lam_low * 1e-3)
H[oklam] = A_val[oklam] + res.y[0][oklam]*f[oklam]  # mol/m2/d
H[~oklam] = A_val[~oklam] + lam_low[~oklam]*f[~oklam]*1e-3  # mol/m2/d

inst = {'t': res.x, 'lam': res.y[0], 'x': res.y[1], 'gl': gl, 'A_val': A_val, 'psi_x': psi_x, 'psi_r': psi_r, 'psi_l': psi_l,
        'psi_p': psi_p, 'f': f, 'objective_term_1': objective_term_1, 'objective_term_2': objective_term_2,
        'theta': theta, 'PLC': PLC, 'H': H, 'lam_low': lam_low, 'lam_up': lam_up, 'E': E,
        'E_crit': E_crit, 'A_crit': A_crit, 'P_crit': P_crit, 'A_opt': A_opt, 'E_opt': E_opt,
        'P_opt': P_opt, 'psi_r_opt': psi_r_opt}

import pickle

pickle_out = open("../no_WUS/no_WUS.vulnerable", "wb")
pickle.dump(inst, pickle_out)
pickle_out.close()
