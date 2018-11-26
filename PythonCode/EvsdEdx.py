from Useful_Funcs import *
from Plant_Env_Props import *
from scipy.optimize import minimize
from scipy.optimize import Bounds
import matplotlib.pyplot as plt

xvals = np.linspace(0.25, 0.4, 100)
psi_x_vals = psi_sat * xvals ** -b
K_x_vals = gamma * xvals ** c
dK_x_valsdx = c * gamma * xvals ** (c-1)
psi_l_vals = np.zeros(xvals.shape)
trans_vals = np.zeros(xvals.shape)
i = 0
for x in xvals:
    OptRes = minimize(trans_opt, psi_x_vals[i], args=(psi_x_vals[i], psi_63, w_exp, Kmax))
    psi_l_vals[i] = OptRes.x
    trans_vals[i] = transpiration(OptRes.x, psi_x_vals[i], psi_63, w_exp, Kmax)
    i += 1


i = 0
ddx_psi_l_vals_max = np.zeros(xvals.shape)
dtransdx_vals_max = np.zeros(xvals.shape)
ddx_psi_l_vals_min = np.zeros(xvals.shape)
dtransdx_vals_min = np.zeros(xvals.shape)
for x in xvals:
    OptRes_max = minimize(dtransdx_opt, (5 * psi_x_vals[i] + psi_l_vals[i]) / 6,
                      args=(xvals[i], psi_sat, b, psi_63, w_exp, Kmax), method='SLSQP',
                      bounds=[(psi_x_vals[i], psi_l_vals[i])])
    ddx_psi_l_vals_max[i] = OptRes_max.x
    dtransdx_vals_max[i] = dtransdx(OptRes_max.x, xvals[i], psi_sat, b, psi_63, w_exp, Kmax)

    OptRes_min = minimize(dtransdx, (psi_x_vals[i] + 2 * psi_l_vals[i]) / 3,
                          args=(xvals[i], psi_sat, b, psi_63, w_exp, Kmax), method='SLSQP',
                          bounds=[(psi_x_vals[i], psi_l_vals[i])], options={'eps':1e-8})

    ddx_psi_l_vals_min[i] = OptRes_min.x
    dtransdx_vals_min[i] = dtransdx(OptRes_min.x, xvals[i], psi_sat, b, psi_63, w_exp, Kmax)
    i += 1

fig, ax = plt.subplots()
dEdx_max = ax.plot(psi_x_vals, dtransdx_vals_max * alpha / lai + dK_x_valsdx / n / z_r)
dEdx_min = ax.plot(psi_x_vals, dtransdx_vals_min * alpha / lai + dK_x_valsdx / n / z_r)

fig2, ax2 = plt.subplots()
psil_max = ax2.plot(psi_x_vals, ddx_psi_l_vals_max)
psil_min = ax2.plot(psi_x_vals, ddx_psi_l_vals_min)
psil_trans_max = ax2.plot(psi_x_vals, psi_l_vals, 'r:')
psil_trans_max = ax2.plot(psi_x_vals, psi_x_vals, 'r--')

fig3, ax3 = plt.subplots()
trans_max = ax3.plot(psi_x_vals, trans_vals * alpha / lai + K_x_vals / n / z_r)
trans_min = ax3.plot(psi_x_vals, np.zeros(psi_x_vals.shape), 'r--')
