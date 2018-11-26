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

fig, ax = plt.subplots(figsize=(6.15,4.5),tight_layout=True)
dEdx_max, = ax.plot(psi_x_vals, dtransdx_vals_max * alpha * n * z_r / lai + dK_x_valsdx)
dEdx_min, = ax.plot(psi_x_vals, dtransdx_vals_min * alpha * n * z_r / lai + dK_x_valsdx)
ax.set_xlabel("Soil water potential, $\psi_x$, MPa", FontSize=16)
ax.set_ylabel("$\partial (E+L) / \partial x$, m.day$^{-1}$", FontSize=16)
plt.setp(ax.get_xticklabels(), FontSize=12)
plt.setp(ax.get_yticklabels(), FontSize=12)
legend = ax.legend((dEdx_max, dEdx_min),
                   ('maximum', 'minimum'), fontsize='large', loc=2, title='$\psi_{63}$=3, s=2')

# fig.savefig('dELdx.eps',transparent=True)
#
fig2, ax2 = plt.subplots(figsize=(6.15,4.5),tight_layout=True)
psil_max, = ax2.plot(psi_x_vals, ddx_psi_l_vals_max)
psil_min, = ax2.plot(psi_x_vals, ddx_psi_l_vals_min)
psil_trans_max, = ax2.plot(psi_x_vals, psi_l_vals, 'r:')
psix, = ax2.plot(psi_x_vals, psi_x_vals, 'r--')
ax2.set_xlabel("Soil water potential, $\psi_x$, MPa", FontSize=16)
ax2.set_ylabel("Optimal $\psi_L$, MPa", FontSize=16)
plt.setp(ax2.get_xticklabels(), FontSize=12)
plt.setp(ax2.get_yticklabels(), FontSize=12)
legend2 = ax2.legend((psil_max, psil_min, psil_trans_max, psix),
                   ('max $\partial (E+L) / \partial x$', 'min $\partial (E+L) / \partial x$',
                    'max transpiration E', '$\psi_x$'),
                     fontsize='large', loc=6, title='$\psi_{63}$=3, s=2')
#
# fig2.savefig('OptPsiLdEdx.eps',transparent=True)

i = 0
psi_63 = 2.5
trans_vals_psilow = np.zeros(xvals.shape)
for x in xvals:
    OptRes = minimize(trans_opt, psi_x_vals[i], args=(psi_x_vals[i], psi_63, w_exp, Kmax))
    trans_vals_psilow[i] = transpiration(OptRes.x, psi_x_vals[i], psi_63, w_exp, Kmax)
    i += 1

i = 0
psi_63 = 4
trans_vals_psihigh = np.zeros(xvals.shape)
for x in xvals:
    OptRes = minimize(trans_opt, psi_x_vals[i], args=(psi_x_vals[i], psi_63, w_exp, Kmax))
    trans_vals_psihigh[i] = transpiration(OptRes.x, psi_x_vals[i], psi_63, w_exp, Kmax)
    i += 1

i = 0
psi_63 = 3
w_exp = 1
trans_vals_slow = np.zeros(xvals.shape)
for x in xvals:
    OptRes = minimize(trans_opt, psi_x_vals[i], args=(psi_x_vals[i], psi_63, w_exp, Kmax))
    trans_vals_slow[i] = transpiration(OptRes.x, psi_x_vals[i], psi_63, w_exp, Kmax)
    i += 1

i = 0
psi_63 = 3
w_exp = 3
trans_vals_shigh = np.zeros(xvals.shape)
for x in xvals:
    OptRes = minimize(trans_opt, psi_x_vals[i], args=(psi_x_vals[i], psi_63, w_exp, Kmax))
    trans_vals_shigh[i] = transpiration(OptRes.x, psi_x_vals[i], psi_63, w_exp, Kmax)
    i += 1

fig3, ax3 = plt.subplots(figsize=(6.15,4.5),tight_layout=True)
trans_max, = ax3.plot(psi_x_vals, trans_vals * alpha * n * z_r / lai + K_x_vals, 'r')
trans_max_psilow, = ax3.plot(psi_x_vals, trans_vals_psilow * alpha * n * z_r / lai + K_x_vals, 'r--')
trans_max_psihigh, = ax3.plot(psi_x_vals, trans_vals_psihigh * alpha * n * z_r / lai + K_x_vals, 'r:')
trans_max_slow, = ax3.plot(psi_x_vals, trans_vals_slow * alpha * n * z_r / lai + K_x_vals, 'b')
trans_max_shigh, = ax3.plot(psi_x_vals, trans_vals_shigh * alpha * n * z_r / lai + K_x_vals, 'k')
ax3.set_xlabel("Soil water potential, $\psi_x$, MPa", FontSize=16)
ax3.set_ylabel("Max water loss, E+L, m.d$^{-1}$", FontSize=16)
plt.setp(ax3.get_xticklabels(), FontSize=12)
plt.setp(ax3.get_yticklabels(), FontSize=12)
ax3.set_ylim(0, 0.018)
legend3_1 = ax3.legend((trans_max, trans_max_psilow, trans_max_psihigh),
                   ('$\psi_{63}$=3', '$\psi_{63}$=2.5', '$\psi_{63}$=4'),
                     fontsize='large', loc=9, title='s=2')
ax3.add_artist(legend3_1)
legend3_2 = ax3.legend((trans_max, trans_max_slow, trans_max_shigh),
                   ('s=2', 's=1', 's=3'),
                     fontsize='large', loc=1, title='$\psi_{63}$=3')

# fig3.savefig('maxTrans.eps',transparent=True)