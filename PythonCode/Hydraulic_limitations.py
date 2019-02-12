import numpy as np
import pandas as pd
from Plant_Env_Props import*
from Useful_Funcs import*
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# --------- Maximum transpiration rate and leaf water potential vs soil water potential ------

x = np.arange(0.1, 0.3, 0.005)

psi_sat = 1.5 * unit6  # Soil water potential at saturation, MPa
b = 3.1  # other parameter
RAI = 5  # m3 m-3
gamma = 0.00072 * unit5   # m/d, for sandy loam page 130 Campbell and Norman
c = 2*b+3

gSR = gSR_val(x, gamma, b, d_r, z_r, RAI, lai)
# -------------------- psi_63=3, s=3 ------------------------
psi_63 = 3  # Pressure at which there is 64% loss of conductivity, MPa
w_exp = 4  # Weibull exponent
Kmax = 2e-3 * unit0  # Maximum plant stem water leaf area-averaged conductivity, mol/m2/d/MPa
reversible = 1

psi_x = psi_sat * x ** -b
psi_l = np.zeros(x.shape)
psi_r = np.zeros(x.shape)
trans_max = np.zeros(x.shape)
i = 0
trans_res = trans_crit(x[i], psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI, lai)
trans_max[i] = trans_res[0]
psi_r[i] = trans_res[1]
psi_l[i] = trans_res[2]
i = 1
for xx in x[1:]:
    # OptRes = minimize(trans_opt, psi_x[i], args=(x[i], psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI, reversible))
    # psi_l[i] = OptRes.x
    # trans_res = transpiration(OptRes.x, x[i], psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI, reversible)
    trans_res = trans_crit(x[i], psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI, lai, psi_l[i-1])
    trans_max[i] = trans_res[0]
    psi_r[i] = trans_res[1]
    psi_l[i] = trans_res[2]
    i += 1

trans_max_interp_3_3 = interp1d(psi_x, trans_max, kind='cubic')
psi_l_interp_3_3 = interp1d(psi_x, psi_l, kind='cubic')
psi_r_interp_3_3 = interp1d(psi_x, psi_r, kind='cubic')

# gRL_3_3 = plant_cond(psi_r, psi_l, psi_63, w_exp, Kmax, reversible)
# gRL_3_3[np.abs(psi_x - psi_r) < 0.01] = 0
gRL_3_3 = trans_max / (psi_l - psi_r)
# -------------------- psi_63=1.5, s=3 ------------------------
psi_63 = 1.5  # Pressure at which there is 64% loss of conductivity, MPa
w_exp = 4  # Weibull exponent
Kmax = 2e-3 * unit0  # Maximum plant stem water leaf area-averaged conductivity, mol/m2/d/MPa
reversible = 1

psi_l = np.zeros(x.shape)
psi_r = np.zeros(x.shape)
trans_max = np.zeros(x.shape)
i = 0
trans_res = trans_crit(x[i], psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI, lai)
trans_max[i] = trans_res[0]
psi_r[i] = trans_res[1]
psi_l[i] = trans_res[2]
i = 1
for xx in x[1:]:
    # OptRes = minimize(trans_opt, psi_x[i], args=(x[i], psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI, reversible))
    # psi_l[i] = OptRes.x
    # trans_res = transpiration(OptRes.x, x[i], psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI, reversible)
    trans_res = trans_crit(x[i], psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI, lai, psi_l[i-1])
    trans_max[i] = trans_res[0]
    psi_r[i] = trans_res[1]
    psi_l[i] = trans_res[2]
    i += 1

trans_max_interp_15_3 = interp1d(psi_x, trans_max, kind='cubic')
psi_l_interp_15_3 = interp1d(psi_x, psi_l, kind='cubic')
psi_r_interp_15_3 = interp1d(psi_x, psi_r, kind='cubic')

# gRL_15_3 = plant_cond(psi_r, psi_l, psi_63, w_exp, Kmax, reversible)
# gRL_15_3[np.abs(psi_x - psi_r) < 0.01] = 0
gRL_15_3 = trans_max / (psi_l - psi_r)
# -------------------- psi_63=1, s=1.2, Kmax=8 ------------------------
psi_63 = 1  # Pressure at which there is 64% loss of conductivity, MPa
w_exp = 1  # Weibull exponent
Kmax = 8e-3 * unit0  # Maximum plant stem water leaf area-averaged conductivity, mol/m2/d/MPa
reversible = 1

psi_l = np.zeros(x.shape)
psi_r = np.zeros(x.shape)
trans_max = np.zeros(x.shape)
i = 0
trans_res = trans_crit(x[i], psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI, lai)
trans_max[i] = trans_res[0]
psi_r[i] = trans_res[1]
psi_l[i] = trans_res[2]
i = 1
for xx in x[1:]:
    # OptRes = minimize(trans_opt, psi_x[i], args=(x[i], psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI, reversible))
    # psi_l[i] = OptRes.x
    # trans_res = transpiration(OptRes.x, x[i], psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI, reversible)
    trans_res = trans_crit(x[i], psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI, lai, psi_l[i-1])
    trans_max[i] = trans_res[0]
    psi_r[i] = trans_res[1]
    psi_l[i] = trans_res[2]
    i += 1

trans_max_interp_3_1 = interp1d(psi_x, trans_max, kind='cubic')
psi_l_interp_3_1 = interp1d(psi_x, psi_l, kind='cubic')
psi_r_interp_3_1 = interp1d(psi_x, psi_r, kind='cubic')

# gRL_3_1 = plant_cond(psi_r, psi_l, psi_63, w_exp, Kmax, reversible)
# gRL_3_1[np.abs(psi_x - psi_r) < 0.01] = 0
gRL_3_1 = trans_max / (psi_l - psi_r)

fig, ax = plt.subplots()
gSR_plot, = ax.semilogy(-psi_x, gSR * 1e3 / unit0, 'r', linewidth=2)
gRL_3_3_plot, = ax.semilogy(-psi_x, gRL_3_3 * 1e3 / unit0, 'k', linewidth=2)
gRL_15_3_plot, = ax.semilogy(-psi_x, gRL_15_3 * 1e3 / unit0, 'k--', linewidth=2)
gRL_3_1_plot, = ax.semilogy(-psi_x, gRL_3_1 * 1e3 / unit0, 'k:', linewidth=2)
ax.set_ylim(1e-3, 1e0 + 4)
ax.set_xlim(-1.5, 0)
plt.setp(ax.get_xticklabels(), FontSize=12)
plt.setp(ax.get_yticklabels(), FontSize=12)

ax.set_xlabel("Soil Water Potential, $\psi_x$, MPa", FontSize=14)
ax.set_ylabel("Maximizing conductance, $g_{max}$, mmol m$^{-2}$ MPa$^{-1}$ s$^{-1}$", FontSize=11)
legend1 = ax.legend((gSR_plot, gRL_3_3_plot),
                   ('$g_{sr}$', '$g_{rl}$'),
                    fontsize='large', loc=2, title='Color')
ax.add_artist(legend1)
legend2 = ax.legend((gRL_3_3_plot, gRL_15_3_plot, gRL_3_1_plot),
                   ('$\psi_{63}=-3$, $s=4$, $g_{rl,max}=2$',
                    '$\psi_{63}=-1.5$, $s=4$, $g_{rl,max}=2$',
                    '$\psi_{63}=-1$, $s=1$, $g_{rl,max}=8$'),
                    fontsize='small', loc=9, title='Line style')
#
# inset = fig.add_axes([.58, 0.16, .3, 0.3])
# trans_3_3, = inset.semilogy(-psi_x,
#              trans_max_interp_3_3(psi_x) * 1e3 / unit0, 'k', linewidth=2)
# trans_15_3, = inset.semilogy(-psi_x,
#              trans_max_interp_15_3(psi_x) * 1e3 / unit0, 'k--', linewidth=2)
# trans_3_1, = inset.semilogy(-psi_x,
#              trans_max_interp_3_1(psi_x) * 1e3 / unit0, 'k:', linewidth=2)
# inset.set_ylim(1e-3, 1e0)
# inset.set_yticks(np.array((1e-3, 1e-2, 1e-1)))
# inset.set_ylabel("$E_{max}$, mmol m$^{-2}$ s$^{-1}$", FontSize=8)
# inset.set_yticklabels(inset.get_yticklabels()[:-1])

# fig.savefig('../g_psix.pdf', bbox_inches='tight')

fig2, ax2 = plt.subplots()

ax2.set_ylabel('Maximizing water potential, $\psi_{max}$, MPa', FontSize=14)
psi_l_3_3, = ax2.plot(-psi_x, -psi_l_interp_3_3(psi_x), 'g', linewidth=2)
psi_l_15_3, = ax2.plot(-psi_x, -psi_l_interp_15_3(psi_x), 'g--', linewidth=2)
psi_l_3_1, = ax2.plot(-psi_x, -psi_l_interp_3_1(psi_x), 'g:', linewidth=2)

psi_r_3_3, = ax2.plot(-psi_x, -psi_r_interp_3_3(psi_x), 'b', linewidth=2)
psi_r_15_3, = ax2.plot(-psi_x, -psi_r_interp_15_3(psi_x), 'b--', linewidth=2)
psi_r_3_1, = ax2.plot(-psi_x, -psi_r_interp_3_1(psi_x), 'b:', linewidth=2)

plt.setp(ax2.get_xticklabels(), FontSize=12)
plt.setp(ax2.get_yticklabels(), FontSize=12)

# ax2.set_ylim(-7, 0)
ax2.set_xlim(-1.5, 0)

legend1 = ax2.legend((psi_l_3_3, psi_r_3_3),
                   ('$\psi_{l,max}$', '$\psi_{r,max}$'),
                    fontsize='large', loc=4, title='Color')

# fig2.savefig('../psil_psir_psix.pdf', bbox_inches='tight')
# ax2.add_artist(legend1)
# legend2 = ax2.legend((trans_3_3, trans_15_3, trans_3_1),
#                    ('$\psi_{63}=-3$, $s=3$, $g_{rl,max}=2$',
#                     '$\psi_{63}=-1.5$, $s=3$, $g_{rl,max}=2$',
#                     '$\psi_{63}=-1.4$, $s=1.3$, $g_{rl,max}=8$'),
#                     fontsize='small', loc=9, title='Line style')

figE, axE = plt.subplots()
trans_3_3, = axE.semilogy(-psi_x,
             trans_max_interp_3_3(psi_x) * 1e3 / unit0, 'k', linewidth=2)
trans_15_3, = axE.semilogy(-psi_x,
             trans_max_interp_15_3(psi_x) * 1e3 / unit0, 'k--', linewidth=2)
trans_3_1, = axE.semilogy(-psi_x,
             trans_max_interp_3_1(psi_x) * 1e3 / unit0, 'k:', linewidth=2)
axE.set_ylim(1e-3, 1e0)
axE.set_ylabel("Maximum transpiration, $E_{max}$, mmol m$^{-2}$ s$^{-1}$", FontSize=14)
axE.set_xlabel("Soil Water Potential, $\psi_x$, MPa", FontSize=14)

plt.setp(axE.get_xticklabels(), FontSize=12)
plt.setp(axE.get_yticklabels(), FontSize=12)
axE.set_xlim(-1.5, 0)
# figE.savefig('../E_psix.pdf', bbox_inches='tight')
# -------- Compute hydraulic limitations on Lambda
#
# sample_times = np.array([6/24, 12/24, 14.5/24, 18/24])
# env_data = np.array([cp_interp(sample_times), VPDinterp(sample_times),
#                      k1_interp(sample_times), k2_interp(sample_times)])
#
# xvals = np.arange(0.12, 0.5, 0.001)
# psi_x_vals = psi_sat * xvals ** -b
# psi_l_vals = np.zeros(xvals.shape)
# psi_r_vals = np.zeros(xvals.shape)
# trans_vals = np.zeros(xvals.shape)
# i = 0
# psi_63 = 3
# w_exp = 2
# for x in xvals:
#     OptRes = minimize(trans_opt, psi_x_vals[i], args=(xvals[i], psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI))
#     psi_l_vals[i] = OptRes.x
#     trans_res = transpiration(OptRes.x, xvals[i], psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI)
#     trans_vals[i] = trans_res[0]
#     psi_r_vals[i] = trans_res[1]
#     i += 1
#
#
def lam(trans, ca, cp, VPD, k1, k2):
    gl_vals = trans / (VPD * 1.6)
    part11 = ca ** 2 * gl_vals + 2 * cp * k1 - ca * (k1 - 2 * gl_vals * k2) + k2 * (k1 + gl_vals * k2)
    part12 = np.sqrt(4 * (cp - ca) * gl_vals * k1 + (k1 + gl_vals * (ca + k2)) ** 2)
    part1 = part11 / part12

    part2 = ca + k2

    part3 = 2 * VPD * 1.6
    # returns lambda in mol/mol
    return (part2 - part1) / part3

k1 = 1  # mol m-2 d-1
k2 = 1.8e-4  # mol mol-1
cp = 25e-6  # mol mol-1
ca = 350e-6  # mol mol-1

VPD = 8e-3  # mol mol-1
lam_val_3_3_8 = lam(trans_max_interp_3_3(psi_x), ca, cp, VPD, k1, k2)
lam_val_15_3_8 = lam(trans_max_interp_15_3(psi_x), ca, cp, VPD, k1, k2)
lam_val_3_1_8 = lam(trans_max_interp_3_1(psi_x), ca, cp, VPD, k1, k2)
lam_up_8 = (ca - cp) * 1e3 / 8e-3 / 1.6

VPD = 16e-3  # mol mol-1
lam_val_3_3_16 = lam(trans_max_interp_3_3(psi_x), ca, cp, VPD, k1, k2)
lam_val_15_3_16 = lam(trans_max_interp_15_3(psi_x), ca, cp, VPD, k1, k2)
lam_val_3_1_16 = lam(trans_max_interp_3_1(psi_x), ca, cp, VPD, k1, k2)
lam_up_16 = (ca - cp) * 1e3 / 16e-3 / 1.6

VPD = 32e-3  # mol mol-1
lam_val_3_3_32 = lam(trans_max_interp_3_3(psi_x), ca, cp, VPD, k1, k2)
lam_val_15_3_32 = lam(trans_max_interp_15_3(psi_x), ca, cp, VPD, k1, k2)
lam_val_3_1_32 = lam(trans_max_interp_3_1(psi_x), ca, cp, VPD, k1, k2)
lam_up_32 = (ca - cp) * 1e3 / 32e-3 / 1.6

# fig3, ax3 = plt.subplots()
#
# lam_plot_3_3_8, = ax3.plot(-psi_x, lam_val_3_3_8 * 1e3 / 8e-3, 'k')
# lam_plot_15_3_8, = ax3.plot(-psi_x, lam_val_15_3_8 * 1e3 / 8e-3, 'k--')
# lam_plot_3_1_8, = ax3.plot(-psi_x, lam_val_3_1_8 * 1e3 / 8e-3, 'k:')
#
# lam_plot_3_3_16, = ax3.plot(-psi_x, lam_val_3_3_16 * 1e3 / 16e-3, 'b')
# lam_plot_15_3_16, = ax3.plot(-psi_x, lam_val_15_3_16 * 1e3 / 16e-3, 'b--')
# lam_plot_3_1_16, = ax3.plot(-psi_x, lam_val_3_1_16 * 1e3 / 16e-3, 'b:')
#
# lam_plot_3_3_32, = ax3.plot(-psi_x, lam_val_3_3_32 * 1e3 / 32e-3, 'g')
# lam_plot_15_3_32, = ax3.plot(-psi_x, lam_val_15_3_32 * 1e3 / 32e-3, 'g--')
# lam_plot_3_1_32, = ax3.plot(-psi_x, lam_val_3_1_32 * 1e3 / 32e-3, 'g:')
#
# ax3.set_xlim(-1.4, 0)
#
# ax3.set_xlabel('Soil water potential, $\psi_x$, MPa', FontSize=14)
# ax3.set_ylabel('Lower bound on $\lambda$, $\lambda_{lower}$, mmol mol$^{-1}$', FontSize=12)
#
# legend1 = ax3.legend((lam_plot_3_3_8, lam_plot_3_3_16, lam_plot_3_3_32),
#                    ('VPD=8 mmol mol$^{-1}$', 'VPD=16 mmol mol$^{-1}$', 'VPD=32 mmol mol$^{-1}$'),
#                     fontsize='small', loc=1, title='Color')
# ax3.add_artist(legend1)
# legend2 = ax3.legend((lam_plot_3_3_8, lam_plot_15_3_8, lam_plot_3_1_8),
#                    ('$\psi_{63}=-3$, $s=4$, $g_{rl,max}=2$',
#                     '$\psi_{63}=-1.5$, $s=4$, $g_{rl,max}=2$',
#                     '$\psi_{63}=-1$, $s=1$, $g_{rl,max}=8$'),
#                     fontsize='small', loc=3, title='Line style')

# fig3.savefig('../lam_psix.pdf', bbox_inches='tight')
