import numpy as np
import pandas as pd
from Plant_Env_Props import*
from Useful_Funcs import*
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# --------- Maximum transpiration rate and leaf water potential vs soil water potential ------

xvals = np.arange(0.1, 0.3, 0.001)

psi_sat = 1.5 * unit6  # Soil water potential at saturation, MPa
b = 3.1  # other parameter
RAI = 5  # m3 m-3
gamma = 0.00072 * unit5   # m/d, for sandy loam page 130 Campbell and Norman
c = 2*b+3

gSR = gSR_val(xvals, gamma, b, d_r, z_r, RAI, lai) * 1e3 / unit0  # mmol s-1 MPa-1 m-2
# -------------------- psi_63=4.3, s=5.4, Ponderosa pine ------------------------
psi_63 = 4.3  # Pressure at which there is 64% loss of conductivity, MPa
w_exp = 5.4  # Weibull exponent
Kmax = 2  # Maximum plant stem water leaf area-averaged conductivity, mmol/m2/s/MPa
reversible = 1

psi_x_vals = psi_sat * xvals ** -b
soil_root = gSR_val(xvals, gamma, b, d_r, z_r, RAI, lai)

res = root(psil_crit_val, psi_x_vals + 0.000001,
                                args=(psi_x_vals, psi_sat, gamma, b, d_r, z_r, RAI, lai, Kmax, psi_63, w_exp),
                                method='hybr')
psi_l_vals_steep_res = res.get('x')

res_psir = root(lambda psi_r: Kmax * psi_63 * gammafunc(1 / w_exp) / w_exp *
                                  (gammaincc(1 / w_exp, (psi_r / psi_63) ** w_exp) -
                                   gammaincc(1 / w_exp, (psi_l_vals_steep_res / psi_63) ** w_exp)) -
                                  soil_root * (psi_r - psi_x_vals),
                    (psi_l_vals_steep_res + psi_x_vals) / 2,
               method='hybr')
psi_r_vals_steep_res = res_psir.get('x')

trans_vals_steep_res = soil_root * (psi_r_vals_steep_res - psi_x_vals)
k_max_vals_steep_res = grl_val(psi_x_vals, psi_63, w_exp, Kmax) * soil_root /\
             (grl_val(psi_x_vals, psi_63, w_exp, Kmax) + soil_root)
k_crit_vals_steep_res = 0.05 * k_max_vals
trans_max_interp_steep_res = interp1d(psi_x_vals, trans_vals_steep_res, kind='cubic')
psi_l_interp_steep_res = interp1d(psi_x_vals, psi_l_vals_steep_res, kind='cubic')
psi_r_interp_steep_res = interp1d(psi_x_vals, psi_r_vals_steep_res, kind='cubic')

# gRL_3_3 = plant_cond(psi_r, psi_l, psi_63, w_exp, Kmax, reversible)
# gRL_3_3[np.abs(psi_x - psi_r) < 0.01] = 0
grl_psil = grl_val(psi_l_vals_steep_res, psi_63, w_exp, Kmax)
grl_psir = grl_val(psi_r_vals_steep_res, psi_63, w_exp, Kmax)
gRL_steep_res = grl_psil * soil_root / (grl_psir - grl_psil + soil_root)
# -------------------- psi_63=1.5, s=5.4, steep_vul ------------------------
psi_63 = 1.5  # Pressure at which there is 64% loss of conductivity, MPa
w_exp = 5.4  # Weibull exponent
Kmax = 2  # Maximum plant stem water leaf area-averaged conductivity, mmol/m2/s/MPa
reversible = 1

res = root(psil_crit_val, psi_x_vals + 0.000001,
                                args=(psi_x_vals, psi_sat, gamma, b, d_r, z_r, RAI, lai, Kmax, psi_63, w_exp),
                                method='hybr')
psi_l_vals_steep_vul = res.get('x')

res_psir = root(lambda psi_r: Kmax * psi_63 * gammafunc(1 / w_exp) / w_exp *
                                  (gammaincc(1 / w_exp, (psi_r / psi_63) ** w_exp) -
                                   gammaincc(1 / w_exp, (psi_l_vals_steep_vul / psi_63) ** w_exp)) -
                                  soil_root * (psi_r - psi_x_vals),
                    (psi_l_vals_steep_vul + psi_x_vals) / 2,
               method='hybr')
psi_r_vals_steep_vul = res_psir.get('x')

trans_vals_steep_vul = soil_root * (psi_r_vals_steep_vul - psi_x_vals)
k_max_vals_steep_vul = grl_val(psi_x_vals, psi_63, w_exp, Kmax) * soil_root /\
             (grl_val(psi_x_vals, psi_63, w_exp, Kmax) + soil_root)
k_crit_vals_steep_vul = 0.05 * k_max_vals
trans_max_interp_steep_vul = interp1d(psi_x_vals, trans_vals_steep_vul, kind='cubic')
psi_l_interp_steep_vul = interp1d(psi_x_vals, psi_l_vals_steep_vul, kind='cubic')
psi_r_interp_steep_vul = interp1d(psi_x_vals, psi_r_vals_steep_vul, kind='cubic')

# gRL_3_3 = plant_cond(psi_r, psi_l, psi_63, w_exp, Kmax, reversible)
# gRL_3_3[np.abs(psi_x - psi_r) < 0.01] = 0
grl_psil = grl_val(psi_l_vals_steep_vul, psi_63, w_exp, Kmax)
grl_psir = grl_val(psi_r_vals_steep_vul, psi_63, w_exp, Kmax)
gRL_steep_vul = grl_psil * soil_root / (grl_psir - grl_psil + soil_root)
# -------------------- psi_63=4.3, s=2, Kmax=2 ------------------------
psi_63 = 4.3  # Pressure at which there is 64% loss of conductivity, MPa
w_exp = 2  # Weibull exponent
Kmax = 2  # Maximum plant stem water leaf area-averaged conductivity, mmol/m2/s/MPa
reversible = 1



res = root(psil_crit_val, psi_x_vals + 0.000001,
                                args=(psi_x_vals, psi_sat, gamma, b, d_r, z_r, RAI, lai, Kmax, psi_63, w_exp),
                                method='hybr')
psi_l_vals_grad_res = res.get('x')

res_psir = root(lambda psi_r: Kmax * psi_63 * gammafunc(1 / w_exp) / w_exp *
                                  (gammaincc(1 / w_exp, (psi_r / psi_63) ** w_exp) -
                                   gammaincc(1 / w_exp, (psi_l_vals_grad_res / psi_63) ** w_exp)) -
                                  soil_root * (psi_r - psi_x_vals),
                    (psi_l_vals_grad_res + psi_x_vals) / 2,
               method='hybr')
psi_r_vals_grad_res = res_psir.get('x')

trans_vals_grad_res = soil_root * (psi_r_vals_grad_res - psi_x_vals)
k_max_vals_grad_res = grl_val(psi_x_vals, psi_63, w_exp, Kmax) * soil_root /\
             (grl_val(psi_x_vals, psi_63, w_exp, Kmax) + soil_root)
k_crit_vals_grad_res = 0.05 * k_max_vals
trans_max_interp_grad_res = interp1d(psi_x_vals, trans_vals_grad_res, kind='cubic')
psi_l_interp_grad_res = interp1d(psi_x_vals, psi_l_vals_grad_res, kind='cubic')
psi_r_interp_grad_res = interp1d(psi_x_vals, psi_r_vals_grad_res, kind='cubic')

# gRL_3_3 = plant_cond(psi_r, psi_l, psi_63, w_exp, Kmax, reversible)
# gRL_3_3[np.abs(psi_x - psi_r) < 0.01] = 0
grl_psil = grl_val(psi_l_vals_grad_res, psi_63, w_exp, Kmax)
grl_psir = grl_val(psi_r_vals_grad_res, psi_63, w_exp, Kmax)
gRL_grad_res = grl_psil * soil_root / (grl_psir - grl_psil + soil_root)

fig, ax = plt.subplots()
# gSR_plot, = ax.semilogy(-psi_x_vals, gSR , 'r', linewidth=2)
# gRL_steep_res_plot, = ax.semilogy(-psi_x_vals, gRL_steep_res, 'k', linewidth=2)
# gRL_steep_vul_plot, = ax.semilogy(-psi_x_vals, gRL_steep_vul, 'k--', linewidth=2)
# gRL_grad_res_plot, = ax.semilogy(-psi_x_vals, gRL_grad_res, 'k:', linewidth=2)
# ax.set_ylim(1e-3, 1e0 )
# ax.set_xlim(-1.5, 0)
gSR_plot, = ax.plot(-psi_x_vals, gSR , 'r', linewidth=2)
gRL_steep_res_plot, = ax.plot(-psi_x_vals, gRL_steep_res, 'k', linewidth=2)
gRL_steep_vul_plot, = ax.plot(-psi_x_vals, gRL_steep_vul, 'k--', linewidth=2)
gRL_grad_res_plot, = ax.plot(-psi_x_vals, gRL_grad_res, 'k:', linewidth=2)
ax.set_ylim(0, 0.125)
# ax.set_xlim(-1.5, 0)
plt.setp(ax.get_xticklabels(), FontSize=12)
plt.setp(ax.get_yticklabels(), FontSize=12)

# ax.set_xlabel("Soil Water Potential, $\psi_x$, MPa", FontSize=14)
ax.set_ylabel("Critical conductance, $g_{crit}$, mmol m$^{-2}$ MPa$^{-1}$ s$^{-1}$", FontSize=11)
legend1 = ax.legend((gSR_plot, gRL_steep_res_plot),
                   ('$g_{sr}$', '$g_{rl}$'),
                    fontsize='large', loc=6, title='Color')
ax.add_artist(legend1)
legend2 = ax.legend((gRL_steep_res_plot, gRL_steep_vul_plot, gRL_grad_res_plot),
                    ('Steep and resistant', 'Gradual and resistant', 'Steep and vulnerable'),
                    fontsize='large', loc=2, title="Vulnerability curve description")
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

ax2.set_ylabel('Critical water potential, $\psi_{crit}$, MPa', FontSize=14)
psi_l_steep_res, = ax2.plot(-psi_x_vals, -psi_l_interp_steep_res(psi_x_vals), 'g', linewidth=2)
psi_l_steep_vul, = ax2.plot(-psi_x_vals, -psi_l_interp_steep_vul(psi_x_vals), 'g--', linewidth=2)
psi_l_grad_res, = ax2.plot(-psi_x_vals, -psi_l_interp_grad_res(psi_x_vals), 'g:', linewidth=2)

psi_r_steep_res, = ax2.plot(-psi_x_vals, -psi_r_interp_steep_res(psi_x_vals), 'b', linewidth=2)
psi_r_steep_vul, = ax2.plot(-psi_x_vals, -psi_r_interp_steep_vul(psi_x_vals), 'b--', linewidth=2)
psi_r_grad_res, = ax2.plot(-psi_x_vals, -psi_r_interp_grad_res(psi_x_vals), 'b:', linewidth=2)

plt.setp(ax2.get_xticklabels(), FontSize=12)
plt.setp(ax2.get_yticklabels(), FontSize=12)

# ax2.set_ylim(-7, 0)
# ax2.set_xlim(-1.5, 0)

legend1 = ax2.legend((psi_l_steep_res, psi_r_steep_res),
                   ('$\psi_{l,crit}$', '$\psi_{r,crit}$'),
                    fontsize='large', loc=4, title='Color')

# fig2.savefig('../psil_psir_psix.pdf', bbox_inches='tight')
# ax2.add_artist(legend1)
# legend2 = ax2.legend((trans_3_3, trans_15_3, trans_3_1),
#                    ('$\psi_{63}=-3$, $s=3$, $g_{rl,max}=2$',
#                     '$\psi_{63}=-1.5$, $s=3$, $g_{rl,max}=2$',
#                     '$\psi_{63}=-1.4$, $s=1.3$, $g_{rl,max}=8$'),
#                     fontsize='small', loc=9, title='Line style')

figE, axE = plt.subplots()
# trans_steep_res, = axE.semilogy(-psi_x_vals,
#              trans_max_interp_steep_res(psi_x_vals), 'k', linewidth=2)
# trans_steep_vul, = axE.semilogy(-psi_x_vals,
#              trans_max_interp_steep_vul(psi_x_vals), 'k--', linewidth=2)
# trans_grad_res, = axE.semilogy(-psi_x_vals,
#              trans_max_interp_grad_res(psi_x_vals), 'k:', linewidth=2)
trans_steep_res, = axE.plot(-psi_x_vals,
             trans_max_interp_steep_res(psi_x_vals), 'k', linewidth=2)
trans_steep_vul, = axE.plot(-psi_x_vals,
             trans_max_interp_steep_vul(psi_x_vals), 'k--', linewidth=2)
trans_grad_res, = axE.plot(-psi_x_vals,
             trans_max_interp_grad_res(psi_x_vals), 'k:', linewidth=2)
# axE.set_ylim(1e-3, 1e0)
axE.set_ylabel("Critical transpiration rate, $E_{crit}$, mmol m$^{-2}$ s$^{-1}$", FontSize=11)
axE.set_xlabel("Soil Water Potential, $\psi_x$, MPa", FontSize=14)

plt.setp(axE.get_xticklabels(), FontSize=12)
plt.setp(axE.get_yticklabels(), FontSize=12)
# axE.set_xlim(-1.5, 0)
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
# def lam(trans, ca, cp, VPD, k1, k2):
#     gl_vals = trans / (VPD * 1.6)
#     part11 = ca ** 2 * gl_vals + 2 * cp * k1 - ca * (k1 - 2 * gl_vals * k2) + k2 * (k1 + gl_vals * k2)
#     part12 = np.sqrt(4 * (cp - ca) * gl_vals * k1 + (k1 + gl_vals * (ca + k2)) ** 2)
#     part1 = part11 / part12
#
#     part2 = ca + k2
#
#     part3 = 2 * VPD * 1.6
#     # returns lambda in mol/mol
#     return (part2 - part1) / part3
#
# k1 = 1  # mol m-2 d-1
# k2 = 1.8e-4  # mol mol-1
# cp = 25e-6  # mol mol-1
# ca = 350e-6  # mol mol-1
#
# VPD = 8e-3  # mol mol-1
# lam_val_3_3_8 = lam(trans_max_interp_3_3(psi_x), ca, cp, VPD, k1, k2)
# lam_val_15_3_8 = lam(trans_max_interp_15_3(psi_x), ca, cp, VPD, k1, k2)
# lam_val_3_1_8 = lam(trans_max_interp_3_1(psi_x), ca, cp, VPD, k1, k2)
# lam_up_8 = (ca - cp) * 1e3 / 8e-3 / 1.6
#
# VPD = 16e-3  # mol mol-1
# lam_val_3_3_16 = lam(trans_max_interp_3_3(psi_x), ca, cp, VPD, k1, k2)
# lam_val_15_3_16 = lam(trans_max_interp_15_3(psi_x), ca, cp, VPD, k1, k2)
# lam_val_3_1_16 = lam(trans_max_interp_3_1(psi_x), ca, cp, VPD, k1, k2)
# lam_up_16 = (ca - cp) * 1e3 / 16e-3 / 1.6
#
# VPD = 32e-3  # mol mol-1
# lam_val_3_3_32 = lam(trans_max_interp_3_3(psi_x), ca, cp, VPD, k1, k2)
# lam_val_15_3_32 = lam(trans_max_interp_15_3(psi_x), ca, cp, VPD, k1, k2)
# lam_val_3_1_32 = lam(trans_max_interp_3_1(psi_x), ca, cp, VPD, k1, k2)
# lam_up_32 = (ca - cp) * 1e3 / 32e-3 / 1.6

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
