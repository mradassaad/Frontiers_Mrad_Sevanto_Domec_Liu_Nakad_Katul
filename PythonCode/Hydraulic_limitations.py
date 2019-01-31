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

gSR = gSR_val(x, gamma, b, d_r, z_r, RAI)
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
for xx in x:
    OptRes = minimize(trans_opt, psi_x[i], args=(x[i], psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI, reversible))
    psi_l[i] = OptRes.x
    trans_res = transpiration(OptRes.x, x[i], psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI, reversible)
    trans_max[i] = trans_res[0]
    psi_r[i] = trans_res[1]
    i += 1

trans_max_interp_3_3 = interp1d(psi_x, trans_max, kind='cubic')
psi_l_interp_3_3 = interp1d(psi_x, psi_l, kind='cubic')
psi_r_interp_3_3 = interp1d(psi_x, psi_r, kind='cubic')

gRL_3_3 = plant_cond(psi_r, psi_l, psi_63, w_exp, Kmax, reversible)
gRL_3_3[np.abs(psi_x - psi_r) < 0.01] = 0
# -------------------- psi_63=1.5, s=3 ------------------------
psi_63 = 1.5  # Pressure at which there is 64% loss of conductivity, MPa
w_exp = 4  # Weibull exponent
Kmax = 2e-3 * unit0  # Maximum plant stem water leaf area-averaged conductivity, mol/m2/d/MPa
reversible = 1

psi_l = np.zeros(x.shape)
psi_r = np.zeros(x.shape)
trans_max = np.zeros(x.shape)
i = 0
for xx in x:
    OptRes = minimize(trans_opt, psi_x[i], args=(x[i], psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI, reversible))
    psi_l[i] = OptRes.x
    trans_res = transpiration(OptRes.x, x[i], psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI, reversible)
    trans_max[i] = trans_res[0]
    psi_r[i] = trans_res[1]
    i += 1

trans_max_interp_15_3 = interp1d(psi_x, trans_max, kind='cubic')
psi_l_interp_15_3 = interp1d(psi_x, psi_l, kind='cubic')
psi_r_interp_15_3 = interp1d(psi_x, psi_r, kind='cubic')

gRL_15_3 = plant_cond(psi_r, psi_l, psi_63, w_exp, Kmax, reversible)
gRL_15_3[np.abs(psi_x - psi_r) < 0.01] = 0
# -------------------- psi_63=1, s=1.2, Kmax=6 ------------------------
psi_63 = 1  # Pressure at which there is 64% loss of conductivity, MPa
w_exp = 1  # Weibull exponent
Kmax = 8e-3 * unit0  # Maximum plant stem water leaf area-averaged conductivity, mol/m2/d/MPa
reversible = 1

psi_l = np.zeros(x.shape)
psi_r = np.zeros(x.shape)
trans_max = np.zeros(x.shape)
i = 0
for xx in x:
    OptRes = minimize(trans_opt, psi_x[i], args=(x[i], psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI, reversible))
    psi_l[i] = OptRes.x
    trans_res = transpiration(OptRes.x, x[i], psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI, reversible)
    trans_max[i] = trans_res[0]
    psi_r[i] = trans_res[1]
    i += 1

trans_max_interp_3_1 = interp1d(psi_x, trans_max, kind='cubic')
psi_l_interp_3_1 = interp1d(psi_x, psi_l, kind='cubic')
psi_r_interp_3_1 = interp1d(psi_x, psi_r, kind='cubic')

gRL_3_1 = plant_cond(psi_r, psi_l, psi_63, w_exp, Kmax, reversible)
gRL_3_1[np.abs(psi_x - psi_r) < 0.01] = 0

fig, ax = plt.subplots()
gSR_plot, = ax.semilogy(-psi_x, gSR * 1e3 / unit0, 'r')
gRL_3_3_plot, = ax.semilogy(-psi_x, gRL_3_3 * 1e3 / unit0, 'k')
gRL_15_3_plot, = ax.semilogy(-psi_x, gRL_15_3 * 1e3 / unit0, 'k--')
gRL_3_1_plot, = ax.semilogy(-psi_x, gRL_3_1 * 1e3 / unit0, 'k:')
ax.set_ylim(1e-2, 1e0 + 4)
ax.set_xlim(-1.4, 0)
plt.setp(ax.get_xticklabels(), FontSize=12)
plt.setp(ax.get_yticklabels(), FontSize=12)

ax.set_xlabel("Soil Water Potential, $\psi_x$, MPa", FontSize=14)
ax.set_ylabel("Maximum conductance, $g_{max}$, mmol m$^{-2}$ MPa$^{-1}$ s$^{-1}$", FontSize=12)
legend1 = ax.legend((gSR_plot, gRL_3_3_plot),
                   ('$g_{sr}$', '$g_{rl}$'),
                    fontsize='large', loc=2, title='Color')
ax.add_artist(legend1)
legend2 = ax.legend((gRL_3_3_plot, gRL_15_3_plot, gRL_3_1_plot),
                   ('$\psi_{63}=-3$, $s=3$, $g_{rl,max}=2$',
                    '$\psi_{63}=-1.5$, $s=3$, $g_{rl,max}=2$',
                    '$\psi_{63}=-1$, $s=1$, $g_{rl,max}=8$'),
                    fontsize='small', loc=9, title='Line style')

# fig.savefig('../g_psix.pdf', bbox_inches='tight')

fig2, ax2 = plt.subplots()
trans_3_3, = ax2.semilogy(-psi_x,
             trans_max_interp_3_3(psi_x) * 1e3 / unit0, 'k', linewidth=3)
trans_15_3, = ax2.semilogy(-psi_x,
             trans_max_interp_15_3(psi_x) * 1e3 / unit0, 'k--', linewidth=3)
trans_3_1, = ax2.semilogy(-psi_x,
             trans_max_interp_3_1(psi_x) * 1e3 / unit0, 'k:', linewidth=3)
ax2.set_xlim(-1.4, 0)
ax2.set_xlabel('Soil water potential, $\psi_x$, MPa', FontSize=14)
ax2.set_ylabel('Maximum transpiration, $E_{max}$, mmol m$^{-2}$ s$^{-1}$', FontSize=12)

ax2_2 = ax2.twinx()
ax2_2.set_ylabel('Water potential, $\psi$, MPa', FontSize=14)
psi_l_3_3, = ax2_2.plot(-psi_x, -psi_l_interp_3_3(psi_x), 'g', alpha=0.7)
psi_l_15_3, = ax2_2.plot(-psi_x, -psi_l_interp_15_3(psi_x), 'g--', alpha=0.7)
psi_l_3_1, = ax2_2.plot(-psi_x, -psi_l_interp_3_1(psi_x), 'g:', alpha=0.7)

psi_r_3_3, = ax2_2.plot(-psi_x, -psi_r_interp_3_3(psi_x), 'b', alpha=0.7)
psi_r_15_3, = ax2_2.plot(-psi_x, -psi_r_interp_15_3(psi_x), 'b--', alpha=0.7)
psi_r_3_1, = ax2_2.plot(-psi_x, -psi_r_interp_3_1(psi_x), 'b:', alpha=0.7)

ax2_2.set_ylim(-7, 0)

legend1 = ax2.legend((trans_3_3, psi_l_3_3, psi_r_3_3),
                   ('$E$', '$\psi_l$', '$\psi_r$'),
                    fontsize='large', loc=2, title='Color')

# fig2.savefig('../E_psil_psir_psix.pdf', bbox_inches='tight')
# ax2.add_artist(legend1)
# legend2 = ax2.legend((trans_3_3, trans_15_3, trans_3_1),
#                    ('$\psi_{63}=-3$, $s=3$, $g_{rl,max}=2$',
#                     '$\psi_{63}=-1.5$, $s=3$, $g_{rl,max}=2$',
#                     '$\psi_{63}=-1.4$, $s=1.3$, $g_{rl,max}=8$'),
#                     fontsize='small', loc=9, title='Line style')

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

VPD = 16e-3  # mol mol-1
lam_val_3_3_16 = lam(trans_max_interp_3_3(psi_x), ca, cp, VPD, k1, k2)
lam_val_15_3_16 = lam(trans_max_interp_15_3(psi_x), ca, cp, VPD, k1, k2)
lam_val_3_1_16 = lam(trans_max_interp_3_1(psi_x), ca, cp, VPD, k1, k2)

VPD = 32e-3  # mol mol-1
lam_val_3_3_32 = lam(trans_max_interp_3_3(psi_x), ca, cp, VPD, k1, k2)
lam_val_15_3_32 = lam(trans_max_interp_15_3(psi_x), ca, cp, VPD, k1, k2)
lam_val_3_1_32 = lam(trans_max_interp_3_1(psi_x), ca, cp, VPD, k1, k2)

fig3, ax3 = plt.subplots()

lam_plot_3_3_8, = ax3.plot(-psi_x, lam_val_3_3_8 * 1e3, 'k')
lam_plot_15_3_8, = ax3.plot(-psi_x, lam_val_15_3_8 * 1e3, 'k--')
lam_plot_3_1_8, = ax3.plot(-psi_x, lam_val_3_1_8 * 1e3, 'k:')

lam_plot_3_3_16, = ax3.plot(-psi_x, lam_val_3_3_16 * 1e3, 'b')
lam_plot_15_3_16, = ax3.plot(-psi_x, lam_val_15_3_16 * 1e3, 'b--')
lam_plot_3_1_16, = ax3.plot(-psi_x, lam_val_3_1_16 * 1e3, 'b:')

lam_plot_3_3_32, = ax3.plot(-psi_x, lam_val_3_3_32 * 1e3, 'g')
lam_plot_15_3_32, = ax3.plot(-psi_x, lam_val_15_3_32 * 1e3, 'g--')
lam_plot_3_1_32, = ax3.plot(-psi_x, lam_val_3_1_32 * 1e3, 'g:')

ax3.set_xlim(-1.4, 0)

ax3.set_xlabel('Soil water potential, $\psi_x$, MPa', FontSize=14)
ax3.set_ylabel('Lower bound on $\lambda$, $\lambda_{lower}$, mmol mol$^{-1}$', FontSize=12)

legend1 = ax3.legend((lam_plot_3_3_8, lam_plot_3_3_16, lam_plot_3_3_32),
                   ('VPD=8 mmol mol$^{-1}$', 'VPD=16 mmol mol$^{-1}$', 'VPD=32 mmol mol$^{-1}$'),
                    fontsize='small', loc=1, title='Color')
ax3.add_artist(legend1)
legend2 = ax3.legend((lam_plot_3_3_8, lam_plot_15_3_8, lam_plot_3_1_8),
                   ('$\psi_{63}=-3$, $s=3$, $g_{rl,max}=2$',
                    '$\psi_{63}=-1.5$, $s=3$, $g_{rl,max}=2$',
                    '$\psi_{63}=-1$, $s=1$, $g_{rl,max}=8$'),
                    fontsize='small', loc=3, title='Line style')

# fig3.savefig('../lam_psix.pdf', bbox_inches='tight')
# lam_upper_0 = np.ones(lam_val_1.shape) * (ca - env_data[0, 0]) / env_data[1, 0] / alpha
# lam_upper_1 = np.ones(lam_val_1.shape) * (ca - env_data[0, 1]) / env_data[1, 1] / alpha
# lam_upper_2 = np.ones(lam_val_1.shape) * (ca - env_data[0, 2]) / env_data[1, 2] / alpha
# lam_upper_3 = np.ones(lam_val_1.shape) * (ca - env_data[0, 3]) / env_data[1, 3] / alpha

# fig, ax = plt.subplots()
# line_low_0, = plt.plot(-psi_x_vals, lam_val_0 * unit1, 'r')
# line_low_1, = plt.plot(-psi_x_vals, lam_val_1 * unit1, 'k')
# # plt.plot(psi_x_vals, lam_val_2 * unit1)
# line_low_3, = plt.plot(-psi_x_vals, lam_val_3 * unit1, 'b')
# line_high_0, = plt.plot(-psi_x_vals, lam_upper_0 * unit1, 'r--')
# line_high_1, = plt.plot(-psi_x_vals, lam_upper_1 * unit1, 'k--')
# line_high_3, = plt.plot(-psi_x_vals, lam_upper_3 * unit1, 'b--')
# ax.set_xlabel("Soil water potential, $\psi_x$, MPa", FontSize=16)
# ax.set_ylabel("$\lambda$ lower and upper bounds, mol.m$^{-2}$", FontSize=16)
# plt.setp(ax.get_xticklabels(), FontSize=12)
# plt.setp(ax.get_yticklabels(), FontSize=12)
# legend1 = ax.legend((line_low_0, line_low_1, line_low_3),
#                    ('06:00', '12:00', '18:00'), fontsize='large', loc=3, title='Color')
# ax.add_artist(legend1)
# legend2 = ax.legend((line_low_1, line_high_1),
#                    ('$\lambda_{lower}$', '$\lambda_{upper}$'), fontsize='large', loc=8, title='Line Style')

# plt.savefig('limits_time.pdf', bbox_inches='tight')


# psi_63 = 2
# i = 0
# for x in xvals:
#     OptRes = minimize(trans_opt, psi_x_vals[i], args=(xvals[i], psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI))
#     psi_l_vals[i] = OptRes.x
#     trans_res = transpiration(OptRes.x, xvals[i], psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI)
#     trans_vals[i] = trans_res[0]
#     psi_r_vals[i] = trans_res[1]
#     i += 1
#
# lam_val_low = lam(trans_vals, ca, alpha, env_data[0, 1], env_data[1, 1], env_data[2, 1], env_data[3, 1])
#
# psi_63 = 3
# w_exp = 1
# i = 0
# for x in xvals:
#     OptRes = minimize(trans_opt, psi_x_vals[i], args=(xvals[i], psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI))
#     psi_l_vals[i] = OptRes.x
#     trans_res = transpiration(OptRes.x, xvals[i], psi_sat, gamma, b, psi_63, w_exp, Kmax, d_r, z_r, RAI)
#     trans_vals[i] = trans_res[0]
#     psi_r_vals[i] = trans_res[1]
#     i += 1
#
# lam_val_high = lam(trans_vals, ca, alpha, env_data[0, 1], env_data[1, 1], env_data[2, 1], env_data[3, 1])
# #
# psi_63 = 3
# w_exp = 1
# i = 0
# for x in xvals:
#     OptRes = minimize(trans_opt, psi_x_vals[i], args=(psi_x_vals[i], psi_63, w_exp, Kmax))
#     psi_l_vals[i] = OptRes.x
#     trans_vals[i] = transpiration(OptRes.x, psi_x_vals[i], psi_63, w_exp, Kmax)
#     i += 1
#
# lam_val_R = lam(trans_vals, ca, alpha, env_data[0, 2], env_data[1, 2], env_data[2, 2], env_data[3, 2])
#
# psi_63 = 3
# w_exp = 3
# i = 0
# for x in xvals:
#     OptRes = minimize(trans_opt, psi_x_vals[i], args=(psi_x_vals[i], psi_63, w_exp, Kmax))
#     psi_l_vals[i] = OptRes.x
#     trans_vals[i] = transpiration(OptRes.x, psi_x_vals[i], psi_63, w_exp, Kmax)
#     i += 1
#
# lam_val_S = lam(trans_vals, ca, alpha, env_data[0, 2], env_data[1, 2], env_data[2, 2], env_data[3, 2])
#
# lam_upper_1 = np.ones(lam_val_1.shape) * (ca - env_data[0, 1]) / env_data[1, 1] / alpha
# lam_upper_2 = np.ones(lam_val_1.shape) * (ca - env_data[0, 2]) / env_data[1, 2] / alpha
# lam_upper_3 = np.ones(lam_val_1.shape) * (ca - env_data[0, 3]) / env_data[1, 3] / alpha
#
# xax = psi_x_vals
#
#
# fig, ax = plt.subplots()
# # plt.plot(xvals, lam_val_1*unit1, 'b-', label='$\psi_{63}=3$, 06:00')
# line_ref, = ax.plot(xax, lam_val_2*unit1, 'r-', label='$\psi_{63}=3$, $s=2$')
# # plt.plot(xvals, lam_val_3*unit1, 'k-', label='$\psi_{63}=3$, 18:00')
# line_low, = ax.plot(xax, lam_val_low*unit1, 'r--', label='$\psi_{63}=2.5$, $s=2$')
# line_high, = ax.plot(xax, lam_val_high*unit1, 'r:', label='$\psi_{63}=4$, $s=2$')
# line_R, = ax.plot(xax, lam_val_R*unit1, 'b-', label='$\psi_{63}=3$, $s=1$')
# line_S, = ax.plot(xax, lam_val_S*unit1, 'k-', label='$\psi_{63}=3$, $s=3$')
# # line_upper, = plt.plot(xax, lam_upper_2*unit1, 'g^')
# ax.set_xlabel("Soil water potential, $\psi_x$", FontSize=16)
# ax.set_ylabel("$\lambda$ lower bound, $\lambda_{lower}$, $mol.m^{-2}$", FontSize=16)
# plt.setp(ax.get_xticklabels(), FontSize=12)
# plt.setp(ax.get_yticklabels(), FontSize=12)
# legend1 = ax.legend((line_ref, line_low, line_high),
#                    ('$\psi_{63}=3$', '$\psi_{63}=2.5$', '$\psi_{63}=4$'), fontsize='large', loc=2, title='s=2')
# ax.add_artist(legend1)
# legend2 = ax.legend((line_ref, line_R, line_S),
#                    ('$s=2$', '$s=1$', '$s=3$'), fontsize='large', loc=6, title='$\psi_{63}=3$')
# ax.set_xlim(0, 2)
# ax.set_ylim(0, 5)
# plt.grid(False)



# plt.figure()
# plt.plot(xvals, lam_upper_1*unit1, 'b-', label='06:00')
# plt.plot(xvals, lam_upper_2*unit1, 'r--', label='12:00')
# plt.plot(xvals, lam_upper_3*unit1, 'y:', label='18:00')
# plt.xlabel("Soil Moisture, x")
# plt.ylabel("$\lambda$ upper bound, $\lambda_{upper}$, $mol.m^{-2}$")
# plt.legend()